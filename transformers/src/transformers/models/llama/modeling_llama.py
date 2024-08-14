# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import math
import random
from typing import List, Optional, Tuple, Union
import copy
import time

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_available,
    logging,
    replace_return_docstrings,
)
from .configuration_llama import LlamaConfig

import time

if is_flash_attn_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


def _get_unpad_data(padding_mask):
    seqlens_in_batch = padding_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(padding_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)).cuda()
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype).cuda()

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.cuda()
    sin = sin.cuda()
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_id: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.layer_id = layer_id
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            padding_mask: Optional[torch.LongTensor] = None,
            memory_l=None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)     ##f (2, 32, 128, 64)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)       ##f (2, 4, 128, 64)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)    ##f (2, 4, 128, 64)
        # keep un_rotated_key_states =================
        un_rotated_key_states = key_states
        wo_cache_value_states = value_states
        
        ##f for num_key_value_groups > 1
        un_rotated_key_states = repeat_kv(un_rotated_key_states, self.num_key_value_groups)     ##f (2, 32, 128, 64)
        wo_cache_value_states = repeat_kv(wo_cache_value_states, self.num_key_value_groups)
        empty_cache_flag = True if past_key_value is None else False
        ##f
        
        # if (self.num_key_value_groups != 1) or (past_key_value is not None):
        #     raise NotImplementedError

        kv_seq_len = key_states.shape[-2]       ##f 128
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # print(cos.shape, sin.shape, position_ids.shape, kv_seq_len, value_states.shape)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)       ##f (2, 32, 128, 64)
        value_states = repeat_kv(value_states, self.num_key_value_groups)       ##f (2, 32, 128, 64)

        ##f 构建local dilated attention mask =======================================================
        def create_dilated_mask(size, dilation):
            # 创建一个size x size大小的矩阵，初始全为0
            mask = torch.zeros(size, size)
            # 将除了对角线以外，与对角线有dilation间隔的位置设置为1
            for i in range(size):
                for j in range(i + 1):
                    if (i - j) % dilation == 0:
                        mask[i, j] = 1
            return mask.to(torch.int8)

        if self.config.local_stride[self.layer_id] and self.config.window_length[self.layer_id] > 0 and self.config.stride[self.layer_id] > 1:
            if hasattr(self, 'stride_mask') and self.stride_mask.shape[-1] == q_len:
                attention_mask = self.stride_mask.to(attention_mask.device)
            else:
                mask_for_seq = create_dilated_mask(q_len, self.config.stride[self.layer_id]).view(1, 1, q_len,
                                                                                                  q_len)

                attn_mask = 1 - mask_for_seq.to(value_states.dtype)
                attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), torch.finfo(value_states.dtype).min)
                attn_mask = attn_mask.repeat(bsz, 1, 1, 1) if attn_mask.shape[0] == 1 else attn_mask
                self.stride_mask = attn_mask
                attention_mask = self.stride_mask.to(attention_mask.device)

        # ==============================================================================================

        # attn from external memory ======
        memory_start_time = time.time()
        retrieved = memory_l.retrieve(query_states)
        #retrieved = None
        past_key_value = memory_l.update_memory_l(un_rotated_key_states, wo_cache_value_states, past_key_value, empty_cache_flag)
        memory_end_time = time.time() - memory_start_time

        if retrieved is not None:
            (window_keys, window_values, window_mask), (topk_keys, topk_values) = retrieved
        else:
            window_keys, window_values, window_mask, topk_keys, topk_values= (None,) * 5

        def attn_with_memory(query_states, key_states, value_states, attention_mask,
                             window_keys, window_values, window_mask,
                             topk_keys, topk_values):

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_weights = attn_weights + attention_mask

            if window_keys is not None:

                window_attn_weights = torch.matmul(query_states, window_keys.transpose(2, 3)) / math.sqrt(
                    self.head_dim) + window_mask
                attn_weights = torch.cat((window_attn_weights, attn_weights), dim=-1)  # [b, n, l, (k+w+l)]
                value_states = torch.cat((window_values, value_states), dim=2)

            if topk_keys is not None:
                topk_attn_weights = torch.matmul(query_states.unsqueeze(-2), topk_keys.transpose(-1, -2)).squeeze(
                    -2) / math.sqrt(self.head_dim)
                # [bnl1d, bnldk -> [bnl1k]]
                attn_weights = torch.cat((topk_attn_weights, attn_weights), dim=-1)

            # if attention_mask is not None:  # Attention_mask added to attention_weights and window_attn_weights separately
            #     if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            #         logger.warning_once(
            #             f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            #         )
            #     attn_weights = attn_weights + attention_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            if topk_keys is not None:
                # print(topk_keys.shape, attn_weights.shape, topk_values.shape, value_states.shape, 'topk------')
                n_topk = topk_keys.shape[-2]
                topk_weights, attn_weights = attn_weights[..., :n_topk], attn_weights[..., n_topk:]
                topk_output = torch.matmul(topk_weights.unsqueeze(-2), topk_values).squeeze(
                    -2)  # [bnl1k, bnlkd -> bnl1d]
                attn_output = torch.matmul(attn_weights, value_states)
                attn_output = attn_output + topk_output
            else:
                attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2).contiguous()

            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

            if self.config.pretraining_tp > 1:
                attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
                o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
                attn_output = sum(
                    [F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
            else:
                attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_key_value

        if self.training:  # Always checkpointing
            attn_output, attn_weights, past_key_value = torch.utils.checkpoint.checkpoint(
                attn_with_memory, query_states, key_states, value_states, attention_mask,
                window_keys, window_values, window_mask,
                topk_keys, topk_values, use_reentrant=True,
            )
        else:
            attn_output, attn_weights, past_key_value = attn_with_memory(
                query_states, key_states, value_states, attention_mask,
                window_keys, window_values, window_mask,
                topk_keys, topk_values)

        return attn_output, attn_weights, past_key_value


class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # LlamaFlashAttention2 attention does not support output_attentions
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dime x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # TODO: llama does not have dropout in the config??
        # It is recommended to use dropout with FA according to the docs
        # when training.
        dropout_rate = 0.0  # if not self.training else self.attn_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            logger.warning_once(
                "The input hidden states seems to be silently casted in float32, this might be related to"
                " the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                " float16."
            )

            query_states = query_states.to(torch.float16)
            key_states = key_states.to(torch.float16)
            value_states = value_states.to(torch.float16)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, padding_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
            self, query_states, key_states, value_states, padding_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            padding_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        # Contains at least one padding token in the sequence
        if padding_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, padding_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=True,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=True
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, padding_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(padding_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            padding_mask = padding_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, padding_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_id: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = (
            LlamaAttention(config=config, layer_id=layer_id)
            if not getattr(config, "_flash_attn_2_enabled", False)
            else LlamaFlashAttention2(config=config)
        )
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            padding_mask: Optional[torch.LongTensor] = None,
            memory_l=None
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
            memory_l=memory_l
        )
        hidden_states = residual + hidden_states

        # Fully Connected

        if self.training:  # Always applies gradient checkpointing
            def custom_forward(hidden_states):
                residual = hidden_states
                hidden_states = self.post_attention_layernorm(hidden_states)
                hidden_states = self.mlp(hidden_states)
                hidden_states = residual + hidden_states
                return hidden_states

            hidden_states = torch.utils.checkpoint.checkpoint(
                custom_forward, hidden_states,
            )
            # print('checkpointing -------------------------------', hidden_states.shape)
        else:
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlamaModel):
            module.gradient_checkpointing = value


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_id) for layer_id in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.memory = ExternalMemory(config) if 'memory_layer' in config.__dict__ else None #

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # print('forward', input_ids.shape)
        if past_key_values is not None:
            for pkv in past_key_values:
                if pkv is None:
                    past_key_values = None
                    break  # For generation
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
            padding_mask = None
        else:
            if 0 in attention_mask:
                padding_mask = attention_mask
            else:
                padding_mask = None

        seq_length, inputs_embeds, attention_mask, position_ids \
            = self.memory.prepare_input_states(inputs_embeds, attention_mask, position_ids, past_key_values_length)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        memory_time = 0
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if False and self.gradient_checkpointing and self.training:  # todo: Do not checkpoint the whole block since the memory layer has states

                def create_custom_forward(module):
                    def custom_forward(hidden_states, attention_mask, position_ids):
                        # None for past_key_value
                        return module(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, padding_mask=padding_mask, memory_l=self.memory[idx])
                    
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer), hidden_states,
                        attention_mask,
                        position_ids
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    padding_mask=padding_mask,
                    memory_l=self.memory[idx]
                )
            # memory_time += memory_end_time

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # hidden_states = self.memory.update_rmt_memory(hidden_states)  # before norm or after norm?
        hidden_states = self.norm(hidden_states)

        # print('before rmt', hidden_states.shape, self.memory.rmt_pad_flag)
        hidden_states = self.memory.update_rmt_memory(hidden_states)
        # print('after update rmt', hidden_states.shape, len(self.memory.rmt_memory))

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels).nanmean()
            # if shift_labels.max() > 0:
            #     print('Effective label!')
            # print(logits.shape, labels.shape, torch.unique(shift_labels), 'model forward')

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
                    logits.device
                )
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class ExternalMemory(nn.Module):
    """
    ExternalMemory that contains 1. k, v memory of each layer; 2. rmt memory of last layer; 3. trainable mem token
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.memory_l = []
        
        for l in range(config.num_hidden_layers):
            config_l = copy.deepcopy(config.__dict__)
            config_l.update({'memory_size': config.memory_size[l], 'window_length': config.window_length[l],
                             'stride': config.stride[l], 'topk': config.topk[l],
                             'pos_type': config.pos_type[l], 'retrieved_length': config.retrieved_length[l],
                             'global_token': config.global_token[l], 'global_pos': config.global_pos[l],
                             'reset': config.reset[l], 'local_stride':config.local_stride[l], 'rmt_size': None, 'rmt_memory_size': None, 'memory_layer': None,
                             'segment_size': config.segment_size})
            config_l = Dict2Class(config_l)
            self.memory_l.append(MemoryLayer(config_l))
        self.rmt_memory = []
        if self.config.rmt_size > 0:
            self.rmt_tokens = nn.Parameter(torch.randn(1, self.config.rmt_size, self.config.hidden_size) * 0.01)
        self.original_seq_length = None

    def update_rmt_memory(self, hidden_states_L):
        # hidden_states_L[b, l, d]
        if self.config.rmt_size == 0:
            return hidden_states_L

        if self.rmt_pad_flag == 3:   # for generation
            self.rmt_memory.append(hidden_states_L[:, -self.config.rmt_size:])
            if len(self.rmt_memory) > self.config.rmt_memory_size:
                del self.rmt_memory[0]
            # delete mem tokens from hidden_states for calculating loss
            hidden_states_L = hidden_states_L[:, -(self.config.rmt_size + self.original_seq_length):-self.config.rmt_size]
        elif self.rmt_pad_flag == 1:  # 左测拼接，删除hidden_states_L开头，不更新rmt_memory
            hidden_states_L = hidden_states_L[:, -(self.original_seq_length):]
        elif self.rmt_pad_flag == 2:  # 不拼接，不删除
            pass
        else:
            raise NotImplementedError

        return hidden_states_L

    def prepare_input_states(self, hidden_states_0, attention_mask, position_ids, past_key_values_length):
        # upldate hidden_states_0, attention_mask, and position_ids for rmt token
        # hidden_states_0[b, l, d]

        self.original_seq_length = hidden_states_0.shape[1]  # for delete mem tokens later, work in multi process?
        for ml in self.memory_l:   # let each memory layer aware of original_seq_length, for dropping rmt tokens
            ml.original_seq_length = self.original_seq_length
            ml.rmt_size = self.config.rmt_size
        # attach mem_token to hidden_states ================
        # print('attn mask', attention_mask.shape, attention_mask.unique())
        # todo: for generation
        if self.original_seq_length == self.config.segment_size and past_key_values_length == 0:  # situation 3
            token_start_pos = 0
            if len(self.rmt_memory) > 0:
                hidden_states_0 = torch.cat((*self.rmt_memory, hidden_states_0,
                                             self.rmt_memory[-1]), dim=1)
                token_start_pos = sum([rm.shape[1] for rm in self.rmt_memory])
            elif self.config.rmt_size > 0:
                hidden_states_0 = torch.cat((hidden_states_0,
                                             self.rmt_tokens.repeat(hidden_states_0.shape[0], 1, 1)), dim=1)
            self.rmt_pad_flag = 3

            position_ids = torch.arange(
                0, hidden_states_0.shape[1], dtype=torch.long, device=hidden_states_0.device
            ).unsqueeze(0).view(-1, hidden_states_0.shape[1])
            attention_mask = torch.cat((
                torch.ones((hidden_states_0.shape[0], token_start_pos), dtype=torch.bool, device=hidden_states_0.device),
                attention_mask,
                torch.ones((hidden_states_0.shape[0], hidden_states_0.shape[1] - attention_mask.shape[1] - token_start_pos), dtype=torch.bool, device=hidden_states_0.device),
            ), dim=1)

        elif self.original_seq_length < self.config.segment_size and past_key_values_length == 0:  # situation 1
            if len(self.rmt_memory) > 0:
                hidden_states_0 = torch.cat((*self.rmt_memory, hidden_states_0), dim=1)
            self.rmt_pad_flag = 1

            position_ids = torch.arange(
                0, hidden_states_0.shape[1], dtype=torch.long, device=hidden_states_0.device
            ).unsqueeze(0).view(-1, hidden_states_0.shape[1])
            print('attn', attention_mask.shape, hidden_states_0.shape)
            if self.original_seq_length == 1:  # after cache is filled and reset to None
                attention_mask = torch.ones((hidden_states_0.shape[0], hidden_states_0.shape[1]), dtype=torch.bool, device=hidden_states_0.device)
            else:
                attention_mask = torch.cat((
                    torch.ones((hidden_states_0.shape[0], hidden_states_0.shape[1] - attention_mask.shape[1]), dtype=torch.bool, device=hidden_states_0.device),
                    attention_mask,
                ), dim=1)

        elif self.original_seq_length == 1 and past_key_values_length > 0:  # situation 2
            self.rmt_pad_flag = 2

            position_ids = torch.tensor([past_key_values_length], dtype=torch.long, device=hidden_states_0.device).reshape(1, 1)
            # attention_mask = torch.ones((hidden_states_0.shape[0], past_key_values_length + 1), dtype=torch.long, device=hidden_states_0.device)
            attention_mask = torch.ones((hidden_states_0.shape[0], hidden_states_0.shape[1]), dtype=torch.bool, device=hidden_states_0.device)
        else:
            raise NotImplementedError

        for ml in self.memory_l:
            ml.rmt_pad_flag = self.rmt_pad_flag

        return hidden_states_0.shape[1], hidden_states_0, attention_mask, position_ids

    def detach_memory(self):
        for m in range(len(self.rmt_memory)):
            self.rmt_memory[m] = self.rmt_memory[m].detach()
        for m in self.memory_l:
            m.detach_memory()

    def reset_memory(self):
        self.rmt_memory = []
        for m in self.memory_l:
            m.reset_memory()

    def __getitem__(self, item):
        return self.memory_l[item]


class MemoryLayer(nn.Module):
    """
    MemoryLayer of specific layer, contains keys and values;
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # print('memory l ', config.memory_size, config.window_length, config.stride, config.topk)
        self.register_buffer('keys', None)    
        self.register_buffer('values', None)
        self.register_buffer('window_mask', None)  # [1, n, l, w]

        
        self.current_memory = 0
        self.segment_size = config.segment_size
        
        # self.lamda = nn.Parameter(torch.zeros(1, 1, config.num_attention_heads, 1))  # No lamda now
        self.chunk_size = []  # e.g. 512; the length of each chunk
        self._init_rope()

        self.kv_cache_unrotated = None
        

    def reset_memory(self):
        self.register_buffer('keys', None)
        self.register_buffer('values', None)
        self.register_buffer('window_mask', None)
        self.current_memory = 0
        self.chunk_size = []
        self.kv_cache_unrotated = None

    def detach_memory(self):
        if self.keys is not None:
            self.keys = self.keys.detach()
            self.values = self.values.detach()
        if self.window_mask is not None:
            self.window_mask = self.window_mask.detach()

    def update_memory_l(self, un_rotated_k, v, past_key_value, empty_cache_flag):
        # print('update memory', un_rotated_k.shape, v.shape, self.rmt_size, self.original_seq_length)

        if self.config.memory_size == 0:
            return past_key_value

        if self.rmt_size > 0:
            if (self.original_seq_length < self.segment_size and empty_cache_flag):  # 1, 删除左侧
                un_rotated_k = un_rotated_k[:, :, -self.original_seq_length:]
                v = v[:, :, -self.original_seq_length:]
                assert self.rmt_pad_flag == 1
            elif (self.original_seq_length == 1 and (not empty_cache_flag)):  # 2. 不删除
                assert self.rmt_pad_flag == 2
            elif self.original_seq_length == self.segment_size and empty_cache_flag:  # situation 3
                if un_rotated_k.shape[2] >= self.original_seq_length + 2 * self.rmt_size:
                    un_rotated_k = un_rotated_k[:, :, -(self.rmt_size + self.original_seq_length):-self.rmt_size]
                    v = v[:, :, -(self.rmt_size + self.original_seq_length):-self.rmt_size]   # Don't put rmt tokens in each layer's memory
                elif un_rotated_k.shape[2] == self.original_seq_length + self.rmt_size:
                    un_rotated_k = un_rotated_k[:, :, self.rmt_size:]
                    v = v[:, :, self.rmt_size:]   # don't put rmt tokens in each layer's memory
                assert self.rmt_pad_flag == 3
            else:
                print(self.original_seq_length, empty_cache_flag)
                raise NotImplementedError

        # ============================================
        if (self.original_seq_length < self.segment_size and empty_cache_flag) \
            or (self.original_seq_length == 1 and (not empty_cache_flag)):   # 1 / 2
            if self.kv_cache_unrotated is None:
                self.kv_cache_unrotated = [un_rotated_k, v]
            else:
                assert un_rotated_k.shape[2] == 1 and v.shape[2] == 1, \
                    (un_rotated_k.shape, v.shape, self.kv_cache_unrotated[0].shape, self.kv_cache_unrotated[1].shape)
                self.kv_cache_unrotated = [torch.cat((self.kv_cache_unrotated[0], un_rotated_k), dim=2),
                                           torch.cat((self.kv_cache_unrotated[1], v), dim=2)]
            if self.kv_cache_unrotated[0].shape[2] == self.segment_size:  # todo: generate magic 512
                un_rotated_k, v = self.kv_cache_unrotated
                print('empty cache', un_rotated_k.shape, v.shape)
                self.kv_cache_unrotated = None
                past_key_value = None
                self.original_seq_length = self.segment_size  # todo: generate
                # print('full', past_key_value)
                # raise ValueError
            elif self.kv_cache_unrotated[0].shape[2] < self.segment_size:
                return past_key_value
            else:
                raise NotImplementedError
            assert self.rmt_pad_flag in [1, 2]
        elif self.original_seq_length == self.segment_size and empty_cache_flag:  # situation 3
            past_key_value = None
            assert self.rmt_pad_flag == 3
        else:
            print(self.original_seq_length, past_key_value[0].shape)
            raise NotImplementedError

        # ====================================================

        # original_seq_length is set by ExternalMemory
        assert self.segment_size == self.original_seq_length
        # assert self.segment_size == 512, ('please check magic 512 in update_memory_l!', self.segment_size)

        if self.keys is None:
            bsz, num_head, _, head_dim = un_rotated_k.shape
            self.keys = torch.zeros((bsz, num_head, self.config.memory_size * self.segment_size, head_dim),
                                    dtype=un_rotated_k.dtype, device=un_rotated_k.device)
            self.values = torch.zeros((bsz, num_head, self.config.memory_size * self.segment_size, head_dim),
                                      dtype=un_rotated_k.dtype, device=un_rotated_k.device)
            self.current_memory = 0
        
        
        if self.current_memory >= self.config.memory_size:
            if self.config.reset:
                self.reset_memory()
                return past_key_value
            else:
                self.keys[:, :, :-self.segment_size, :] = self.keys[:, :, self.segment_size:, :]
                self.values[:, :, :-self.segment_size, :] = self.values[:, :, self.segment_size:, :]
                self.current_memory -= 1
                del self.chunk_size[0]
                assert len(self.chunk_size) == self.current_memory, (len(self.chunk_size), self.current_memory)

        if self.config.pos_type == 'pos0':
            cos, sin = self.rotary_emb(v, seq_len=un_rotated_k.shape[-2])
            position_ids = torch.zeros(1, un_rotated_k.shape[-2], device=cos.device, dtype=torch.long)
            cos, sin = cos.to(v.device), sin.to(v.device)
            _, un_rotated_k = apply_rotary_pos_emb(un_rotated_k, un_rotated_k, cos, sin, position_ids)


        self.keys[:, :, self.current_memory * self.segment_size: self.current_memory * self.segment_size + self.segment_size, :] = un_rotated_k
        self.values[:, :, self.current_memory * self.segment_size: self.current_memory * self.segment_size + self.segment_size, :] = v
        self.current_memory += 1

        self.chunk_size.append(v.shape[2])

        return past_key_value

    def retrieve(self, query):
        """
        :param query: [b, n, l d]
        :return:  (keys[bnc'd], values[bnc'd], attn_mask[bnlc']), (topk_keys[bnlkd], topk_values[bnlkd])
        """
        if len(self.chunk_size) > 0:
            batch_size, n_head, query_length, head_dim = query.shape
            window_mask = self._generate_window_mask(query_length) if self.config.window_length > 0 else None
            
            ##f global token
            if self.config.global_token > 0:
                # print('self.config.global_token - ', self.config.global_token[self.layer_id])
                assert self.config.global_token <= window_mask.shape[-1]
                if self.config.global_pos == 'random':
                    random.seed(42)  # todo: seed here may affect other random process
                    self.global_tokens = random.sample(list(range(window_mask.shape[-1])), k=self.config.global_token)
                else:
                    self.global_tokens = [gt for gt in range(self.config.global_token)]
                if window_mask is not None:
                    for j in self.global_tokens:
                        window_mask[..., j] = True
                        
            window_mask_bool = window_mask
            if window_mask is not None:
                with torch.no_grad():
                    window_mask = 1 - window_mask.to(self.values.dtype)
                    # print('window_mask 0/1 - ', window_mask)
                    window_mask = window_mask.masked_fill(window_mask.to(torch.bool), torch.finfo(query.dtype).min)
                    # print('window_mask 0/-infinite - ', window_mask)
                    window_mask = window_mask.repeat(batch_size, 1, 1, 1)
                    
            # if self.config.topk > 0:
            len_kv = sum(self.chunk_size)
            topk_opt_by_global_token = self.config.global_token > 0
            if self.config.topk > 0 or topk_opt_by_global_token:  ##f for global token
                topk_index = self._generate_topk_index_out_of_windows(query, window_mask_bool, topk_opt_by_global_token)  # [b, n, l, 32]
                if window_mask is not None and self.config.global_token > 0:
                    for j in self.global_tokens:
                        window_mask[..., j] = False
                with torch.no_grad():

                    B_indices = torch.arange(batch_size)[:, None, None, None].expand(-1, n_head, query_length, topk_index.shape[-1])
                    N_indices = torch.arange(n_head)[None, :, None, None].expand(batch_size, -1, query_length, topk_index.shape[-1])
                    topk_keys = self.keys[:, :, :self.current_memory*self.segment_size][B_indices, N_indices, topk_index, :].clone()
                    topk_values = self.values[:, :, :self.current_memory*self.segment_size][B_indices, N_indices, topk_index, :].clone()

                    # print('delta', (topk_keys -topk_keys1).abs().mean(), (topk_values-topk_values1).abs().mean())

            else:
                topk_keys, topk_values = None, None


            if window_mask is not None:
                valid_length = min(self.config.window_length, self.current_memory*self.segment_size)
                window_keys, window_values, window_mask = (self.keys[:, :, self.current_memory*self.segment_size-valid_length:self.current_memory*self.segment_size].clone(),
                                                           self.values[:, :, self.current_memory*self.segment_size-valid_length:self.current_memory*self.segment_size].clone(),
                                                           window_mask[..., self.current_memory*self.segment_size-valid_length:self.current_memory*self.segment_size].clone())
                
                return (window_keys, window_values, window_mask), (topk_keys, topk_values)
            else:
                return (None, None, None), (topk_keys, topk_values)

        return None

    def _generate_topk_index_out_of_windows(self, query, window_mask, topk_opt_by_global_token=False):      ##f query - (2, 2, 6, 6), window_mask - (1, 2, 6, 4)
        # query [b, n, l, d]; window_mask [1, n, l, w]
        batch_size, n_head, len_q, _ = query.shape
        len_kv = sum(self.chunk_size)
        with torch.no_grad():

            all_topk_index = []
            assert self.current_memory * self.segment_size <= 2048 ** 2
            
            for bi in range(batch_size):  # take care if memory is too large
                for ni in range(n_head):
                    seg_l = 2048 ** 2 // (self.current_memory * self.segment_size)
                    for li in range(0, len_q, seg_l):
                        end_li = min(li + seg_l, len_q)
                        
                        attn_weight = torch.matmul(query[bi, ni, li: end_li], self.keys[bi, ni, :self.current_memory*self.segment_size].transpose(-1, -2))  # [l, lkv]
                        
                        if window_mask is not None:
                            assert window_mask.shape[0] == 1
                            attn_weight = attn_weight.masked_fill(window_mask[0, ni, li: end_li], torch.finfo(query.dtype).min)
                        current_topk = min(attn_weight.shape[-1], self.config.topk)
                        topk_index = torch.topk(attn_weight, k=current_topk, dim=-1)[1]   # [l, 32]
                        del attn_weight
                        all_topk_index.append(topk_index)
            topk_index = torch.cat(all_topk_index, dim=0).view(batch_size, n_head, len_q, -1)
            
            # print(batch_size, n_head, len_q, self.config.global_token)
            expand_tensor = torch.zeros((batch_size, n_head, len_q, self.config.global_token), dtype=topk_index.dtype, device=topk_index.device)
            for gt in range(self.config.global_token):
                expand_tensor[..., gt] = self.global_tokens[gt]
                
            if topk_opt_by_global_token and self.config.topk > 0:
                topk_index = torch.cat((topk_index, expand_tensor), dim=-1)
            elif topk_opt_by_global_token:
                topk_index = expand_tensor
                
        return topk_index

    def _generate_topk_mask(self, query):
        batch_size, n_head, len_q, _ = query.shape
        len_kv = sum(self.chunk_size)
        with torch.no_grad():
            attn_weight = torch.matmul(query, self.keys[:, :, :self.current_memory*self.segment_size].transpose(2, 3))  # [l, lkv]
            
            topk_thres = torch.quantile(attn_weight.float(), q=1 - self.config.topk / len_kv, dim=-1, keepdim=True).to(attn_weight.dtype)
            topk_mask = torch.where(attn_weight > topk_thres, True, False)
            # print('topk ratio', (topk_mask > 0).float().sum() / batch_size / n_head / len_q, topk_mask.shape,  )
        return topk_mask  # [b, n, l, k]

    def _generate_window_mask(self, query_length):
        len_kv = sum(self.chunk_size)
        if (self.window_mask is not None) and (self.window_mask.shape[-1] >= len_kv):
            return self.window_mask[:, :, :query_length, -len_kv:]

        window_mask = [self._generate_window_mask_for_chunk(query_length, chunk_i)
                       for chunk_i in range(len(self.chunk_size))]  # can be further optimized
        self.window_mask = torch.cat(window_mask, dim=-1).bool().cuda()
        return self.window_mask

    def _generate_window_mask_for_chunk(self, query_length, chunk_i):
        # return window_mask [1, n, l, w]
        len_memory = sum(self.chunk_size)
        window_size = min(self.config.window_length, len_memory)
        start_index = len_memory - window_size  # window span from start_index
        window_size_all = [min(self.config.window_length - ql, len_memory) for ql in range(query_length)]
        start_index_all = [len_memory - wsq for wsq in window_size_all]  # start index for each item in a batch
        if start_index > sum(self.chunk_size[:chunk_i + 1]):
            return torch.zeros((1, self.config.num_attention_heads, query_length, self.chunk_size[chunk_i]))

        mask_for_seq = []
        # print('-chunking------------------------', chunk_i, start_index, self.chunk_size)
        prev_len = sum(self.chunk_size[:chunk_i])
        end_index = sum(self.chunk_size[:chunk_i + 1])
        for ql in range(query_length):
            start_index_q = max(start_index_all[ql], prev_len)
            import random
            if start_index_q >= end_index:
                mask = torch.zeros(end_index - prev_len)
            else:
                mask = torch.cat((torch.zeros((start_index_q - prev_len)),
                                  torch.arange(start_index_q, end_index) % self.config.stride == (ql % self.config.stride)), dim=0)
            mask_for_seq.append(mask.to(torch.int8))

        mask_for_seq = torch.stack(mask_for_seq, dim=0).view(1, 1, query_length, self.chunk_size[chunk_i]
                                                             ).repeat(1, self.config.num_attention_heads, 1,
                                                                      1)  # [1, n, l, w]
        return mask_for_seq

    def _init_rope(self):
        if self.config.pos_type == 'pos0':
            # copied from llama._init_rope
            if self.config.rope_scaling is None:
                self.rotary_emb = LlamaRotaryEmbedding(
                    self.config.hidden_size // self.config.num_attention_heads,
                    max_position_embeddings=self.config.max_position_embeddings,
                    base=self.config.rope_theta,
                )
            else:
                scaling_type = self.config.rope_scaling["type"]
                scaling_factor = self.config.rope_scaling["factor"]
                if scaling_type == "linear":
                    self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                        self.config.hidden_size // self.config.num_attention_heads,
                        max_position_embeddings=self.config.max_position_embeddings,
                        scaling_factor=scaling_factor,
                        base=self.config.rope_theta,
                    )
                elif scaling_type == "dynamic":
                    self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                        self.config.hidden_size // self.config.num_attention_heads,
                        max_position_embeddings=self.config.max_position_embeddings,
                        scaling_factor=scaling_factor,
                        base=self.config.rope_theta,
                    )
                else:
                    raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
        elif self.config.pos_type in ['none', 0]:  # 0 is default
            pass
        else:
            raise NotImplementedError

class Dict2Class:
    def __init__(self, the_dict):
        if the_dict:
            self.__dict__.update(the_dict)


if __name__ == '__main__':
    config = LlamaConfig(hidden_size=512, intermediate_size=2048, num_hidden_layers=12, num_attention_heads=8,
                         max_position_embeddings=2048, vocab_size=50260,
                         memory_params_other={'rmt_size': 4, 'rmt_memory_size': 2},
                         memory_layer=[9, 11],
                         memory_params_each_layer={'memory_size': [4, 1], 'window_length': [80, 23], 'stride': [1, 1], 'topk': [8, 8],
                                                   'pos_type': ['pos0', 'pos0'], 'retrieved_length': [None, None]})
    model = LlamaModel(config)
    model.cuda()
    for i in range(10):
        seq_len = 10
        inputs_ids = torch.arange(seq_len).reshape(1, seq_len).cuda().repeat(2, 1)
        attention_mask = torch.ones(2, 1, seq_len, seq_len).cuda()
        position_ids = torch.arange(seq_len).reshape(1, seq_len).cuda()
        print(f'step {i} inputs_ids', inputs_ids.shape)
        output = model(inputs_ids, attention_mask, position_ids)
    print('Unit test of memory  ==========================================')
    # raise NotImplementedError

