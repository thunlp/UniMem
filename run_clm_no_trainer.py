#!/usr/bin/env python
# coding=utf-8
# Adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py


# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import random
from itertools import chain
from pathlib import Path
import time

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    LlamaForCausalLM
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from torch.utils.tensorboard import SummaryWriter
# from supervised_dataset import SupervisedDataset, build_supervised_dataset

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.35.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv, txt or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv, txt or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="RMT",
        help="The scheduler type to use.",
        choices=["RMT", "MemTrans", "Trans-XL", "Longformer", "mix", "Vanilla", "Custom"],
    )
    parser.add_argument(
        "--rmt_size",
        type=int,
        default=0,
        help="Num of Summary tokens.",
    )
    parser.add_argument(
        "--rmt_memory_size",
        type=int,
        default=0,
        help="RMT memory cache size.",
    )
    parser.add_argument(
        "--memory_size",
        type=int,
        default=0,
        help="Memory Cache Size.",
    )
    parser.add_argument(
        "--window_length",
        type=int,
        default=0,
        help="Sparse Attention Sliding Window Length.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Dilated Sliding Window gap.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=0,
        help="KNN num. each token attend to topk tokens in memory",
    )
    parser.add_argument(
        "--memory_layer",
        nargs='+',
        # type=str,
        default=[str(i) for i in range(32)],
        help="The index of layers with memory.",
    )
    parser.add_argument(
        "--global_tokens",
        type=int,
        default=0,
        help="The number of global tokens(start from 0).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Whether clear all memory at full capacity.",
    )
    parser.add_argument(
        "--local_stride",
        action="store_true",
        help="Whether clear all memory at full capacity.",
    )
    parser.add_argument(
        "--global_pos",
        type=str,
        default='beginning',
        help="The pos of global token, in the beginning of sequence / random",
    )
    parser.add_argument(
        "--pos_type",
        type=str,
        default='pos0',
        help="pos0 or none",
    )
    parser.add_argument('--n_bptt_step', type=int, default=1, help='num of BPTT steps for truncation')
    parser.add_argument('--eval_per_n_step', type=int, default=512, help='evaluate per n training steps')
    parser.add_argument('--cache_dir', type=str, default=None, help='Local dataset path')
    parser.add_argument('--sft', action='store_true')
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        pass
        # print('No data provided')
        # raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`train_file` should be a csv, json or txt file.")
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`validation_file` should be a csv, json or txt file.")

    if args.push_to_hub:
        if args.output_dir is None:
            raise ValueError("Need an `output_dir` to create a repo when `--push_to_hub` is passed.")

    return args


def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_clm_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            repo_id = create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id
            # Clone repo locally
            repo = Repository(args.output_dir, clone_from=repo_id, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    cache_dir = args.cache_dir
    print('Current Dataset - ', cache_dir)
    # raw_datasets = load_dataset(cache_dir)
    # raw_datasets = {}
    raw_datasets = load_dataset('json', data_files=cache_dir)
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            'json', 
            data_files=cache_dir,
            split=f"train[:{args.validation_split_percentage}%]",
        )
        raw_datasets["train"] = load_dataset(
            'json', 
            data_files=cache_dir,
            split=f"train[{args.validation_split_percentage}%:]",
        )
    # logger.warning('load data from cached wikitext '+  cache_dir + str(list(raw_datasets.keys())))
    # logging.warning('smaller dataset for faster tokenizing')
    # raw_datasets["train"] = load_dataset(
    #     cache_dir,
    #     split=f"train[90%:]",
    # )

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    elif (args.train_file is not None) or (args.validation_file is not None):
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    print('Current mode - ', args.mode)
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
        print('load pretrained model')
        if args.mode == 'RMT':
            config.update_memory_params(memory_params_other={'rmt_size': 40, 'rmt_memory_size': 1, 'segment_size': args.block_size},
                             memory_layer=[9,],
                             memory_params_each_layer={'memory_size': 0, 'window_length': 0, 'stride': 2, 'topk': 32, 'pos_type': 'pos0', 'retrieved_length': 32, 'reset': args.reset, 'local_stride': args.local_stride,
                                                       'global_pos': args.global_pos, 'global_token': 0,})
        elif args.mode == 'MemTrans':
            config.update_memory_params(memory_params_other={'rmt_size': 0, 'rmt_memory_size': 0, 'segment_size': args.block_size},
                             memory_layer=[11, 21],
                            #  memory_layer=[i for i in range(32)],
                             memory_params_each_layer={'memory_size': 20, 'window_length': 0, 'stride': 2, 'topk': 32, 'pos_type': 'pos0', 'retrieved_length': 32, 'reset': False, 'local_stride': args.local_stride,
                                                       'global_pos': args.global_pos, 'global_token': 0})
        elif args.mode == 'Trans-XL':
            config.update_memory_params(memory_params_other={'rmt_size': 0, 'rmt_memory_size': 0, 'segment_size': args.block_size},
                             memory_layer=[i for i in range(22)],
                             memory_params_each_layer={'memory_size': 1, 'window_length': 512, 'stride': 1, 'topk': 0, 'pos_type': 'pos0', 'retrieved_length': 32, 'reset': False, 'local_stride': args.local_stride,
                                                       'global_pos': args.global_pos, 'global_token': 0})
        elif args.mode == 'Longformer':
            config.update_memory_params(memory_params_other={'rmt_size': 0, 'rmt_memory_size': 0, 'segment_size': args.block_size},
                            #  memory_layer=[16, 24],
                             memory_layer=[i for i in range(22)],
                             memory_params_each_layer={'memory_size': 2, 'window_length': 512, 'stride': 2, 'topk': 0, 'pos_type': 'pos0', 'retrieved_length': 32, 'reset': True, 'local_stride': args.local_stride,
                                                       'global_pos': args.global_pos, 'global_token': 4})
        elif args.mode == 'Vanilla':
            config.update_memory_params(memory_params_other={'rmt_size': 0, 'rmt_memory_size': 0, 'segment_size': args.block_size},
                                memory_layer = [i for i in range(22)],
                                memory_params_each_layer = {'memory_size': 0, 'window_length': 0, 'stride': 0,
                                                            'topk': 0, 'pos_type': 'pos0', 'retrieved_length': 32, 'reset': args.reset, 'local_stride': args.local_stride,
                                                            'global_pos': args.global_pos, 'global_token': 0})
        elif args.mode == 'mix':
            config.update_memory_params(memory_params_other={'rmt_size': 40, 'rmt_memory_size': 1, 'segment_size': args.block_size},
                                        memory_layer=[i for i in range(11, 22)],
                                        memory_params_each_layer={'memory_size': 20, 'window_length': 512, 'stride': 1,
                                                                  'topk': 4, 'pos_type': 'pos0',
                                                                  'retrieved_length': 32, 'reset': False, 'local_stride': args.local_stride,
                                                                  'global_pos': args.global_pos, 'global_token': 4})
        elif args.mode == 'Custom':
            config.update_memory_params(memory_params_other={'rmt_size': args.rmt_size, 'rmt_memory_size': args.rmt_memory_size, 'segment_size': args.block_size},
                                        memory_layer=[int(i) for i in args.memory_layer],
                                        memory_params_each_layer={'memory_size': args.memory_size, 'window_length': args.window_length, 'stride': args.stride,
                                                                  'topk': args.topk, 'pos_type': args.pos_type,
                                                                  'retrieved_length': 32, 'global_token': args.global_tokens, 'global_pos': args.global_pos,
                                                                  'reset': args.reset, 'local_stride': args.local_stride})
    else:
        from transformers import LlamaConfig
        if args.debug == True:
            config = LlamaConfig(hidden_size=12, intermediate_size=36, num_hidden_layers=2, num_attention_heads=2,
                                max_position_embeddings=2048, memory_params_other={'rmt_size': 0, 'rmt_memory_size': 0, 'segment_size': args.block_size},
                                memory_layer=[0,],
                                memory_params_each_layer={'memory_size': 3, 'window_length': 8, 'stride': 2, 'topk': 2, 'pos_type': 'pos0', 'retrieved_length': 32, 'global_token': 0, 'reset': False, 'local_stride': args.local_stride})
        elif args.mode == 'RMT':
            config = LlamaConfig(hidden_size=512, intermediate_size=2048, num_hidden_layers=12, num_attention_heads=8,
                             max_position_embeddings=2048, vocab_size=50260, memory_params_other={'rmt_size': 20, 'rmt_memory_size': 2, 'segment_size': args.block_size},
                             memory_layer=[9,],
                             memory_params_each_layer={'memory_size': 0, 'window_length': 0, 'stride': 2, 'topk': 32, 'pos_type': 'pos0', 'retrieved_length': 32})
        elif args.mode == 'MemTrans':
            config = LlamaConfig(hidden_size=512, intermediate_size=2048, num_hidden_layers=12, num_attention_heads=8,
                             max_position_embeddings=2048, vocab_size=50260, memory_params_other={'rmt_size': 0, 'rmt_memory_size': 0, 'segment_size': args.block_size},
                             memory_layer=[9,],
                             memory_params_each_layer={'memory_size': 8, 'window_length': 0, 'stride': 2, 'topk': 32, 'pos_type': 'pos0', 'retrieved_length': 32})

        elif args.mode == 'Trans-XL':
            config = LlamaConfig(hidden_size=512, intermediate_size=2048, num_hidden_layers=12, num_attention_heads=8,
                             max_position_embeddings=2048, vocab_size=50260, memory_params_other={'rmt_size': 0, 'rmt_memory_size': 0, 'segment_size': args.block_size},
                             memory_layer=[i for i in range(12)],
                             memory_params_each_layer={'memory_size': 1, 'window_length': 512, 'stride': 1, 'topk': 0, 'pos_type': 'pos0', 'retrieved_length': 32})
            
        elif args.mode == 'Longformer':
            config = LlamaConfig(hidden_size=512, intermediate_size=2048, num_hidden_layers=12, num_attention_heads=8,
                             max_position_embeddings=2048, vocab_size=50260, memory_params_other={'rmt_size': 0, 'rmt_memory_size': 0, 'segment_size': args.block_size},
                             memory_layer=[i for i in range(12)],
                             memory_params_each_layer={'memory_size': 3, 'window_length': 1536, 'stride': 2, 'topk': 0, 'pos_type': 'pos0', 'retrieved_length': 32, 'reset': True})

        elif args.mode == 'Vanilla':
            config = LlamaConfig(hidden_size=512, intermediate_size=2048, num_hidden_layers=12, num_attention_heads=8,
                                 max_position_embeddings=2048, vocab_size=50260,
                                 memory_params_other={'rmt_size': 0, 'rmt_memory_size': 0, 'segment_size': args.block_size},
                                 memory_layer=[i for i in range(12)],
                                 memory_params_each_layer={'memory_size': 0, 'window_length': 0, 'stride': 0,
                                                           'topk': 0, 'pos_type': 'pos0', 'retrieved_length': 32})
        elif args.mode == 'mix':
            config = LlamaConfig(hidden_size=512, intermediate_size=2048, num_hidden_layers=12, num_attention_heads=8,
                             max_position_embeddings=2048, vocab_size=50260, memory_params_other={'rmt_size': 40, 'rmt_memory_size': 2, 'segment_size': args.block_size},
                             memory_layer=[9,],
                             memory_params_each_layer={'memory_size': 0, 'window_length': 0, 'stride': 2, 'topk': 32, 'pos_type': 'pos0', 'retrieved_length': 32})
        else: 
            print('Error mode' - args.mode)
            exit()

        logger.warning("args.model_type does not take effect. You are instantiating a small llama.")
        # config = CONFIG_MAPPING[args.model_type]()
        # logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        assert not args.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    else:
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained("oobabooga/llama-tokenizer", cache_dir='/home/tanglikai/llama/unimem/examples/language/llama2/cache')
        logger.warning("You are using cached oobabooga/llama-tokenizer.")
        # raise ValueError(
        #     "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
        #     "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        # )
        
    special_tokens_dict = {'additional_special_tokens': ['<start>','<end>','<pad>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<pad>')
    
    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    model.gradient_checkpointing_enable()
    
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        print('resize', embedding_size, '->', len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        result = tokenizer(examples[text_column_name], max_length=args.block_size, truncation=True, padding='max_length')
        result["labels"] = result["input_ids"].copy()
        return result

    with accelerator.main_process_first():
        if args.sft:
            raise NotImplementedError
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > config.max_position_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            block_size = min(1024, config.max_position_embeddings)
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    
    # def group_texts(examples):
    #     ##f 已经在预处理阶段做过了
    #     result = {
    #         k: t
    #         for k, t in examples.items()
    #     }
    #     result["labels"] = result["input_ids"].copy()
    #     return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map

    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets
        # lm_datasets = tokenized_datasets.map(
        #     group_texts,
        #     batched=True,
        #     num_proc=args.preprocessing_num_workers,
        #     load_from_cache_file=not args.overwrite_cache,
        #     desc=f"Grouping texts in chunks of {block_size}",
        # )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]
    
    print(len(train_dataset))
    print(len(eval_dataset))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size, drop_last=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size, drop_last=True
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config.update({k: str(v) for k, v in experiment_config.items() if isinstance(v, list)})  # fixme
        print(experiment_config)
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    start_time = time.time()
    memory_opTime = 0
    
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    loss_across_bptt = 0
    perplexity = 9999
    for epoch in range(starting_epoch, args.num_train_epochs):
        if isinstance(model, LlamaForCausalLM):
            if model.model.memory is not None:
                model.model.memory.reset_memory()
        else:
            if model.module.model.memory is not None:
                model.module.model.memory.reset_memory()

        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        
        
        start_time = time.time()
        memory_optTime = 0
        writer = SummaryWriter()
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # memory_opTime += memory_time
                if accelerator.is_main_process:
                    print('step - ', step, ', loss - ', loss, ', memory_time - ', 0)
                loss_across_bptt += loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                if step % args.n_bptt_step == 0:
                    accelerator.backward(loss_across_bptt)
                    if isinstance(model, LlamaForCausalLM):
                        if model.model.memory is not None:
                            model.model.memory.detach_memory()
                    else:
                        if model.module.model.memory is not None:
                            model.module.model.memory.detach_memory()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    loss_across_bptt = 0
            if step % 100 == 0:
                if accelerator.is_main_process:
                    print(args.mode, args.n_bptt_step, 'step - ', step, loss.item())

            effective_flag = (loss > 0).float()
            all_loss = accelerator.gather_for_metrics(loss.repeat(args.per_device_train_batch_size)).sum()
            all_effective = accelerator.gather_for_metrics(effective_flag.repeat(args.per_device_train_batch_size)).sum()
            if all_loss > 0:
                if args.with_tracking:
                    accelerator.log(
                        {
                            "train_loss_step": all_loss.item() / all_effective,
                        },
                        step=completed_steps,
                    )
                

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

                    if accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                        )
                        tokenizer.save_pretrained(args.output_dir)
            if completed_steps >= args.max_train_steps:
                break

            if step % args.eval_per_n_step == args.eval_per_n_step - args.n_bptt_step:
                attn_time = time.time() - start_time - memory_opTime
                test_start_time = time.time()
                if accelerator.is_main_process:
                    print('step', step, args.mode, args.n_bptt_step, completed_steps, 'eval--------------------------------')
                if isinstance(model, LlamaForCausalLM):
                    if model.model.memory is not None:
                        model.model.memory.reset_memory()
                else:
                    if model.module.model.memory is not None:
                        model.module.model.memory.reset_memory()
                model.eval()
                losses = []
                for step, batch in enumerate(eval_dataloader):
                    # if step >= 512:
                    #     break  # todo
                    with torch.no_grad():
                        outputs = model(**batch)

                    loss = outputs.loss
                    losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))
                if isinstance(model, LlamaForCausalLM):
                    if model.model.memory is not None:
                        model.model.memory.reset_memory()
                else:
                    if model.module.model.memory is not None:
                        model.module.model.memory.reset_memory()
                model.train()

                losses = torch.cat(losses)
                try:
                    eval_loss = torch.mean(losses)
                    perplexity = math.exp(eval_loss)

                except OverflowError:
                    perplexity = float("inf")

                logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

                if args.with_tracking:
                    # accelerator.log({'Loss by RunTime': eval_loss.item(), 'Perplexity by RunTime': perplexity},
                    #                 step=int(attn_time))
                    accelerator.log(
                        {
                            "perplexity": perplexity,
                            "eval_loss": eval_loss,
                            "epoch": epoch,
                            "step": completed_steps,
                        },
                        step=completed_steps,
                    )
                start_time += time.time() - test_start_time  # substrate test time from training time

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)


if __name__ == "__main__":
    main()
