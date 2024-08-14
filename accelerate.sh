#!/bin/bash

echo $CUDA_VISIBLE_DEVICES
mode=$1
data_path=$2
ckpt_path=$3
accelerate launch --config_file deepspeed_config.yaml  \
run_clm_no_trainer.py \
    --block_size 512 \
    --output_dir ./logs/${mode} \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --with_tracking \
    --cache_dir ${data_path} \
    --mode $mode \
    --n_bptt_step 1 \
    --report_to tensorboard \
    --model_name_or_path ${ckpt_path} \
    --eval_per_n_step 2048 \
    --checkpointing_steps  512 \
    --num_train_epochs 1


#--tokenizer_name '/home/qinyujia/project/unimem-master/gpt2' \