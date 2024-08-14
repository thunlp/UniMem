import sys
import time
import inspect

from transformers import AutoTokenizer
from typing import Any
import numpy as np
from tqdm import tqdm

import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default='Your path',
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='Your path',
        help="The path of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Training batch size",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=255,
        help="Chunk of training data, chunk length = chunk_size * max_length",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=10240,
        help="Training batch size",
    )
    args = parser.parse_args()
    return args


def get_tokenizer(tokenizer_name):
    tokenizer = tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, use_fast=not False, trust_remote_code=False
        )
    special_tokens_dict = {'additional_special_tokens': ['<start>','<end>','<pad>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    return tokenizer


def convert_data_to_id(tokenizer: AutoTokenizer, data: Any):
    input_ids = tokenizer.encode(data)
    # output_ids = tokenizer.encode(data["output"])
    # ids = [tokenizer.bos_id] + input_ids + [tokenizer.eos_id]
    ids = input_ids
    ids = np.array(ids, dtype=np.int32)
    context = np.zeros((ids.shape[0],), dtype=np.int8)
    context[: len(input_ids) + 1] = 1
    return ids, context

def data_to_id(tokenizer, data: Any):
    return convert_data_to_id(tokenizer, data)


args = parse_args()
# initialize()
tokenizer = get_tokenizer(args.tokenizer_name)


import os

folder_path = args.dataset


for filename in os.listdir(folder_path):
    filenames.append(folder_path+'/'+filename)



for filename in filenames:
# data = open('/home/jeeves/zhllama-memory/pile_sample.json').readlines()
    data = open(filename).readlines()

    complete_dataset = ''

    print('############ Start data reading ###########')

    start = time.time()
    cnt = 0
    error_cnt = 0
    token_ids = np.array([], dtype=np.int32)
    max_length = args.max_length
    # max_length = 40
    max_seq_length = 26000

    data_52w_token_per_line = []
    data_bsz1 = []
    data_bsz2 = []
    local_cnt = 0
    temp_dic_list = []
    dic_list_520k = []
    chunk_size = args.chunk_size
    # chunk_size = 2
    batch_size = args.batch_size
    tokenizer_name = args.tokenizer_name.split('/')[-1]
    dataset_name = args.dataset.split('/')[-1]


    with open('output.json', 'a+') as f:
        for idx, line in enumerate(data):
            # idx = idx + 1985
            print('Cur idx - ', idx)
            try:
                print('Start Spliting')
                try:
                    line = json.loads(line)
                except UnicodeDecodeError:
                    print('Error line load: ', idx)
                    continue
                # complete_dataset += line['text'] + '\n\n'
                cur_texts = []
                temp = '<start>' + line['text']
                temp = temp.split()
                print('Current line - ', idx, ', tokens - ', len(temp))
                # temp = temp[:MAX_LENGTH]
                # temp_cnt = 0
                cur_texts.append(' '.join(temp[:max_seq_length]))       ##f 0-1
                # print(cur_texts)

                
                while len(temp) > max_seq_length:
                    temp = temp[max_seq_length:]       ##f 2: 
                    cur_texts.append(' '.join(temp[:max_seq_length])) 
                    
                
                print('Start Encoding')
                min_length_flag = False
                for temp_line in cur_texts:
                    try:
                        token_id, _ = data_to_id(tokenizer, temp_line)
                        if len(cur_texts) < 2 and len(token_id) < args.min_length:
                            min_length_flag = True
                            break
                        # print(len(token_id))
                        token_ids = np.concatenate((token_ids, token_id), dtype=np.int32)
                        if len(token_ids) > max_length*chunk_size:
                            break
                        # print(len(token_ids))
                    except UnicodeDecodeError:
                        print('Error line - encoding: ', idx)
                # print(token_ids)
                print('current line token ids - ', len(token_ids))
                if min_length_flag:
                    continue

                ##f 到这边是得到了当前行的token id
                ##f 这里要做的是 大于1040k，就bsz1和bsz2同时加
                print('Start Decoding')
                
                while len(token_ids) > max_length*chunk_size:
                    while len(token_ids) > max_length:
                        try:
                            temp_text = tokenizer.decode(token_ids[: max_length])
                            # print(temp_text)
                            real_len_gap = max_length - len(tokenizer.encode(temp_text))
                            if real_len_gap > 0:
                                temp_text += '<pad>' * real_len_gap
                            # temp_text = tokenizer.decode(token_ids[max_length * cnt : max_length * (cnt+1)])
                            temp_dic = {'text': temp_text}

                            temp_dic_list.append(temp_dic)
                            # f.write(json.dumps(temp_dic) + "\n")
                            local_cnt = local_cnt + 1
                            
                            token_ids = token_ids[max_length:]
                            if local_cnt == chunk_size:
                                local_cnt = 0
                                cnt = cnt + 1
                                token_ids = np.array([], dtype=np.int32)
                                dic_list_520k.append(temp_dic_list)
                                temp_dic_list = []
                                break
                        except UnicodeDecodeError:
                            # 处理解码错误，可以选择跳过这些字节或采取其他操作
                            print('Error line - decoding: ', idx)
                            # cnt = cnt + 1
                            token_ids = token_ids[max_length:]
                    # if cnt 
                    
                print('Current data - ', cnt)
                
            except UnicodeDecodeError:
                print('Error line: ', idx)
        

        for idx in range(0, len(dic_list_520k)-batch_size, batch_size):
            for line_i in range(len(dic_list_520k[0])):
                for i in range(batch_size):
                    f.write(json.dumps(dic_list_520k[idx+i][line_i]) + "\n")
                
                
        
