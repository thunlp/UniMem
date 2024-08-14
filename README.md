
# UniMem: Towards a Unified View of Long-Context Large Language Models

## Note
This codebase accompanies paper "**UniMem: Towards a Unified View of Long-Context Large Language Models**" ([Link](https://arxiv.org/abs/2402.03009)) and implements several long-context methods in a unified framework. The implementation is adapted from [transformers](https://github.com/huggingface/transformers) codebases which is open-sourced, and our modification is constrained in src/transformers/models/llama.

## Installation instructions

### Install neccessary packages
(assume you have pytorch installed)
```shell
pip install datasets accelerate
pip install deepspeed==0.13.1
cd ./transformers
pip install -e .
cd ..
```
### Prepare dataset
Download data from huggingface in json format(follow the format {"text": ...} in each line), and preprocess them with: 
```shell 
python preprocess_longText_strict_multi_file_pretrain_huggingface.py --dataset {path to downloaded dataset} --max_length 512
```
The processed data will be saved in the same path as the downloaded data

### Prepare model checkpoint
Since most our experiments finetune model from a pretrained model, like Llama2-7B, the checkpoint should be first downloaded from huggingface
Llama2-7B: https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main


## Run an experiment (training)

```shell
bash accelerate.sh mix {path to preprocessed data} {path to pretrained model checkpoint}
```
This cmd trains with our proposed Unimix, if you'd like to trains with other method, replace "mix" with "RMT", "MemTrans", "Trans-XL", "Longformer", or "Vanilla"

You can also customize the config in all the design dimensions 

## Citation
If you find UniMem useful for your research and applications, please cite using this BibTeX:

```@article{fang2024unimem,
  title={Unimem: Towards a unified view of long-context large language models},
  author={Fang, Junjie and Tang, Likai and Bi, Hongzhe and Qin, Yujia and Sun, Si and Li, Zhenyu and Li, Haolun and Li, Yongjian and Cong, Xin and Yan, Yukun and others},
  journal={arXiv preprint arXiv:2402.03009},
  year={2024}
}
```

## License

Code licensed under the Apache License v2.0
