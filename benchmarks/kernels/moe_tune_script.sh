#!/bin/bash

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


## ---- Mixtral fp8 tuning example ---- ##
python benchmark_moe.py --model /data/models/mistral-ai-models/Mixtral-8x22B-Instruct-v0.1-FP8/ --tp-size 1 --tune --dtype fp8_w8a8 
python benchmark_moe.py --model /data/models/mistral-ai-models/Mixtral-8x22B-Instruct-v0.1-FP8/ --tp-size 2 --tune --dtype fp8_w8a8
python benchmark_moe.py --model /data/models/mistral-ai-models/Mixtral-8x22B-Instruct-v0.1-FP8/ --tp-size 4 --tune --dtype fp8_w8a8
python benchmark_moe.py --model /data/models/mistral-ai-models/Mixtral-8x22B-Instruct-v0.1-FP8/ --tp-size 8 --tune --dtype fp8_w8a8


## ---- Mixtral fp16 tuning example ---- ##
# we don't need --dtype fp16; it has been set as default for rocm in the script.

python benchmark_moe.py --model /data/models/mistral-ai-models/Mixtral-8x22B-v0.1/ --tp-size 1 --tune
python benchmark_moe.py --model /data/models/mistral-ai-models/Mixtral-8x22B-v0.1/ --tp-size 2 --tune
python benchmark_moe.py --model /data/models/mistral-ai-models/Mixtral-8x22B-v0.1/ --tp-size 4 --tune
python benchmark_moe.py --model /data/models/mistral-ai-models/Mixtral-8x22B-v0.1/ --tp-size 8 --tune



## ---- After the tuning is finished ---- ##
# The tuning script saves the configurations in a json file at the same directory from where you launch the script.
# The name of the json file will look something like this: E=8,N=14336,device_name=AMD_Instinct_MI300X.json
# 
# [IMPORTANT] -> Once the tuning is complete, move the tuned config file(s) to the following path:
#                      vllm/vllm/model_executor/layers/fused_moe/configs/


## ---- Notes ---- ##
# 1. The tuned file is specific for a TP size. This means a tuned file obtained for --tp-size 8 can only be used when running the model under TP=8 setting.
# 2. The script uses Ray for multi-gpu tuning. Export HIP_VISIBLE_DEVICES accordingly to expose the required no. of GPUs and use multiple gpus for tuning.