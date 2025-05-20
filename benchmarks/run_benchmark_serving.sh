#!/bin/bash

cd /home/test/test01/zhushengguang/CODES/vllm/benchmarks
#MODEL="/home/test/test01/Llama-2-7b-chat-hf"
MODEL="/home/test/test01/zhushengguang/models/DeepSeek-R1-Distill-Qwen-7B"
PORT=8000
# DATASET="random"  # options: sharegpt, random, sonnet, hf, burstgpt
DATASET="hf"
# DATASET_PATH="home/test/test01/zhushengguang/Datasets/NuminaMath-CoT"
DATASET_PATH="AI-MO/aimo-validation-aime"
# DATASET_PATH="AI-MO/NuminaMath-CoT"
NUM_PROMPTS=4096


#使用cuda12.4
# export PATH=$HOME/cuda-12.4/bin:$PATH
# export CUDA_HOME=$HOME/cuda-12.4
# export CUDA_PATH=$HOME/cuda-12.4
# export CUDACXX=$HOME/cuda-12.4/bin/nvcc
# export C_INCLUDE_PATH=$HOME/cuda-12.4/include:$C_INCLUDE_PATH
# export CPLUS_INCLUDE_PATH=$HOME/cuda-12.4/include:$CPLUS_INCLUDE_PATH
# export LD_LIBRARY_PATH=$HOME/.local/lib64:$HOME/cuda-12.4/lib64:$LD_LIBRARY_PATH

export HF_ENDPOINT=https://hf-mirror.com
#echo "vllm serve ${MODEL} -tp 1 -dp 8"

python3 benchmark_serving.py \
    --model ${MODEL} \
    --host 0.0.0.0 \
    --port ${PORT} \
    --dataset-name ${DATASET} \
    --dataset-path ${DATASET_PATH} \
    --num-prompts ${NUM_PROMPTS} \
    --ignore-eos \
    # --random-input-len 1024 \
    # --random-output-len 128 \
    # --save-result \
    # --result-dir ./benchmark_results \
    # --append-result \
    # --save-detailed \
