#!/bin/bash

# Set the initial value of num_prefix
export NCCL_IGNORE_DISABLED_P2P=1
export HF_ENDPOINT=https://hf-mirror.com

num_prefix=1
HF_ENDPOINT=https://hf-mirror.com python benchmark/benchmark_prefix_translation.py \
    --model /mnt/data1/pzx/model/llama2-7b \
    --num-prefix-examples "$num_prefix" \
    --request-rate 24 \
    --duration 1200 \
    --n1 1.0 \
    &>> log.txt

# Loop over request_rate values for the first num_prefix
for request_rate in 1 24 32 40 48 50 51 52; do
    # Execute the Python script and append the output to log.txt
    export NCCL_IGNORE_DISABLED_P2P=1
    HF_ENDPOINT=https://hf-mirror.com python benchmark/benchmark_prefix_translation.py \
        --model /mnt/data1/pzx/model/llama2-7b \
        --num-prefix-examples "$num_prefix" \
        --request-rate "$request_rate" \
        --duration 1200 \
        --n1 1.0 \
        &>> log.txt
done

export NCCL_IGNORE_DISABLED_P2P=1

# Set the updated value of num_prefix
num_prefix=5

# Loop over request_rate values for the updated num_prefix
for request_rate in 1 24 36 40 41 42 43 44 45 46 47 48; do
    export NCCL_IGNORE_DISABLED_P2P=1
    # Execute the Python script and append the output to log.txt
    HF_ENDPOINT=https://hf-mirror.com python benchmark/benchmark_prefix_translation.py \
        --model /mnt/data1/pzx/model/llama2-7b \
        --num-prefix-examples "$num_prefix" \
        --request-rate "$request_rate" \
        --duration 1200 \
        --n1 1.0 \
        &>> log.txt
done
