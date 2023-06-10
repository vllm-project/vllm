#!/bin/bash

PORT=8001
MODEL=$1  # huggyllama/llama-13b

docker run --gpus all --shm-size 1g -p $PORT:80 \
           -v $PWD/data:/data \
           ghcr.io/huggingface/text-generation-inference:0.8 \
           --model-id $MODEL \
           --sharded false  \
           --max-input-length 2047 \
           --max-total-tokens 2048 \
           --max-batch-total-tokens 100
