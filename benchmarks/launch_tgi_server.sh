#!/bin/bash

PORT=8000
MODEL=$1
TOKENS=$2

docker run --gpus all --shm-size 1g -p $PORT:80 \
           -v $PWD/data:/data \
           ghcr.io/huggingface/text-generation-inference:0.8 \
           --model-id $MODEL \
           --sharded false  \
           --max-input-length 1024 \
           --max-total-tokens 2048 \
           --max-best-of 5 \
           --max-concurrent-requests 5000 \
           --max-batch-total-tokens $TOKENS
