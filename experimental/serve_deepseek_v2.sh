#!/bin/bash

# Serve DeepSeek V2 model with vLLM
# This script demonstrates how to serve the DeepSeek V2 model using vLLM's V1 engine

# MODEL_NAME="gaunernst/DeepSeek-V2-Lite-Chat-FP8"
MODEL_NAME="deepseek-ai/DeepSeek-V2-Lite-Chat"
HOST="0.0.0.0"
PORT=8006

DATA_PARALLEL_SIZE=3
DATA_PARALLEL_SIZE_LOCAL=$DATA_PARALLEL_SIZE

export VLLM_USE_V1=1
export VLLM_ALL2ALL_BACKEND="pplx"
export VLLM_USE_DEEP_GEMM=1

# Launch the vLLM server
vllm serve $MODEL_NAME --trust-remote-code \
    --disable-log-requests \
    --host $HOST \
    --port $PORT \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --enable-eplb \
    --num-redundant-experts 32 \
    --enforce-eager \
    --data-parallel-backend ray \
    --data-parallel-size $DATA_PARALLEL_SIZE \
    --data-parallel-size-local $DATA_PARALLEL_SIZE_LOCAL \
    --data-parallel-start-rank 0