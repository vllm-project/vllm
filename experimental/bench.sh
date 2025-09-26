#!/bin/bash

# MODEL_NAME="deepseek-ai/DeepSeek-V3.1"
MODEL_NAME="Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"
# MODEL_NAME="Qwen/Qwen3-235B-A22B-Thinking-2507-FP8"
HOST="localhost"
PORT=8006

vllm bench serve \
    --model $MODEL_NAME \
    --host $HOST \
    --port $PORT \
    --dataset-name random \
    --random-input-len 128 \
    --random-output-len 128 \
    --num-prompts 512
