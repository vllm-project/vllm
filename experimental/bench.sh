#!/bin/bash

MODEL_NAME="deepseek-ai/DeepSeek-V2-Lite-Chat"
HOST="localhost"
PORT=8006

vllm bench serve \
    --model $MODEL_NAME \
    --host $HOST \
    --port $PORT \
    --num-prompts 5
