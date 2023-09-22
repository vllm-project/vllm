#!/bin/bash

set -xe

cd /app/vllm
source activate vllm
echo 'Starting vllm api server...'
python -u -m vllm.entrypoints.api_server \
             --host 0.0.0.0 \
             --port 8000 \
             --model $MODEL_NAME \
             --tensor-parallel-size $NUM_GPUS
