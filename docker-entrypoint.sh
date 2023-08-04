#!/bin/sh

MODEL=${MODEL:-stabilityai/StableBeluga-7B}
NUM_SHARD=${NUM_SHARD:-1}

echo 'Starting vllm sagemaker server...'

conda run -n vllm \
    python -u -m vllm.entrypoints.api_server \
    --model $MODEL \
    --tensor-parallel-size $NUM_SHARD
