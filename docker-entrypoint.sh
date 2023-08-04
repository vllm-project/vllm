#!/bin/sh

MODEL=${MODEL:-stabilityai/StableBeluga-7B}
NUM_SHARD=${NUM_SHARD:-1}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-80}

python -u -m vllm.entrypoints.api_server \
    --model $MODEL \
    --host $HOST \
    --port $PORT \
    --tensor-parallel-size $NUM_SHARD
