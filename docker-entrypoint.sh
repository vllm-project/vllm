#!/bin/sh

MODEL=${MODEL:-stabilityai/StableBeluga-7B}
TOKENIZER=${TOKENIZER:-$MODEL}
NUM_SHARD=${NUM_SHARD:-1}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-80}

python -u -m vllm.entrypoints.api_server \
    --model $MODEL \
    --tokenizer $TOKENIZER \
    --host $HOST \
    --port $PORT \
    --tensor-parallel-size $NUM_SHARD
