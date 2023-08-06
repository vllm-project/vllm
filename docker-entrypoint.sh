#!/bin/sh

set -e

MODEL=${MODEL:-stabilityai/StableBeluga-7B}
TOKENIZER=${TOKENIZER:-$MODEL}
NUM_SHARD=${NUM_SHARD:-1}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-80}

echo "=== Config ==="
echo "MODEL=$MODEL"
echo "TOKENIZER=$TOKENIZER"
echo "NUM_SHARD=$NUM_SHARD"
echo "HOST=$HOST"
echo "PORT=$PORT"
echo "=============="

# Check if model starts with s3://
if [[ $MODEL == s3://* ]]; then
    echo "Downloading model from S3..."
    s5cmd cp $MODEL ./model

    if [[ $MODEL == $TOKENIZER ]]; then
        TOKENIZER=./model
        MODEL=./model
    else
        MODEL=./model
    fi
fi

# Check if tokenizer starts with s3://
if [[ $TOKENIZER == s3://* ]]; then
    echo "Downloading tokenizer from S3..."
    s5cmd cp $TOKENIZER ./tokenizer
    TOKENIZER=./tokenizer
fi

python -u -m vllm.entrypoints.api_server \
    --model $MODEL \
    --tokenizer $TOKENIZER \
    --host $HOST \
    --port $PORT \
    --tensor-parallel-size $NUM_SHARD
