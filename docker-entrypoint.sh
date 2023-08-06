#!/bin/sh

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
    MODEL=./model
fi

python -u -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --tokenizer $TOKENIZER \
    --host $HOST \
    --port $PORT \
    --tensor-parallel-size $NUM_SHARD
