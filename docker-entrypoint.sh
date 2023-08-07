#!/bin/sh

set -e

ENTRYPOINT=${ENTRYPOINT:-vllm.entrypoints.api_server}
MODEL=${MODEL:-stabilityai/StableBeluga-7B}
TOKENIZER=${TOKENIZER:-$MODEL}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-80}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-1}
BLOCK_SIZE=${BLOCK_SIZE:-16}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-2560}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}

# Enable HF transfer by default
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}

echo "=== Environment ==="
echo "ENTRYPOINT=$ENTRYPOINT"
echo "MODEL=$MODEL"
echo "TOKENIZER=$TOKENIZER"
echo "HOST=$HOST"
echo "PORT=$PORT"
echo "TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE"
echo "PIPELINE_PARALLEL_SIZE=$PIPELINE_PARALLEL_SIZE"
echo "BLOCK_SIZE=$BLOCK_SIZE"
echo "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
echo "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
echo "MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "HF_HUB_ENABLE_HF_TRANSFER=$HF_HUB_ENABLE_HF_TRANSFER"
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

python -u -m $ENTRYPOINT \
    --model $MODEL \
    --tokenizer $TOKENIZER \
    --host $HOST \
    --port $PORT \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --pipeline-parallel-size $PIPELINE_PARALLEL_SIZE \
    --block-size $BLOCK_SIZE \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
    --max-num-seqs $MAX_NUM_SEQS
