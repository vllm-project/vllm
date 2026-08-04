#!/bin/bash
# Launch LMCache and vLLM containers for multiprocessing tests
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Container names (exported so cleanup script can use them)
export LMCACHE_CONTAINER_NAME="${LMCACHE_CONTAINER_NAME:-lmcache-mp-test}"
export VLLM_CONTAINER_NAME="${VLLM_CONTAINER_NAME:-vllm-mp-test}"
export VLLM_BASELINE_CONTAINER_NAME="${VLLM_BASELINE_CONTAINER_NAME:-vllm-baseline-test}"

# Configuration
LMCACHE_PORT="${LMCACHE_PORT:-6555}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASELINE_PORT="${VLLM_BASELINE_PORT:-9000}"

CPU_BUFFER_SIZE="${CPU_BUFFER_SIZE:-80}"
MAX_WORKERS="${MAX_WORKERS:-4}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"

# Check GPU memory and set gpu-memory-utilization if > 100GB
GPU_MEMORY_UTIL_ARG=""
GPU_MEMORY_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
GPU_MEMORY_GB=$((GPU_MEMORY_MB / 1024))
echo "Detected GPU memory: ${GPU_MEMORY_GB}GB (${GPU_MEMORY_MB}MB)"

if [ "$GPU_MEMORY_GB" -gt 90 ]; then
    echo "GPU memory > 100GB, adding --gpu-memory-utilization 0.5"
    GPU_MEMORY_UTIL_ARG="--gpu-memory-utilization 0.5"
fi

echo "=== Launching LMCache container ==="
echo "Container name: $LMCACHE_CONTAINER_NAME"
echo "Port: $LMCACHE_PORT"

docker run -d \
    --name "$LMCACHE_CONTAINER_NAME" \
    --runtime nvidia \
    --gpus all \
    --network host \
    --ipc host \
    --entrypoint /opt/venv/bin/python3 \
    lmcache/vllm-openai:test \
    -m lmcache.v1.multiprocess.server \
    --cpu-buffer-size "$CPU_BUFFER_SIZE" \
    --max-workers "$MAX_WORKERS" \
    --port "$LMCACHE_PORT"

echo "LMCache container started"

# Wait a bit for LMCache to initialize
echo "Waiting for LMCache to initialize..."
sleep 10

echo "=== Launching vLLM container ==="
echo "Container name: $VLLM_CONTAINER_NAME"
echo "Model: $MODEL"

docker run -d \
    --name "$VLLM_CONTAINER_NAME" \
    --runtime nvidia \
    --gpus all \
    --volume /data1/yihua-model:/root/.cache/huggingface \
    --network host \
    --ipc host \
    --env CUDA_VISIBLE_DEVICES=0 \
    --env VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    --env VLLM_ATTENTION_BACKEND=FLASH_ATTN \
    --env VLLM_BATCH_INVARIANT=1 \
    --env PYTHONHASHSEED=0 \
    lmcache/vllm-openai:test \
    "$MODEL" \
    --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\", \"kv_role\":\"kv_both\", \"kv_load_failure_policy\": \"recompute\", \"kv_connector_extra_config\": {\"lmcache.mp.port\": $LMCACHE_PORT, \"lmcache.mp.mq_timeout\": 10}}" \
    --port "$VLLM_PORT" \
    --no-async-scheduling \
    $GPU_MEMORY_UTIL_ARG

echo "vLLM container started"

echo "=== Launching vLLM baseline container (without LMCache) ==="
echo "Container name: $VLLM_BASELINE_CONTAINER_NAME"
echo "Port: $VLLM_BASELINE_PORT"

docker run -d \
    --name "$VLLM_BASELINE_CONTAINER_NAME" \
    --runtime nvidia \
    --gpus all \
    --volume /data1/yihua-model:/root/.cache/huggingface \
    --network host \
    --ipc host \
    --env CUDA_VISIBLE_DEVICES=1 \
    --env VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    --env VLLM_ATTENTION_BACKEND=FLASH_ATTN \
    --env VLLM_BATCH_INVARIANT=1 \
    --env PYTHONHASHSEED=0 \
    lmcache/vllm-openai:test \
    "$MODEL" \
    --port "$VLLM_BASELINE_PORT" \
    --no-async-scheduling \
    $GPU_MEMORY_UTIL_ARG

echo "vLLM baseline container started"

echo "=== Containers launched ==="
docker ps --filter "name=$LMCACHE_CONTAINER_NAME" --filter "name=$VLLM_CONTAINER_NAME" --filter "name=$VLLM_BASELINE_CONTAINER_NAME"

