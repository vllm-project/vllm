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
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-6556}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASELINE_PORT="${VLLM_BASELINE_PORT:-9000}"

CPU_BUFFER_SIZE="${CPU_BUFFER_SIZE:-80}"
MAX_WORKERS="${MAX_WORKERS:-4}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"

# Pick 2 free GPUs dynamically for the test
echo "=== Selecting free GPUs for testing ==="
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/.buildkite/scripts/pick-free-gpu.sh" 70000 2
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "❌ Failed to select 2 free GPUs"
    exit 1
fi

# Parse the selected GPUs into an array
IFS=',' read -ra SELECTED_GPUS <<< "$CUDA_VISIBLE_DEVICES"
if [ ${#SELECTED_GPUS[@]} -ne 2 ]; then
    echo "❌ Expected 2 GPUs, but got ${#SELECTED_GPUS[@]}: ${CUDA_VISIBLE_DEVICES}"
    exit 1
fi

GPU_FOR_VLLM="${SELECTED_GPUS[0]}"
GPU_FOR_BASELINE="${SELECTED_GPUS[1]}"
echo "Selected GPU ${GPU_FOR_VLLM} for vLLM with LMCache"
echo "Selected GPU ${GPU_FOR_BASELINE} for vLLM baseline"

# Check GPU memory and set gpu-memory-utilization if > 100GB
GPU_MEMORY_UTIL_ARG=""
GPU_MEMORY_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${GPU_FOR_VLLM}" | tr -d ' ')
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
    --gpus "device=${GPU_FOR_VLLM}" \
    --network host \
    --ipc host \
    --entrypoint /opt/venv/bin/python3 \
    lmcache/vllm-openai:test \
    -m lmcache.v1.multiprocess.http_server \
    --l1-size-gb "$CPU_BUFFER_SIZE" \
    --eviction-policy LRU \
    --max-workers "$MAX_WORKERS" \
    --port "$LMCACHE_PORT" \
    --http-port "$LMCACHE_HTTP_PORT"

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
    --gpus "device=${GPU_FOR_VLLM}" \
    --volume ~/.cache/huggingface:/root/.cache/huggingface \
    --network host \
    --ipc host \
    --env VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    --env VLLM_BATCH_INVARIANT=1 \
    --env PYTHONHASHSEED=0 \
    lmcache/vllm-openai:test \
    "$MODEL" \
    --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\", \"kv_role\":\"kv_both\", \"kv_load_failure_policy\": \"recompute\", \"kv_connector_extra_config\": {\"lmcache.mp.port\": $LMCACHE_PORT, \"lmcache.mp.mq_timeout\": 10}}" \
    --attention-backend FLASH_ATTN \
    --port "$VLLM_PORT" \
    $GPU_MEMORY_UTIL_ARG

echo "vLLM container started"

echo "=== Launching vLLM baseline container (without LMCache) ==="
echo "Container name: $VLLM_BASELINE_CONTAINER_NAME"
echo "Port: $VLLM_BASELINE_PORT"

docker run -d \
    --name "$VLLM_BASELINE_CONTAINER_NAME" \
    --runtime nvidia \
    --gpus "device=${GPU_FOR_BASELINE}" \
    --volume ~/.cache/huggingface:/root/.cache/huggingface \
    --network host \
    --ipc host \
    --env VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    --env VLLM_BATCH_INVARIANT=1 \
    --env PYTHONHASHSEED=0 \
    lmcache/vllm-openai:test \
    "$MODEL" \
    --port "$VLLM_BASELINE_PORT" \
    --attention-backend FLASH_ATTN \
    $GPU_MEMORY_UTIL_ARG

echo "vLLM baseline container started"

echo "=== Containers launched ==="
docker ps --filter "name=$LMCACHE_CONTAINER_NAME" --filter "name=$VLLM_CONTAINER_NAME" --filter "name=$VLLM_BASELINE_CONTAINER_NAME"

