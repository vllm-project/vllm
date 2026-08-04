#!/usr/bin/env bash
# Launch LMCache MP server, vLLM with LMCache, and vLLM baseline
# as native background processes (no Docker).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

# Configuration (inherited from run-mp-test.sh)
LMCACHE_PORT="${LMCACHE_PORT:-6555}"
vllm_port="${VLLM_PORT:-8000}"
vllm_baseline_port="${VLLM_BASELINE_PORT:-9000}"
CPU_BUFFER_SIZE="${CPU_BUFFER_SIZE:-80}"
MAX_WORKERS="${MAX_WORKERS:-4}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
BUILD_ID="${BUILD_ID:-local_$$}"

# K8s assigns exactly 2 GPUs as devices 0 and 1 (overridable for local runs).
GPU_FOR_VLLM="${GPU_FOR_VLLM:-0}"
GPU_FOR_BASELINE="${GPU_FOR_BASELINE:-1}"
echo "Using GPU $GPU_FOR_VLLM for vLLM with LMCache"
echo "Using GPU $GPU_FOR_BASELINE for vLLM baseline"

# Check GPU memory and set gpu-memory-utilization for very large GPUs.
# Without this, vLLM allocates so much KV cache that APC covers all prefixes
# and LMCache's cache path is never exercised, making the test pass vacuously.
GPU_MEMORY_UTIL_ARG=""
GPU_MEMORY_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${GPU_FOR_VLLM}" | tr -d ' ')
GPU_MEMORY_GB=$((GPU_MEMORY_MB / 1024))
echo "Detected GPU memory: ${GPU_MEMORY_GB}GB (${GPU_MEMORY_MB}MB)"

if [ -n "${GPU_MEMORY_UTILIZATION:-}" ]; then
    # Explicit override (e.g. large models like gemma-4-31B whose ~63GB of
    # weights alone exceed the default 0.5 fraction and would fail to load).
    echo "Using configured --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION}"
    GPU_MEMORY_UTIL_ARG="--gpu-memory-utilization ${GPU_MEMORY_UTILIZATION}"
elif [ "$GPU_MEMORY_GB" -gt 90 ]; then
    echo "GPU memory > 90GB, adding --gpu-memory-utilization 0.5"
    GPU_MEMORY_UTIL_ARG="--gpu-memory-utilization 0.5"
fi

# Attention backend for both vLLM servers. Defaults to FLASH_ATTN (what the
# batch-invariant lm_eval needs). Models with heterogeneous head dimensions
# (e.g. gemma-4) must NOT pin FLASH_ATTN -- set ATTENTION_BACKEND=auto so vLLM
# selects the backend itself (gemma-4 auto-forces TRITON_ATTN).
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASH_ATTN}"
ATTENTION_BACKEND_ARG=""
if [ -n "$ATTENTION_BACKEND" ] && [ "$ATTENTION_BACKEND" != "auto" ]; then
    ATTENTION_BACKEND_ARG="--attention-backend $ATTENTION_BACKEND"
fi

# Optionally run vLLM in eager mode (skip CUDA graph capture) for both servers.
# Off by default: verified to break the bit-exact run1 == run2 check in the
# determinism tests (lm_eval) -- eager changes the kernel path enough to diverge
# across the cold/warm batch difference even under VLLM_BATCH_INVARIANT. Enable
# (ENFORCE_EAGER=1) only for large models whose CUDA-graph capture would
# otherwise time out at launch (those tests use a tolerance, not bit-exactness).
ENFORCE_EAGER_ARG=""
if [ "${ENFORCE_EAGER:-0}" = "1" ] || [ "${ENFORCE_EAGER:-0}" = "true" ]; then
    ENFORCE_EAGER_ARG="--enforce-eager"
fi

# Pin max model length for both servers; defaults to "auto" (vLLM derives the
# largest length that fits KV memory). Verified not to affect the bit-exact
# determinism tests. Override via MAX_MODEL_LEN if a model needs a fixed length.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-auto}"
MAX_MODEL_LEN_ARG="--max-model-len ${MAX_MODEL_LEN}"

# LMCache server chunk size in tokens. Empty -> server default.
CHUNK_SIZE_ARG=""
if [ -n "${CHUNK_SIZE:-}" ]; then
    CHUNK_SIZE_ARG="--chunk-size ${CHUNK_SIZE}"
fi

# vLLM batch-invariant mode. On by default; GDN/Mamba backends do not support it.
BATCH_INVARIANT="${BATCH_INVARIANT:-1}"

# Mamba KV cache mode + prefix caching, set only for hybrid Mamba models.
MAMBA_ARGS=""
if [ -n "${MAMBA_CACHE_MODE:-}" ]; then
    MAMBA_ARGS="--mamba-cache-mode ${MAMBA_CACHE_MODE} --enable-prefix-caching"
fi

# Max tokens per scheduler step. Empty -> vLLM default.
MAX_NUM_BATCHED_TOKENS_ARG=""
if [ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]; then
    MAX_NUM_BATCHED_TOKENS_ARG="--max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS}"
fi

# L1 lazy allocation mode. Default is lazy (--l1-use-lazy). Set L1_USE_LAZY=false
# to disable lazy allocation, which enables POSIX SHM-backed L1 pool for the
# engine_driven SHM transfer path. When lazy is enabled (default), the SHM pool
# is disabled and engine_driven falls back to pickle transport.
L1_LAZY_ARG=""
if [ "${L1_USE_LAZY:-true}" = "false" ]; then
    L1_LAZY_ARG="--no-l1-use-lazy"
    echo "L1 lazy allocation disabled (SHM transport enabled)"
fi

# Store PIDs in a file so cleanup.sh can find them
PID_FILE="/tmp/lmcache_mp_pids_${BUILD_ID}"
> "$PID_FILE"

# ── 1. LMCache Multiprocess Server ──────────────────────────
echo "=== Launching LMCache MP server ==="
echo "Port: $LMCACHE_PORT"

# Optional GDS L1 slab tier (gds_* tests). When GDS_L1_PATH is set, the L1
# medium becomes an NVMe slab accessed via cuFile DMA instead of pinned DRAM;
# --l1-size-gb then sizes the slab. The path must be on a GDS-capable
# filesystem (local NVMe), provided by the /scratch hostPath mount.
GDS_L1_ARG=""
if [ -n "${GDS_L1_PATH:-}" ]; then
    echo "GDS L1 tier enabled; slab directory: $GDS_L1_PATH"
    GDS_L1_ARG="--gds-l1-path ${GDS_L1_PATH}"
fi

CUDA_VISIBLE_DEVICES="${GPU_FOR_VLLM}" \
lmcache server \
    --l1-size-gb "$CPU_BUFFER_SIZE" \
    --eviction-policy LRU \
    --max-workers "$MAX_WORKERS" \
    $CHUNK_SIZE_ARG \
    --port "$LMCACHE_PORT" \
    ${GDS_L1_ARG} \
    ${L1_LAZY_ARG} \
    > "/tmp/build_${BUILD_ID}_lmcache.log" 2>&1 &

LMCACHE_PID=$!
echo "$LMCACHE_PID" >> "$PID_FILE"
echo "LMCache MP server started (PID=$LMCACHE_PID)"

# Wait for LMCache to initialize
echo "Waiting for LMCache to initialize..."
sleep 10

# Unset VLLM_PORT so vLLM's internal get_open_port() picks a random
# ephemeral port for torch.distributed instead of trying serving_port+1.
# Without this, both instances fight over the same internal port.
unset VLLM_PORT

# ── 2. vLLM with LMCache ────────────────────────────────────
echo "=== Launching vLLM with LMCache ==="
echo "Model: $MODEL"
echo "Port: $vllm_port"

CUDA_VISIBLE_DEVICES="${GPU_FOR_VLLM}" \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_SERVER_DEV_MODE=1 \
VLLM_BATCH_INVARIANT=${BATCH_INVARIANT} \
PYTHONHASHSEED=0 \
vllm serve "$MODEL" \
    --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\", \"kv_role\":\"kv_both\", \"kv_load_failure_policy\": \"recompute\", \"kv_connector_extra_config\": {\"lmcache.mp.port\": $LMCACHE_PORT, \"lmcache.mp.mq_timeout\": 10}}" \
    $ATTENTION_BACKEND_ARG \
    --port "$vllm_port" \
    --no-async-scheduling \
    $MAX_MODEL_LEN_ARG \
    $ENFORCE_EAGER_ARG \
    $GPU_MEMORY_UTIL_ARG \
    $MAMBA_ARGS \
    $MAX_NUM_BATCHED_TOKENS_ARG \
    > "/tmp/build_${BUILD_ID}_vllm.log" 2>&1 &

VLLM_PID=$!
echo "$VLLM_PID" >> "$PID_FILE"
echo "vLLM with LMCache started (PID=$VLLM_PID)"

# ── 3. vLLM Baseline (without LMCache) ──────────────────────
# Only launched for tests that compare against a baseline (2-GPU pods).
# Single-GPU tests set LAUNCH_BASELINE=false and skip this entirely.
if [[ "${LAUNCH_BASELINE:-true}" == "true" ]]; then
    echo "=== Launching vLLM baseline ==="
    echo "Port: $vllm_baseline_port"

    CUDA_VISIBLE_DEVICES="${GPU_FOR_BASELINE}" \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    VLLM_SERVER_DEV_MODE=1 \
    VLLM_BATCH_INVARIANT=1 \
    PYTHONHASHSEED=0 \
    vllm serve "$MODEL" \
        $ATTENTION_BACKEND_ARG \
        --port "$vllm_baseline_port" \
        --no-async-scheduling \
        $MAX_MODEL_LEN_ARG \
        $ENFORCE_EAGER_ARG \
        $GPU_MEMORY_UTIL_ARG \
        > "/tmp/build_${BUILD_ID}_vllm_baseline.log" 2>&1 &

    VLLM_BASELINE_PID=$!
    echo "$VLLM_BASELINE_PID" >> "$PID_FILE"
    echo "vLLM baseline started (PID=$VLLM_BASELINE_PID)"
else
    echo "=== Skipping vLLM baseline (LAUNCH_BASELINE=false, 1-GPU test) ==="
fi

echo "=== All processes launched ==="
echo "PIDs: LMCache=$LMCACHE_PID, vLLM=$VLLM_PID, Baseline=${VLLM_BASELINE_PID:-skipped}"
