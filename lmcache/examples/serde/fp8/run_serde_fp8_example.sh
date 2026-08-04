#!/usr/bin/env bash
#
# End-to-end example: LMCache MP server + disk L2 adapter + fp8 serde + vLLM.
#
# Flow:
#   1. Start `lmcache server` with:
#        - L1 (CPU) cache enabled
#        - L2 disk (fs) adapter
#        - fp8 quantization serde on the L2 adapter
#   2. Start vLLM connected via LMCacheMPConnector
#   3. Send an inference request (cold path: data flows L1 -> L2 with serialize)
#   4. Force-clear L1 (CPU) cache via the lmcache HTTP API
#   5. Re-send the same request — L1 misses, L2 prefetch triggers deserialize
#
# Requirements:
#   - vLLM installed and runnable (`vllm serve`)
#   - lmcache CLI installed (`lmcache server --help`)
#   - 1 GPU available
#   - GPU + PyTorch with fp8 support (Hopper / Ada / RTX 40+ recommended)
set -e
set -o pipefail

# Prefer the LMCache repo's uv venv if present, so `lmcache` and `vllm` on PATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
if [ -d "${REPO_ROOT}/.venv/bin" ]; then
    export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
fi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
GPU_DEVICE="${GPU_DEVICE:-0}"

LMCACHE_PORT="${LMCACHE_PORT:-6555}"        # ZMQ port (vLLM <-> lmcache)
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"  # HTTP port (clear-cache, status)
VLLM_PORT="${VLLM_PORT:-8000}"

L1_SIZE_GB="${L1_SIZE_GB:-20}"              # CPU cache size

TMP_DIR="${TMP_DIR:-/tmp/lmcache_serde_example}"
L2_DISK_PATH="${L2_DISK_PATH:-${TMP_DIR}/disk}"
mkdir -p "$TMP_DIR"
mkdir -p "$L2_DISK_PATH"

# L2 adapter JSON: disk (fs) backend with fp8 serde enabled
L2_ADAPTER_JSON=$(cat <<EOF
{
  "type": "fs",
  "base_path": "${L2_DISK_PATH}",
  "serde": {"type": "fp8", "fp8_dtype": "float8_e4m3fn"}
}
EOF
)

# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------
LMCACHE_PID=""
VLLM_PID=""
cleanup() {
    echo "--- Cleaning up ---"
    [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null || true
    [ -n "$LMCACHE_PID" ] && kill "$LMCACHE_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

wait_for_url() {
    local url="$1"
    local timeout="${2:-300}"
    local elapsed=0
    while ! curl -sf "$url" > /dev/null 2>&1; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [ "$elapsed" -ge "$timeout" ]; then
            echo "Timeout waiting for $url"
            return 1
        fi
    done
}

# ---------------------------------------------------------------------------
# Step 1: Launch lmcache MP server (CPU L1 + disk L2 with fp8 serde)
# ---------------------------------------------------------------------------
echo "============================================"
echo "=== Step 1: Starting LMCache MP server ==="
echo "============================================"
echo "L1 (CPU): ${L1_SIZE_GB} GB"
echo "L2 (disk): ${L2_DISK_PATH}"
echo "Serde: fp8 (float8_e4m3fn)"

lmcache server \
    --l1-size-gb "$L1_SIZE_GB" \
    --eviction-policy LRU \
    --l2-store-policy default \
    --l2-prefetch-policy default \
    --l2-adapter "$L2_ADAPTER_JSON" \
    --port "$LMCACHE_PORT" \
    --http-port "$LMCACHE_HTTP_PORT" \
    2>&1 | tee "$TMP_DIR/lmcache.log" &
LMCACHE_PID=$!
echo "lmcache server PID=$LMCACHE_PID"

echo "Waiting for lmcache HTTP health..."
wait_for_url "http://localhost:${LMCACHE_HTTP_PORT}/healthcheck" 60 || {
    echo "lmcache failed to start. Last 50 lines of log:"
    tail -50 "$TMP_DIR/lmcache.log" || true
    exit 1
}
echo "lmcache server ready."

# ---------------------------------------------------------------------------
# Step 2: Launch vLLM with LMCacheMPConnector
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "=== Step 2: Starting vLLM ==="
echo "============================================"
echo "Model: $MODEL"

KV_TRANSFER_CONFIG=$(cat <<EOF
{
  "kv_connector": "LMCacheMPConnector",
  "kv_role": "kv_both",
  "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": {
    "lmcache.mp.port": ${LMCACHE_PORT},
    "lmcache.mp.mq_timeout": 10
  }
}
EOF
)

env -u VLLM_PORT \
    CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    PYTHONHASHSEED=0 \
vllm serve "$MODEL" \
    --port "$VLLM_PORT" \
    --no-enable-prefix-caching \
    --enforce-eager \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.6}" \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    2>&1 | tee "$TMP_DIR/vllm.log" &
VLLM_PID=$!
echo "vLLM PID=$VLLM_PID"

echo "Waiting for vLLM /v1/models (this can take a few minutes)..."
wait_for_url "http://localhost:${VLLM_PORT}/v1/models" 600 || {
    echo "vLLM failed to start. Last 50 lines:"
    tail -50 "$TMP_DIR/vllm.log" || true
    exit 1
}
echo "vLLM ready."

# ---------------------------------------------------------------------------
# Step 3: First inference — cold path (L1 -> L2 store with fp8 serialize)
# ---------------------------------------------------------------------------
# Prompt must be long enough to fill at least one 256-token LMCache chunk
# so KV actually gets stored to L2. We generate a ~1000+ token prompt.
PROMPT=""
for i in $(seq 1 8); do
    PROMPT+="The history and significance of the Roman empire spans more than a thousand years and profoundly shaped Western civilization. "
    PROMPT+="Its legal, architectural, linguistic, and political legacies persist to this day, influencing modern governments, languages, art, engineering, and law. "
    PROMPT+="The empire's trajectory from the founding of Rome through the Republic, the transition to the Principate under Augustus, the Pax Romana, the crisis of the third century, "
    PROMPT+="the Dominate under Diocletian, the adoption of Christianity under Constantine, the splitting into Western and Eastern halves, and the eventual collapse of the West "
    PROMPT+="is one of history's great narratives. Key figures include Julius Caesar, Augustus, Marcus Aurelius, Diocletian, Constantine, Justinian, and many others. "
done
PROMPT+="Tell me a long, detailed story about the rise, peak, and eventual fall of Rome, naming important figures and events."

echo ""
echo "============================================"
echo "=== Step 3: First inference (cold) ==="
echo "============================================"
echo "Expected: KV is computed, written to L1, then async-stored to L2 disk via fp8 serialize"

curl -s -X POST "http://localhost:${VLLM_PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL\",
        \"prompt\": \"$PROMPT\",
        \"max_tokens\": 32,
        \"temperature\": 0
    }" | python3 -c "import sys, json; d=json.load(sys.stdin); print('Response:', d['choices'][0]['text'][:200], '...')"

# Give the store controller a couple seconds to flush to disk
echo "Waiting 5s for L2 store to flush..."
sleep 5

# Sanity check: disk path should now contain some files
echo "Disk L2 contents:"
ls -lh "$L2_DISK_PATH" | head -10 || true

# ---------------------------------------------------------------------------
# Step 4: Clear L1 (CPU) cache
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "=== Step 4: Force-clearing L1 (CPU) cache ==="
echo "============================================"
curl -s -X POST "http://localhost:${LMCACHE_HTTP_PORT}/cache/clear" | python3 -m json.tool
echo "L1 cleared. Next request will miss L1 and trigger L2 prefetch."

# ---------------------------------------------------------------------------
# Step 5: Re-run the same request — triggers L2 prefetch + fp8 deserialize
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "=== Step 5: Second inference (L1 miss -> L2 prefetch) ==="
echo "============================================"
echo "Expected: L1 miss, L2 lookup hit, prefetch loads serialized bytes,"
echo "          fp8 deserialize back into KV, vLLM resumes from cache."

curl -s -X POST "http://localhost:${VLLM_PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL\",
        \"prompt\": \"$PROMPT\",
        \"max_tokens\": 32,
        \"temperature\": 0
    }" | python3 -c "import sys, json; d=json.load(sys.stdin); print('Response:', d['choices'][0]['text'][:200], '...')"

# ---------------------------------------------------------------------------
# Step 6: Show metrics / status
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "=== Step 6: LMCache status ==="
echo "============================================"
curl -s "http://localhost:${LMCACHE_HTTP_PORT}/status" \
    | python3 -m json.tool | head -80

echo ""
echo "============================================"
echo "Done. Logs are under: $TMP_DIR"
echo "============================================"
