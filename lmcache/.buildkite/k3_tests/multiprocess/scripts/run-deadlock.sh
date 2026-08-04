#!/usr/bin/env bash
# Self-contained deadlock regression test.
#
# Launches DeepSeek-V2-Lite-Chat with TP=2 (both GPUs) + LMCache server,
# sends 50 requests with ~30K token prefixes, and verifies they all
# complete within 3 minutes.  A CUDA-driver/GIL deadlock would cause
# requests to hang indefinitely, failing the timeout.
#
# This test is self-contained: it handles its own server lifecycle
# instead of using the standard launch-processes.sh / wait-for-servers.sh.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

# ── Configuration ───────────────────────────────────────────
MODEL="deepseek-ai/DeepSeek-V2-Lite-Chat"
LMCACHE_PORT="${LMCACHE_PORT:-15554}"
VLLM_PORT="${VLLM_PORT:-8000}"
BUILD_ID="${BUILD_ID:-local_$$}"
PID_FILE="/tmp/lmcache_mp_pids_${BUILD_ID}"
TIMEOUT_SECONDS=180   # 3 minutes

# ── Install py-spy for deadlock diagnosis ──────────────────
echo "=== Installing py-spy ==="
uv pip install py-spy
PY_SPY="$(which py-spy)"
echo "py-spy installed at: $PY_SPY"

PYSPY_LOG="/tmp/build_${BUILD_ID}_pyspy.log"

# ── Helper: dump stacks of server processes via py-spy ─────
dump_stacks() {
    echo "" | tee -a "$PYSPY_LOG"
    echo "=== py-spy stack dump (native + Python) ===" | tee -a "$PYSPY_LOG"

    if kill -0 "$LMCACHE_PID" 2>/dev/null; then
        echo "" | tee -a "$PYSPY_LOG"
        echo "--- LMCache server (PID=$LMCACHE_PID) ---" | tee -a "$PYSPY_LOG"
        sudo "$PY_SPY" dump --pid "$LMCACHE_PID" --native 2>&1 | tee -a "$PYSPY_LOG" || true
    fi

    # Copy to repo root so cleanup.sh collects it as a Buildkite artifact
    cp "$PYSPY_LOG" "${REPO_ROOT}/build_${BUILD_ID}_pyspy.log" 2>/dev/null || true
}

# ── 1. Launch LMCache server ───────────────────────────────
echo "=== Launching LMCache server ==="
echo "Port: $LMCACHE_PORT"

lmcache server \
    --host localhost \
    --port "$LMCACHE_PORT" \
    --chunk-size 256 \
    --l1-size-gb 50 \
    --eviction-policy LRU \
    --max-workers 2 \
    > "/tmp/build_${BUILD_ID}_lmcache.log" 2>&1 &

LMCACHE_PID=$!
echo "$LMCACHE_PID" >> "$PID_FILE"
echo "LMCache server started (PID=$LMCACHE_PID)"
sleep 10

# ── 2. Launch vLLM with DeepSeek TP=2 ─────────────────────
echo "=== Launching vLLM (DeepSeek TP=2) ==="
echo "Model: $MODEL"
echo "Port: $VLLM_PORT"

# Save VLLM_PORT before unsetting — vLLM's internal get_open_port()
# would otherwise collide with the serving port for torch.distributed.
SAVED_VLLM_PORT="$VLLM_PORT"
unset VLLM_PORT

FLASHINFER_DISABLE_VERSION_CHECK=1 \
VLLM_SERVER_DEV_MODE=1 \
vllm serve "$MODEL" \
    --tensor-parallel-size 2 \
    --distributed-executor-backend mp \
    --block-size 64 \
    --trust-remote-code \
    --load-format dummy \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.8 \
    --max-model-len 65536 \
    --hf-overrides '{"max_position_embeddings":65536}' \
    --max-num-seqs 32 \
    --max-num-batched-tokens 16000 \
    --scheduling-policy fcfs \
    --port "$SAVED_VLLM_PORT" \
    --enforce-eager \
    --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\", \"kv_role\":\"kv_both\", \"kv_load_failure_policy\": \"recompute\", \"kv_connector_extra_config\": {\"lmcache.mp.port\": $LMCACHE_PORT, \"lmcache.mp.mq_timeout\": 60}}" \
    > "/tmp/build_${BUILD_ID}_vllm.log" 2>&1 &

VLLM_PID=$!
echo "$VLLM_PID" >> "$PID_FILE"
echo "vLLM started (PID=$VLLM_PID)"

VLLM_PORT="$SAVED_VLLM_PORT"

# ── 3. Wait for vLLM to be ready ──────────────────────────
echo "=== Waiting for vLLM to be ready ==="
if ! wait_for_server "$VLLM_PORT" 600; then
    echo "vLLM failed to start. Last 100 lines of log:"
    tail -100 "/tmp/build_${BUILD_ID}_vllm.log" 2>/dev/null || true
    exit 1
fi

# ── 4. Run benchmark with timeout ─────────────────────────
echo "=== Running lmcache bench engine (random-prefill, 50 reqs, ~30K tokens) ==="
echo "Timeout: ${TIMEOUT_SECONDS}s"

if ! timeout "$TIMEOUT_SECONDS" lmcache bench engine \
        --engine-url "http://localhost:${VLLM_PORT}" \
        --workload random-prefill \
        --tokens-per-gb-kvcache 6000 \
        --rp-request-length 30000 \
        --rp-num-requests 50 \
        --no-interactive \
        --no-csv \
        -q; then
    echo "FAIL: Benchmark failed or timed out (possible deadlock)"
    echo ""
    echo "=== LMCache log (last 50 lines) ==="
    tail -50 "/tmp/build_${BUILD_ID}_lmcache.log" 2>/dev/null || true
    echo ""
    echo "=== vLLM log (last 50 lines) ==="
    tail -50 "/tmp/build_${BUILD_ID}_vllm.log" 2>/dev/null || true
    exit 1
fi

echo ""
echo "=== Benchmark completed within ${TIMEOUT_SECONDS}s ==="
echo "PASS: No deadlock detected"
