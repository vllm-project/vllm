#!/usr/bin/env bash
# Test LMCache fault tolerance: verify vLLM requests complete after
# the LMCache MP server is killed mid-flight.
#
# Flow:
#   1. Run a warmup bench (measures baseline timing)
#   2. Run bench again, killing LMCache server mid-flight
#   3. Run bench fully without LMCache server
#   4. Verify all prompts completed in every phase
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

# Configuration (inherited from run-mp-test.sh)
VLLM_PORT="${VLLM_PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
BUILD_ID="${BUILD_ID:-local_$$}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
LMCACHE_PORT="${LMCACHE_PORT:-6555}"

# Bench parameters
NUM_PROMPTS="${NUM_PROMPTS:-50}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-10000}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-1}"
RANDOM_SEED="${RANDOM_SEED:-42}"

CPU_BUFFER_SIZE="${CPU_BUFFER_SIZE:-80}"
MAX_WORKERS="${MAX_WORKERS:-4}"

# Output directory
FT_DIR="$RESULTS_DIR/fault_tolerance"
mkdir -p "$FT_DIR"

echo "=== Fault Tolerance Test ==="
echo "Model: $MODEL"
echo "vLLM Port: $VLLM_PORT"
echo "LMCache Port: $LMCACHE_PORT"
echo "Bench: $NUM_PROMPTS prompts, input_len=$RANDOM_INPUT_LEN, output_len=$RANDOM_OUTPUT_LEN"
echo "Results dir: $FT_DIR"
echo ""

# ── Step 0: Restart LMCache + vLLM with fresh state ─────────
# Previous steps may have left processes in an unknown state.
# Restart both so vLLM registers its GPU context with the new server.
echo "============================================"
echo "=== Restarting LMCache + vLLM ==="
echo "============================================"

PID_FILE="/tmp/lmcache_mp_pids_${BUILD_ID}"
GPU_DEVICE="${GPU_FOR_VLLM:-0}"

# Kill existing LMCache + vLLM (keep baseline on line 3)
if [ -f "$PID_FILE" ]; then
    OLD_LMCACHE_PID=$(sed -n '1p' "$PID_FILE")
    OLD_VLLM_PID=$(sed -n '2p' "$PID_FILE")
    for pid in $OLD_LMCACHE_PID $OLD_VLLM_PID; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "Killing PID $pid"
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    sleep 2
fi

# Launch LMCache with L1 config
CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" \
lmcache server \
    --l1-size-gb "$CPU_BUFFER_SIZE" \
    --eviction-policy LRU \
    --max-workers "$MAX_WORKERS" \
    --port "$LMCACHE_PORT" \
    > "/tmp/build_${BUILD_ID}_lmcache_ft.log" 2>&1 &

NEW_LMCACHE_PID=$!
echo "LMCache server started (PID=$NEW_LMCACHE_PID)"
sleep 10

# Launch vLLM with LMCache
GPU_MEMORY_UTIL_ARG=""
GPU_MEMORY_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${GPU_DEVICE}" | tr -d ' ')
GPU_MEMORY_GB=$((GPU_MEMORY_MB / 1024))
if [ "$GPU_MEMORY_GB" -gt 90 ]; then
    GPU_MEMORY_UTIL_ARG="--gpu-memory-utilization 0.5"
fi

env -u VLLM_PORT \
    CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    VLLM_SERVER_DEV_MODE=1 \
    VLLM_BATCH_INVARIANT=1 \
    PYTHONHASHSEED=0 \
vllm serve "$MODEL" \
    --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\", \"kv_role\":\"kv_both\", \"kv_load_failure_policy\": \"recompute\", \"kv_connector_extra_config\": {\"lmcache.mp.port\": $LMCACHE_PORT, \"lmcache.mp.mq_timeout\": 10}}" \
    --attention-backend FLASH_ATTN \
    --port "$VLLM_PORT" \
    --no-async-scheduling \
    $GPU_MEMORY_UTIL_ARG \
    > "/tmp/build_${BUILD_ID}_vllm_ft.log" 2>&1 &

NEW_VLLM_PID=$!
echo "vLLM started (PID=$NEW_VLLM_PID)"

# Update PID file
if [ -f "$PID_FILE" ]; then
    sed -i "1s/.*/$NEW_LMCACHE_PID/" "$PID_FILE"
    sed -i "2s/.*/$NEW_VLLM_PID/" "$PID_FILE"
else
    echo "$NEW_LMCACHE_PID" > "$PID_FILE"
    echo "$NEW_VLLM_PID" >> "$PID_FILE"
fi

if ! wait_for_server "$VLLM_PORT" 300; then
    echo "vLLM failed to start"
    tail -50 "/tmp/build_${BUILD_ID}_lmcache_ft.log" || true
    tail -50 "/tmp/build_${BUILD_ID}_vllm_ft.log" || true
    exit 1
fi
echo ""

# ── Helpers ──────────────────────────────────────────────────

run_bench() {
    local description="$1"
    local result_file="$2"

    echo ""
    echo "--- $description ---"

    vllm bench serve \
        --seed "$RANDOM_SEED" \
        --port "$VLLM_PORT" \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len "$RANDOM_INPUT_LEN" \
        --random-output-len "$RANDOM_OUTPUT_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --ignore-eos \
        --backend openai-chat \
        --endpoint /v1/chat/completions \
        --result-dir "$FT_DIR" \
        --result-filename "$result_file" \
        --save-result

    local completed
    completed=$(python3 -c "
import json
with open('$FT_DIR/$result_file') as f:
    data = json.load(f)
print(data.get('completed', 0))
")

    echo "$description: $completed / $NUM_PROMPTS completed"

    if [ "$completed" -ne "$NUM_PROMPTS" ]; then
        echo "FAIL: Expected $NUM_PROMPTS completed, got $completed"
        return 1
    fi

    echo "PASS: All $NUM_PROMPTS prompts completed"
    return 0
}

get_lmcache_pid() {
    local pid_file="/tmp/lmcache_mp_pids_${BUILD_ID}"
    if [[ -f "$pid_file" ]]; then
        head -1 "$pid_file"
    fi
}

# ── Step 1: Warmup bench ─────────────────────────────────────
echo "============================================"
echo "=== Fault Tolerance Step 1: Warmup bench ==="
echo "============================================"

if ! run_bench "Warmup (with LMCache)" "ft_warmup.json"; then
    echo "FAIL: Warmup bench failed"
    exit 1
fi

# Extract duration to calibrate kill timing
WARMUP_DURATION=$(python3 -c "import json; print(json.load(open('$FT_DIR/ft_warmup.json'))['duration'])")
KILL_DELAY=$(python3 -c "print(max(3, int($WARMUP_DURATION * 0.4)))")
echo "Warmup took ${WARMUP_DURATION}s. Will kill LMCache after ${KILL_DELAY}s in next run."

# ── Step 2: Bench with mid-flight LMCache kill ───────────────
echo ""
echo "============================================"
echo "=== Fault Tolerance Step 2: Mid-flight kill ==="
echo "============================================"

LMCACHE_PID=$(get_lmcache_pid)
if [ -z "$LMCACHE_PID" ] || ! kill -0 "$LMCACHE_PID" 2>/dev/null; then
    echo "FAIL: LMCache server not running (PID=$LMCACHE_PID)"
    exit 1
fi

echo "LMCache server PID: $LMCACHE_PID"
echo "Will kill after ${KILL_DELAY}s into bench."

# Start bench in background
run_bench "Mid-flight kill" "ft_midflight.json" &
BENCH_PID=$!

# Wait, then kill LMCache
sleep "$KILL_DELAY"
echo "Killing LMCache server (PID: $LMCACHE_PID)..."
kill "$LMCACHE_PID" 2>/dev/null
wait "$LMCACHE_PID" 2>/dev/null || true
echo "LMCache server killed at +${KILL_DELAY}s."

# Wait for bench to finish
echo "Waiting for bench to complete..."
if ! wait "$BENCH_PID"; then
    echo "FAIL: Bench did not complete after mid-flight LMCache kill."
    echo "--- vLLM log (last 50 lines) ---"
    tail -50 "/tmp/build_${BUILD_ID}_vllm.log" 2>/dev/null || true
    exit 1
fi

# ── Step 3: Bench fully without LMCache server ───────────────
echo ""
echo "============================================"
echo "=== Fault Tolerance Step 3: Without LMCache ==="
echo "============================================"

if ! run_bench "Without LMCache" "ft_without_lmcache.json"; then
    echo "FAIL: Bench failed without LMCache server."
    echo "--- vLLM log (last 50 lines) ---"
    tail -50 "/tmp/build_${BUILD_ID}_vllm.log" 2>/dev/null || true
    exit 1
fi

# ── Summary ──────────────────────────────────────────────────
echo ""
echo "============================================"
echo "=== Fault Tolerance Test PASSED ==="
echo "============================================"

warmup_completed=$(python3 -c "import json; print(json.load(open('$FT_DIR/ft_warmup.json'))['completed'])")
warmup_duration=$(python3 -c "import json; print(f\"{json.load(open('$FT_DIR/ft_warmup.json'))['duration']:.1f}\")")
midflight_completed=$(python3 -c "import json; print(json.load(open('$FT_DIR/ft_midflight.json'))['completed'])")
midflight_duration=$(python3 -c "import json; print(f\"{json.load(open('$FT_DIR/ft_midflight.json'))['duration']:.1f}\")")
without_completed=$(python3 -c "import json; print(json.load(open('$FT_DIR/ft_without_lmcache.json'))['completed'])")
without_duration=$(python3 -c "import json; print(f\"{json.load(open('$FT_DIR/ft_without_lmcache.json'))['duration']:.1f}\")")

echo "  Warmup (with LMCache):  $warmup_completed/$NUM_PROMPTS in ${warmup_duration}s"
echo "  Mid-flight kill:        $midflight_completed/$NUM_PROMPTS in ${midflight_duration}s (killed at +${KILL_DELAY}s)"
echo "  Without LMCache:        $without_completed/$NUM_PROMPTS in ${without_duration}s"
echo ""
