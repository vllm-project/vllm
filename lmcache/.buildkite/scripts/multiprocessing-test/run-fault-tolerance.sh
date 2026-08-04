#!/bin/bash
# Test fault tolerance: verify vLLM requests complete after LMCache server dies.
#
# Flow:
#   1. Warmup bench (calibrate timing)
#   2. Start bench in background, kill LMCache container mid-flight
#   3. Verify all prompts completed
#   4. Quick curl health check (vLLM still alive)
#
# NOTE: This test is destructive — it kills the LMCache container.
# Run it as the LAST test step.

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Configuration (reuse exported vars from run-mp-test.sh)
VLLM_PORT="${VLLM_PORT:-8000}"
LMCACHE_CONTAINER_NAME="${LMCACHE_CONTAINER_NAME:-lmcache-mp-test}"
VLLM_CONTAINER_NAME="${VLLM_CONTAINER_NAME:-vllm-mp-test}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-10000}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-1}"
RANDOM_SEED="${RANDOM_SEED:-42}"

# Output directory
FT_RESULTS_DIR="$RESULTS_DIR/fault_tolerance"
mkdir -p "$FT_RESULTS_DIR"

echo "=== Fault Tolerance Test ==="
echo "vLLM port: $VLLM_PORT"
echo "LMCache container: $LMCACHE_CONTAINER_NAME"
echo "Bench: $NUM_PROMPTS prompts, input_len=$RANDOM_INPUT_LEN"
echo ""

# Helper: run vllm bench serve
run_bench() {
    local description="$1"
    local result_file="$2"

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
        --result-dir "$FT_RESULTS_DIR" \
        --result-filename "$result_file" \
        --save-result

    local completed
    completed=$(python3 -c "
import json
with open('$FT_RESULTS_DIR/$result_file') as f:
    data = json.load(f)
print(data.get('completed', 0))
")

    echo "$description: $completed / $NUM_PROMPTS completed"
    if [ "$completed" -ne "$NUM_PROMPTS" ]; then
        echo "ERROR: Expected $NUM_PROMPTS completed, got $completed"
        return 1
    fi
    echo "All $NUM_PROMPTS prompts completed"
}

main() {
    setup_venv vllm openai

    # Step 1: Warmup bench (measure timing for kill calibration)
    echo "============================================"
    echo "=== Step 1: Warmup bench ==="
    echo "============================================"
    if ! run_bench "Warmup (with LMCache)" "bench_warmup.json"; then
        echo "Warmup bench failed"
        exit 1
    fi

    WARMUP_DURATION=$(python3 -c "import json; print(json.load(open('$FT_RESULTS_DIR/bench_warmup.json'))['duration'])")
    KILL_DELAY=$(python3 -c "print(max(3, int($WARMUP_DURATION * 0.4)))")
    echo "Warmup took ${WARMUP_DURATION}s. Will kill LMCache after ${KILL_DELAY}s."
    echo ""

    # Step 2: Bench with mid-flight LMCache kill
    echo "============================================"
    echo "=== Step 2: Bench with mid-flight LMCache kill ==="
    echo "============================================"

    run_bench "Mid-flight kill" "bench_midflight.json" &
    BENCH_PID=$!

    echo "Waiting ${KILL_DELAY}s before killing LMCache container..."
    sleep "$KILL_DELAY"

    echo "Killing LMCache container: $LMCACHE_CONTAINER_NAME"
    docker kill "$LMCACHE_CONTAINER_NAME" 2>/dev/null || true
    docker rm -f "$LMCACHE_CONTAINER_NAME" 2>/dev/null || true
    echo "LMCache container killed."

    echo "Waiting for bench to complete..."
    if ! wait "$BENCH_PID"; then
        echo "Bench did not complete after mid-flight LMCache kill"
        echo "--- vLLM container logs ---"
        docker logs --tail 50 "$VLLM_CONTAINER_NAME" 2>&1 || true
        exit 1
    fi
    echo ""

    # Step 3: Quick curl health check
    echo "============================================"
    echo "=== Step 3: Quick curl health check ==="
    echo "============================================"
    for i in 1 2 3; do
        if ! curl -sf --max-time 120 \
            "http://localhost:${VLLM_PORT}/v1/completions" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"$MODEL\",
                \"prompt\": \"Question: What is $i + $i?\\nAnswer:\",
                \"max_tokens\": 32,
                \"temperature\": 0
            }" > /dev/null 2>&1; then
            echo "Request $i failed - vLLM became unresponsive"
            exit 1
        fi
        echo "  Request $i: OK"
    done
    echo "vLLM still responsive"
    echo ""

    # Summary
    warmup_dur=$(python3 -c "import json; print(f\"{json.load(open('$FT_RESULTS_DIR/bench_warmup.json'))['duration']:.1f}\")")
    midflight_dur=$(python3 -c "import json; print(f\"{json.load(open('$FT_RESULTS_DIR/bench_midflight.json'))['duration']:.1f}\")")

    echo "============================================"
    echo "=== Fault Tolerance Test PASSED ==="
    echo "============================================"
    echo "  Warmup:            $NUM_PROMPTS/$NUM_PROMPTS in ${warmup_dur}s"
    echo "  Mid-flight kill:   $NUM_PROMPTS/$NUM_PROMPTS in ${midflight_dur}s (killed at +${KILL_DELAY}s)"
    echo "  Results: $FT_RESULTS_DIR/"
}

main "$@"
