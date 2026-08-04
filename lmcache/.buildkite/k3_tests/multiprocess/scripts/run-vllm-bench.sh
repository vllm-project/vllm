#!/usr/bin/env bash
# Run vllm bench serve test against both vLLM servers.
# Compares performance between LMCache-enabled and baseline vLLM.
# Adapted from the old Docker-based run-vllm-bench.sh.
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

# Configuration
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASELINE_PORT="${VLLM_BASELINE_PORT:-9000}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-10000}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-1}"
BUILD_ID="${BUILD_ID:-local_$$}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"

# Expected values
EXPECTED_TOTAL_INPUT_TOKENS=$((NUM_PROMPTS * RANDOM_INPUT_LEN))
EXPECTED_COMPLETED=$NUM_PROMPTS
MAX_SLOWDOWN_PERCENT=5

# Reproducible seed
RANDOM_SEED="${RANDOM_SEED:-$(date +%s)}"

# Output directory
VLLM_BENCH_DIR="$RESULTS_DIR/vllm_bench"

echo "=== vLLM Bench Serve Test ==="
echo "Model: $MODEL"
echo "vLLM Port (with LMCache): $VLLM_PORT"
echo "vLLM Baseline Port (without LMCache): $VLLM_BASELINE_PORT"
echo "Number of prompts: $NUM_PROMPTS"
echo "Random input length: $RANDOM_INPUT_LEN"
echo "Random output length: $RANDOM_OUTPUT_LEN"
echo "Results dir: $VLLM_BENCH_DIR"
echo ""

mkdir -p "$VLLM_BENCH_DIR"

run_vllm_bench() {
    local port="$1"
    local result_filename="$2"
    local description="$3"
    local seed="$4"

    echo "=== Running vllm bench serve ($description) ==="
    echo "Port: $port, Seed: $seed"

    vllm bench serve \
        --seed "$seed" \
        --port "$port" \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len "$RANDOM_INPUT_LEN" \
        --random-output-len "$RANDOM_OUTPUT_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --ignore-eos \
        --backend openai-chat \
        --endpoint /v1/chat/completions \
        --result-dir "$VLLM_BENCH_DIR" \
        --result-filename "$result_filename" \
        --save-result

    echo "$description benchmark completed"
    echo ""
}

extract_json_field() {
    local json_file="$1"
    local field="$2"
    python3 -c "
import json
with open('$json_file', 'r') as f:
    data = json.load(f)
print(data.get('$field', 'null'))
"
}

verify_results() {
    local lmcache_result="$VLLM_BENCH_DIR/lmcache.json"
    local baseline_result="$VLLM_BENCH_DIR/baseline.json"

    echo "=== Verifying benchmark results ==="

    if [ ! -f "$lmcache_result" ]; then
        echo "LMCache result file not found: $lmcache_result"
        return 1
    fi
    if [ ! -f "$baseline_result" ]; then
        echo "Baseline result file not found: $baseline_result"
        return 1
    fi

    # Extract values
    lmcache_total_input_tokens=$(extract_json_field "$lmcache_result" "total_input_tokens")
    lmcache_completed=$(extract_json_field "$lmcache_result" "completed")
    lmcache_throughput=$(extract_json_field "$lmcache_result" "total_token_throughput")

    baseline_total_input_tokens=$(extract_json_field "$baseline_result" "total_input_tokens")
    baseline_completed=$(extract_json_field "$baseline_result" "completed")
    baseline_throughput=$(extract_json_field "$baseline_result" "total_token_throughput")

    echo "=== LMCache Results ==="
    echo "  total_input_tokens: $lmcache_total_input_tokens"
    echo "  completed: $lmcache_completed"
    echo "  total_token_throughput: $lmcache_throughput"
    echo ""
    echo "=== Baseline Results ==="
    echo "  total_input_tokens: $baseline_total_input_tokens"
    echo "  completed: $baseline_completed"
    echo "  total_token_throughput: $baseline_throughput"
    echo ""

    local failed=0

    echo "=== Verification ==="

    # vLLM's random dataset decodes and re-encodes token sequences, which can
    # drift slightly from the requested length (see RandomDataset in
    # vllm/benchmarks/datasets.py). Allow 1% tolerance.
    local token_tolerance=$((EXPECTED_TOTAL_INPUT_TOKENS / 100))

    check_input_tokens() {
        local label="$1"
        local actual="$2"
        local diff=$((actual - EXPECTED_TOTAL_INPUT_TOKENS))
        local abs_diff=${diff#-}
        if [ "$abs_diff" -le "$token_tolerance" ] 2>/dev/null; then
            echo "$label total_input_tokens: $actual (expected: $EXPECTED_TOTAL_INPUT_TOKENS ±$token_tolerance) PASS"
        else
            echo "$label total_input_tokens: $actual (expected: $EXPECTED_TOTAL_INPUT_TOKENS ±$token_tolerance) FAIL"
            failed=1
        fi
    }

    check_input_tokens "LMCache" "$lmcache_total_input_tokens"
    check_input_tokens "Baseline" "$baseline_total_input_tokens"

    if [ "$lmcache_completed" -eq "$EXPECTED_COMPLETED" ] 2>/dev/null; then
        echo "LMCache completed: $lmcache_completed (expected: $EXPECTED_COMPLETED) PASS"
    else
        echo "LMCache completed: $lmcache_completed (expected: $EXPECTED_COMPLETED) FAIL"
        failed=1
    fi

    if [ "$baseline_completed" -eq "$EXPECTED_COMPLETED" ] 2>/dev/null; then
        echo "Baseline completed: $baseline_completed (expected: $EXPECTED_COMPLETED) PASS"
    else
        echo "Baseline completed: $baseline_completed (expected: $EXPECTED_COMPLETED) FAIL"
        failed=1
    fi

    # Throughput comparison
    throughput_check=$(python3 -c "
lmcache_tp = $lmcache_throughput
baseline_tp = $baseline_throughput
max_slowdown = $MAX_SLOWDOWN_PERCENT
min_acceptable = baseline_tp * (1 - max_slowdown / 100.0)
if baseline_tp > 0:
    slowdown_pct = ((baseline_tp - lmcache_tp) / baseline_tp) * 100
else:
    slowdown_pct = 0
if lmcache_tp >= min_acceptable:
    print(f'PASS|{slowdown_pct:.2f}')
else:
    print(f'FAIL|{slowdown_pct:.2f}')
")

    throughput_status=$(echo "$throughput_check" | cut -d'|' -f1)
    slowdown_pct=$(echo "$throughput_check" | cut -d'|' -f2)

    if [ "$throughput_status" = "PASS" ]; then
        echo "Throughput: LMCache is ${slowdown_pct}% slower (max allowed: ${MAX_SLOWDOWN_PERCENT}%) PASS"
    else
        echo "Throughput: LMCache is ${slowdown_pct}% slower (max allowed: ${MAX_SLOWDOWN_PERCENT}%) FAIL"
        failed=1
    fi

    # Sanity check: on a random (no-reuse) workload, LMCache should NOT be
    # significantly faster than baseline. If it is, the benchmark setup is
    # asymmetric and the results are unreliable as a regression test.
    local max_speedup_pct=10
    speedup_check=$(python3 -c "
lmcache_tp = $lmcache_throughput
baseline_tp = $baseline_throughput
if baseline_tp > 0:
    speedup_pct = ((lmcache_tp - baseline_tp) / baseline_tp) * 100
else:
    speedup_pct = 0
if speedup_pct > $max_speedup_pct:
    print(f'WARN|{speedup_pct:.2f}')
else:
    print(f'OK|{speedup_pct:.2f}')
")

    local speedup_status speedup_pct
    speedup_status=$(echo "$speedup_check" | cut -d'|' -f1)
    speedup_pct=$(echo "$speedup_check" | cut -d'|' -f2)

    if [ "$speedup_status" = "WARN" ]; then
        echo "WARNING: LMCache is ${speedup_pct}% faster than baseline on random workload (max expected: ${max_speedup_pct}%)"
        echo "This suggests a measurement asymmetry, not a real cache benefit."
        failed=1
    else
        echo "Speedup sanity check: LMCache is ${speedup_pct}% faster (max expected: ${max_speedup_pct}%) OK"
    fi

    echo ""
    return "$failed"
}

warmup_server() {
    local port="$1"
    local description="$2"
    local num_warmup="${3:-3}"

    echo "=== Warming up $description (port $port) ==="
    # Send a few chat completion requests to warm up the tokenizer,
    # chat template (Jinja2), and engine pipeline. Without this, the
    # first-ever batch of requests incurs ~25s of cold-start overhead
    # (BPE compilation, template compilation, etc.) which skews the
    # benchmark since lm-eval (Step 3) only warms the LMCache server.
    for i in $(seq 1 "$num_warmup"); do
        curl -s -X POST "http://localhost:${port}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"${MODEL}\",
                \"messages\": [{\"role\": \"user\", \"content\": \"Warmup request ${i}. The quick brown fox jumps over the lazy dog.\"}],
                \"max_tokens\": 1
            }" > /dev/null 2>&1
    done
    echo "$description warmup complete"
}

echo "Using random seed: $RANDOM_SEED"
echo ""

# Warm up both servers so tokenizer/template compilation does not
# skew the benchmark. See THROUGHPUT_FIX_PROPOSAL.md for details.
echo "============================================"
echo "=== Warming up both servers ==="
echo "============================================"
warmup_server "$VLLM_PORT" "vLLM with LMCache"
warmup_server "$VLLM_BASELINE_PORT" "vLLM baseline"
echo ""

# Baseline first
echo "============================================"
echo "=== Benchmark: Baseline vLLM (without LMCache) ==="
echo "============================================"
run_vllm_bench "$VLLM_BASELINE_PORT" "baseline.json" "Baseline vLLM" "$RANDOM_SEED"

# LMCache
echo "============================================"
echo "=== Benchmark: vLLM with LMCache ==="
echo "============================================"
run_vllm_bench "$VLLM_PORT" "lmcache.json" "vLLM with LMCache" "$RANDOM_SEED"

# Verify
echo "============================================"
echo "=== Verifying benchmark results ==="
echo "============================================"
if ! verify_results; then
    echo "Verification failed"
    exit 1
fi

echo "============================================"
echo "=== vLLM Bench test completed ==="
echo "============================================"
