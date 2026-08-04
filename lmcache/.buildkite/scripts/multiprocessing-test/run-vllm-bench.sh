#!/bin/bash
# Run vllm bench serve test against both vLLM servers
# Compares performance between LMCache-enabled and baseline vLLM

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common.sh"

# Configuration
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASELINE_PORT="${VLLM_BASELINE_PORT:-9000}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-10000}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-1}"

# Expected values
EXPECTED_TOTAL_INPUT_TOKENS=$((NUM_PROMPTS * RANDOM_INPUT_LEN))
EXPECTED_COMPLETED=$NUM_PROMPTS
MAX_SLOWDOWN_PERCENT=5

# Generate a random seed once for reproducibility across both benchmarks
RANDOM_SEED="${RANDOM_SEED:-$(date +%s)}"

# Output directory (subdirectory of shared RESULTS_DIR)
VLLM_BENCH_DIR="$RESULTS_DIR/vllm_bench"

echo "=== vLLM Bench Serve Test ==="
echo "Model: $MODEL"
echo "vLLM Port (with LMCache): $VLLM_PORT"
echo "vLLM Baseline Port (without LMCache): $VLLM_BASELINE_PORT"
echo "Number of prompts: $NUM_PROMPTS"
echo "Random input length: $RANDOM_INPUT_LEN"
echo "Random output length: $RANDOM_OUTPUT_LEN"
echo "Virtual env: $VENV_DIR"
echo "Build ID: $BUILD_ID"
echo "Results dir: $VLLM_BENCH_DIR"
echo ""

# Create results directory
mkdir -p "$VLLM_BENCH_DIR"

# Run vllm bench serve
run_vllm_bench() {
    local port="$1"
    local result_filename="$2"
    local description="$3"
    local seed="$4"
    
    echo "=== Running vllm bench serve ($description) ==="
    echo "Port: $port"
    echo "Seed: $seed"
    echo "Result file: $VLLM_BENCH_DIR/$result_filename"
    
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
    
    echo "✅ $description benchmark completed"
    echo ""
}

# Extract a numeric field from JSON file
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

# Verify benchmark results
verify_results() {
    local lmcache_result="$VLLM_BENCH_DIR/lmcache.json"
    local baseline_result="$VLLM_BENCH_DIR/baseline.json"
    
    echo "=== Verifying benchmark results ==="
    
    # Check if result files exist
    if [ ! -f "$lmcache_result" ]; then
        echo "❌ LMCache result file not found: $lmcache_result"
        return 1
    fi
    
    if [ ! -f "$baseline_result" ]; then
        echo "❌ Baseline result file not found: $baseline_result"
        return 1
    fi
    
    echo "LMCache result: $lmcache_result"
    echo "Baseline result: $baseline_result"
    echo ""
    
    # Extract values from LMCache result
    lmcache_total_input_tokens=$(extract_json_field "$lmcache_result" "total_input_tokens")
    lmcache_completed=$(extract_json_field "$lmcache_result" "completed")
    lmcache_throughput=$(extract_json_field "$lmcache_result" "total_token_throughput")
    
    # Extract values from baseline result
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
    
    # Verification
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
            echo "✅ $label total_input_tokens: $actual (expected: $EXPECTED_TOTAL_INPUT_TOKENS ±$token_tolerance)"
        else
            echo "❌ $label total_input_tokens: $actual (expected: $EXPECTED_TOTAL_INPUT_TOKENS ±$token_tolerance)"
            failed=1
        fi
    }

    check_input_tokens "LMCache" "$lmcache_total_input_tokens"
    check_input_tokens "Baseline" "$baseline_total_input_tokens"
    
    # Check completed for LMCache
    if [ "$lmcache_completed" -eq "$EXPECTED_COMPLETED" ] 2>/dev/null; then
        echo "✅ LMCache completed: $lmcache_completed (expected: $EXPECTED_COMPLETED)"
    else
        echo "❌ LMCache completed: $lmcache_completed (expected: $EXPECTED_COMPLETED)"
        failed=1
    fi
    
    # Check completed for baseline
    if [ "$baseline_completed" -eq "$EXPECTED_COMPLETED" ] 2>/dev/null; then
        echo "✅ Baseline completed: $baseline_completed (expected: $EXPECTED_COMPLETED)"
    else
        echo "❌ Baseline completed: $baseline_completed (expected: $EXPECTED_COMPLETED)"
        failed=1
    fi
    
    # Check throughput comparison (LMCache should not be more than 10% slower)
    throughput_check=$(python3 -c "
lmcache_tp = $lmcache_throughput
baseline_tp = $baseline_throughput
max_slowdown = $MAX_SLOWDOWN_PERCENT

# Calculate the minimum acceptable throughput (90% of baseline)
min_acceptable = baseline_tp * (1 - max_slowdown / 100.0)

# Calculate actual slowdown percentage
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
        echo "✅ Throughput comparison: LMCache is ${slowdown_pct}% slower (max allowed: ${MAX_SLOWDOWN_PERCENT}%)"
    else
        echo "❌ Throughput comparison: LMCache is ${slowdown_pct}% slower (max allowed: ${MAX_SLOWDOWN_PERCENT}%)"
        failed=1
    fi
    
    echo ""
    
    if [ "$failed" -eq 1 ]; then
        return 1
    fi
    
    return 0
}

# Main execution
main() {
    setup_venv vllm openai
    
    echo "Using random seed: $RANDOM_SEED"
    echo ""
    
    # Run benchmark against baseline vLLM (without LMCache)
    echo "============================================"
    echo "=== Benchmark: Baseline vLLM (without LMCache) ==="
    echo "============================================"
    run_vllm_bench "$VLLM_BASELINE_PORT" "baseline.json" "Baseline vLLM" "$RANDOM_SEED"
    
    # Run benchmark against vLLM with LMCache
    echo "============================================"
    echo "=== Benchmark: vLLM with LMCache ==="
    echo "============================================"
    run_vllm_bench "$VLLM_PORT" "lmcache.json" "vLLM with LMCache" "$RANDOM_SEED"
    
    # Verify results
    echo "============================================"
    echo "=== Verifying benchmark results ==="
    echo "============================================"
    if ! verify_results; then
        echo "❌ Verification failed"
        exit 1
    fi
    
    echo "============================================"
    echo "=== ✅ vLLM Bench test completed ==="
    echo "============================================"
    echo "Results saved to: $VLLM_BENCH_DIR"
    echo "  - LMCache: $VLLM_BENCH_DIR/lmcache.json"
    echo "  - Baseline: $VLLM_BENCH_DIR/baseline.json"
}

main "$@"
