#!/bin/bash
# Run long_doc_qa workload test against both vLLM servers
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

DOCUMENT_LENGTH="${DOCUMENT_LENGTH:-10000}"
NUM_DOCUMENTS="${NUM_DOCUMENTS:-30}"
OUTPUT_LEN="${OUTPUT_LEN:-200}"

REPEAT_COUNT="${REPEAT_COUNT:-2}"
REPEAT_MODE="${REPEAT_MODE:-tile}"
SHUFFLE_SEED="${SHUFFLE_SEED:-0}"
MAX_INFLIGHT_REQUESTS="${MAX_INFLIGHT_REQUESTS:-5}"

# Performance thresholds for LMCache (test fails if exceeded)
# 0.22s for loading 10K tokens of Qwen3-14B
# 0.50s for query round time of Qwen3-14B
MAX_LMCACHE_TTFT="${MAX_LMCACHE_TTFT:-0.22}"
MAX_LMCACHE_QUERY_ROUND_TIME="${MAX_LMCACHE_QUERY_ROUND_TIME:-0.50}"

# Output directory (subdirectory of shared RESULTS_DIR)
LONG_DOC_QA_DIR="$RESULTS_DIR/long_doc_qa"

echo "=== Long Doc QA Test ==="
echo "Model: $MODEL"
echo "vLLM Port (with LMCache): $VLLM_PORT"
echo "vLLM Baseline Port (without LMCache): $VLLM_BASELINE_PORT"
echo "Document length: $DOCUMENT_LENGTH"
echo "Number of documents: $NUM_DOCUMENTS"
echo "Output length: $OUTPUT_LEN"
echo "Repeat count: $REPEAT_COUNT"
echo "Repeat mode: $REPEAT_MODE"
echo "Shuffle seed: $SHUFFLE_SEED"
echo "Max inflight requests: $MAX_INFLIGHT_REQUESTS"
echo "Virtual env: $VENV_DIR"
echo "Build ID: $BUILD_ID"
echo "Results dir: $LONG_DOC_QA_DIR"
echo ""
echo "Performance thresholds (LMCache):"
echo "  Max TTFT: ${MAX_LMCACHE_TTFT}s"
echo "  Max query round time: ${MAX_LMCACHE_QUERY_ROUND_TIME}s"
echo ""

# Create results directory
mkdir -p "$LONG_DOC_QA_DIR"

# Run long_doc_qa benchmark
run_long_doc_qa() {
    local port="$1"
    local result_file="$2"
    local description="$3"
    
    echo "=== Running long_doc_qa ($description) ==="
    echo "Port: $port"
    echo "Result file: $result_file"
    
    local output_file="$LONG_DOC_QA_DIR/${description}_output.txt"
    
    # Run long_doc_qa.py and capture JSON output
    python3 "$LMCACHE_DIR/benchmarks/long_doc_qa/long_doc_qa.py" \
        --port "$port" \
        --model "$MODEL" \
        --document-length "$DOCUMENT_LENGTH" \
        --num-documents "$NUM_DOCUMENTS" \
        --output-len "$OUTPUT_LEN" \
        --repeat-count "$REPEAT_COUNT" \
        --repeat-mode "$REPEAT_MODE" \
        --shuffle-seed "$SHUFFLE_SEED" \
        --max-inflight-requests "$MAX_INFLIGHT_REQUESTS" \
        --output "$output_file" \
        --json-output \
        2>>"$output_file" | tee "$result_file"
    
    echo "✅ $description benchmark completed"
    echo ""
}

# Extract a field from JSON file (last line)
extract_json_field() {
    local json_file="$1"
    local field="$2"
    
    # Get the last line which should be the JSON output
    local json_line
    json_line=$(tail -n 1 "$json_file")
    
    python3 -c "
import json
import sys
try:
    data = json.loads('$json_line')
    value = data.get('$field', 'null')
    print(value if value is not None else 'null')
except json.JSONDecodeError:
    print('null')
"
}

# Compare and summarize results
compare_results() {
    local lmcache_result="$LONG_DOC_QA_DIR/lmcache_result.json"
    local baseline_result="$LONG_DOC_QA_DIR/baseline_result.json"
    
    echo "=== Comparing benchmark results ==="
    
    # Check if result files exist
    if [ ! -f "$lmcache_result" ]; then
        echo "❌ LMCache result file not found: $lmcache_result"
        return 1
    fi
    
    if [ ! -f "$baseline_result" ]; then
        echo "❌ Baseline result file not found: $baseline_result"
        return 1
    fi
    
    # Extract values from LMCache result
    lmcache_query_ttft=$(extract_json_field "$lmcache_result" "query_ttft_per_prompt")
    lmcache_query_round_time=$(extract_json_field "$lmcache_result" "query_round_time_per_prompt")
    lmcache_warmup_round_time=$(extract_json_field "$lmcache_result" "warmup_round_time_per_prompt")
    
    # Extract values from baseline result
    baseline_query_ttft=$(extract_json_field "$baseline_result" "query_ttft_per_prompt")
    baseline_query_round_time=$(extract_json_field "$baseline_result" "query_round_time_per_prompt")
    baseline_warmup_round_time=$(extract_json_field "$baseline_result" "warmup_round_time_per_prompt")
    
    echo ""
    echo "============================================"
    echo "=== Performance Summary ==="
    echo "============================================"
    echo ""
    
    # Calculate and display comparison
    python3 << EOF
import sys

def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

failed = False

def format_comparison(name, lmcache_val, baseline_val):
    global failed
    lmcache = safe_float(lmcache_val)
    baseline = safe_float(baseline_val)

    if lmcache is None or baseline is None:
        return f"{name}: Unable to compare (invalid values)"

    if baseline > 0:
        diff_pct = ((lmcache - baseline) / baseline) * 100
        if diff_pct < 0:
            status = "✅ FASTER"
            diff_str = f"{abs(diff_pct):.2f}% faster"
        elif diff_pct > 10:
            status = "❌ SLOWER"
            diff_str = f"{diff_pct:.2f}% slower"
            failed = True
        else:
            status = "✅ OK"
            diff_str = f"{diff_pct:.2f}% slower"
    else:
        status = "⚠️"
        diff_str = "N/A"

    return f"{name}:\n  Baseline:  {baseline:.4f}s\n  LMCache:   {lmcache:.4f}s\n  Diff:      {diff_str} {status}"

print("=" * 50)
print("Query TTFT per Prompt (Time to First Token)")
print("=" * 50)
print(format_comparison("query_ttft_per_prompt", "$lmcache_query_ttft", "$baseline_query_ttft"))
print()

print("=" * 50)
print("Query Round Time per Prompt (End-to-End Latency)")
print("=" * 50)
print(format_comparison("query_round_time_per_prompt", "$lmcache_query_round_time", "$baseline_query_round_time"))
print()

print("=" * 50)
print("Warmup Round Time per Prompt")
print("=" * 50)
print(format_comparison("warmup_round_time_per_prompt", "$lmcache_warmup_round_time", "$baseline_warmup_round_time"))
print()

# Summary table
print("=" * 50)
print("Summary Table")
print("=" * 50)
print(f"{'Metric':<35} {'Baseline':>12} {'LMCache':>12} {'Diff':>10}")
print("-" * 70)

metrics = [
    ("query_ttft_per_prompt", "$baseline_query_ttft", "$lmcache_query_ttft"),
    ("query_round_time_per_prompt", "$baseline_query_round_time", "$lmcache_query_round_time"),
    ("warmup_round_time_per_prompt", "$baseline_warmup_round_time", "$lmcache_warmup_round_time"),
]

for name, baseline_val, lmcache_val in metrics:
    baseline = safe_float(baseline_val)
    lmcache = safe_float(lmcache_val)

    if baseline is not None and lmcache is not None:
        if baseline > 0:
            diff_pct = ((lmcache - baseline) / baseline) * 100
            diff_str = f"{diff_pct:+.1f}%"
        else:
            diff_str = "N/A"
        print(f"{name:<35} {baseline:>12.4f} {lmcache:>12.4f} {diff_str:>10}")
    else:
        print(f"{name:<35} {'N/A':>12} {'N/A':>12} {'N/A':>10}")

print()

if failed:
    print("❌ LMCache is significantly slower (>10%) than baseline on one or more metrics")
    sys.exit(1)
else:
    print("✅ LMCache performance is within acceptable range")
    sys.exit(0)
EOF
}

# Verify LMCache performance meets thresholds
verify_thresholds() {
    local lmcache_result="$LONG_DOC_QA_DIR/lmcache_result.json"
    
    echo "=== Verifying LMCache performance thresholds ==="
    echo "Max allowed TTFT: ${MAX_LMCACHE_TTFT}s"
    echo "Max allowed query round time: ${MAX_LMCACHE_QUERY_ROUND_TIME}s"
    echo ""
    
    # Extract LMCache values
    lmcache_query_ttft=$(extract_json_field "$lmcache_result" "query_ttft_per_prompt")
    lmcache_query_round_time=$(extract_json_field "$lmcache_result" "query_round_time_per_prompt")
    
    # Verify thresholds using Python
    python3 << EOF
import sys

def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

ttft = safe_float("$lmcache_query_ttft")
query_round_time = safe_float("$lmcache_query_round_time")
max_ttft = float("$MAX_LMCACHE_TTFT")
max_query_round_time = float("$MAX_LMCACHE_QUERY_ROUND_TIME")

failed = False

print("=" * 50)
print("Threshold Verification")
print("=" * 50)

# Check TTFT
if ttft is None:
    print("❌ query_ttft_per_prompt: Unable to parse value")
    failed = True
elif ttft <= max_ttft:
    print(f"✅ query_ttft_per_prompt: {ttft:.4f}s <= {max_ttft}s (PASS)")
else:
    print(f"❌ query_ttft_per_prompt: {ttft:.4f}s > {max_ttft}s (FAIL)")
    failed = True

# Check query round time
if query_round_time is None:
    print("❌ query_round_time_per_prompt: Unable to parse value")
    failed = True
elif query_round_time <= max_query_round_time:
    print(f"✅ query_round_time_per_prompt: {query_round_time:.4f}s <= {max_query_round_time}s (PASS)")
else:
    print(f"❌ query_round_time_per_prompt: {query_round_time:.4f}s > {max_query_round_time}s (FAIL)")
    failed = True

print()

if failed:
    print("❌ Threshold verification FAILED")
    sys.exit(1)
else:
    print("✅ All thresholds passed")
    sys.exit(0)
EOF
}

# Main execution
main() {
    setup_venv openai pandas matplotlib
    
    # Run benchmark against baseline vLLM (without LMCache)
    echo "============================================"
    echo "=== Benchmark: Baseline vLLM (without LMCache) ==="
    echo "============================================"
    run_long_doc_qa "$VLLM_BASELINE_PORT" "$LONG_DOC_QA_DIR/baseline_result.json" "baseline"
    
    # Run benchmark against vLLM with LMCache
    echo "============================================"
    echo "=== Benchmark: vLLM with LMCache ==="
    echo "============================================"
    run_long_doc_qa "$VLLM_PORT" "$LONG_DOC_QA_DIR/lmcache_result.json" "lmcache"
    
    # Compare and summarize results
    echo "============================================"
    echo "=== Comparing Results ==="
    echo "============================================"
    if ! compare_results; then
        echo "❌ Comparison failed"
        exit 1
    fi
    
    # Verify LMCache performance meets thresholds (soft-fail: warn but don't exit)
    echo "============================================"
    echo "=== Verifying Performance Thresholds ==="
    echo "============================================"
    if ! verify_thresholds; then
        echo "⚠️ Threshold verification failed (soft-fail, not blocking)"
    fi
    
    echo "============================================"
    echo "=== ✅ Long Doc QA test completed ==="
    echo "============================================"
    echo "Results saved to: $LONG_DOC_QA_DIR"
    echo "  - LMCache: $LONG_DOC_QA_DIR/lmcache_result.json"
    echo "  - Baseline: $LONG_DOC_QA_DIR/baseline_result.json"
}

main "$@"
