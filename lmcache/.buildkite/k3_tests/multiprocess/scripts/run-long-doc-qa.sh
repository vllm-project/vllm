#!/usr/bin/env bash
# Run long_doc_qa workload test against both vLLM servers.
# Compares performance between LMCache-enabled and baseline vLLM.
# Adapted from the old Docker-based run-long-doc-qa.sh.
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

# Configuration
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASELINE_PORT="${VLLM_BASELINE_PORT:-9000}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
BUILD_ID="${BUILD_ID:-local_$$}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
LMCACHE_DIR="${LMCACHE_DIR:-$REPO_ROOT}"

DOCUMENT_LENGTH="${DOCUMENT_LENGTH:-10000}"
NUM_DOCUMENTS="${NUM_DOCUMENTS:-30}"
OUTPUT_LEN="${OUTPUT_LEN:-200}"
REPEAT_COUNT="${REPEAT_COUNT:-2}"
REPEAT_MODE="${REPEAT_MODE:-tile}"
SHUFFLE_SEED="${SHUFFLE_SEED:-0}"
MAX_INFLIGHT_REQUESTS="${MAX_INFLIGHT_REQUESTS:-5}"

# Relative performance thresholds (compared against baseline run in same job)
# Negative values mean LMCache must be *faster* than baseline by at least that %.
# Recent CI runs show ~77-84% TTFT improvement and ~27-40% round-time improvement,
# so requiring 60% and 15% respectively leaves comfortable headroom.
MAX_TTFT_SLOWDOWN_PCT="${MAX_TTFT_SLOWDOWN_PCT:--60}"
MAX_ROUND_TIME_SLOWDOWN_PCT="${MAX_ROUND_TIME_SLOWDOWN_PCT:--15}"

# Output directory
LONG_DOC_QA_DIR="$RESULTS_DIR/long_doc_qa"

echo "=== Long Doc QA Test ==="
echo "Model: $MODEL"
echo "vLLM Port (with LMCache): $VLLM_PORT"
echo "vLLM Baseline Port (without LMCache): $VLLM_BASELINE_PORT"
echo "Document length: $DOCUMENT_LENGTH"
echo "Number of documents: $NUM_DOCUMENTS"
echo "Output length: $OUTPUT_LEN"
echo "Results dir: $LONG_DOC_QA_DIR"
echo ""
echo "Performance thresholds (relative to baseline, negative = must be faster):"
echo "  Max TTFT slowdown: ${MAX_TTFT_SLOWDOWN_PCT}% (LMCache must be >= $(echo "$MAX_TTFT_SLOWDOWN_PCT" | tr -d '-')% faster)"
echo "  Max round time slowdown: ${MAX_ROUND_TIME_SLOWDOWN_PCT}% (LMCache must be >= $(echo "$MAX_ROUND_TIME_SLOWDOWN_PCT" | tr -d '-')% faster)"
echo ""

mkdir -p "$LONG_DOC_QA_DIR"

run_long_doc_qa() {
    local port="$1"
    local result_file="$2"
    local description="$3"

    echo "=== Running long_doc_qa ($description) ==="
    local output_file="$LONG_DOC_QA_DIR/${description}_output.txt"

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

    echo "$description benchmark completed"
    echo ""
}

extract_json_field() {
    local json_file="$1"
    local field="$2"
    local json_line
    json_line=$(tail -n 1 "$json_file")
    python3 -c "
import json, sys
try:
    data = json.loads('''$json_line''')
    value = data.get('$field', 'null')
    print(value if value is not None else 'null')
except json.JSONDecodeError:
    print('null')
"
}

compare_results() {
    local lmcache_result="$LONG_DOC_QA_DIR/lmcache_result.json"
    local baseline_result="$LONG_DOC_QA_DIR/baseline_result.json"

    echo "=== Comparing benchmark results ==="

    if [ ! -f "$lmcache_result" ] || [ ! -f "$baseline_result" ]; then
        echo "Result files not found"
        return 1
    fi

    lmcache_query_ttft=$(extract_json_field "$lmcache_result" "query_ttft_per_prompt")
    lmcache_query_round_time=$(extract_json_field "$lmcache_result" "query_round_time_per_prompt")
    lmcache_warmup_round_time=$(extract_json_field "$lmcache_result" "warmup_round_time_per_prompt")

    baseline_query_ttft=$(extract_json_field "$baseline_result" "query_ttft_per_prompt")
    baseline_query_round_time=$(extract_json_field "$baseline_result" "query_round_time_per_prompt")
    baseline_warmup_round_time=$(extract_json_field "$baseline_result" "warmup_round_time_per_prompt")

    echo ""
    echo "============================================"
    echo "=== Performance Summary ==="
    echo "============================================"

    python3 << EOF
import sys

def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def format_comparison(name, lmcache_val, baseline_val):
    lmcache = safe_float(lmcache_val)
    baseline = safe_float(baseline_val)
    if lmcache is None or baseline is None:
        return f"{name}: Unable to compare (invalid values)"
    if baseline > 0:
        diff_pct = ((lmcache - baseline) / baseline) * 100
        diff_str = f"{abs(diff_pct):.2f}% {'faster' if diff_pct < 0 else 'slower'}"
    else:
        diff_str = "N/A"
    return f"{name}:\n  Baseline:  {baseline:.4f}s\n  LMCache:   {lmcache:.4f}s\n  Diff:      {diff_str}"

print(format_comparison("query_ttft_per_prompt", "$lmcache_query_ttft", "$baseline_query_ttft"))
print()
print(format_comparison("query_round_time_per_prompt", "$lmcache_query_round_time", "$baseline_query_round_time"))
print()
print(format_comparison("warmup_round_time_per_prompt", "$lmcache_warmup_round_time", "$baseline_warmup_round_time"))
print()

# Summary table
print(f"{'Metric':<35} {'Baseline':>12} {'LMCache':>12} {'Diff':>10}")
print("-" * 70)
metrics = [
    ("query_ttft_per_prompt", "$baseline_query_ttft", "$lmcache_query_ttft"),
    ("query_round_time_per_prompt", "$baseline_query_round_time", "$lmcache_query_round_time"),
    ("warmup_round_time_per_prompt", "$baseline_warmup_round_time", "$lmcache_warmup_round_time"),
]
for name, bval, lval in metrics:
    b = safe_float(bval)
    l = safe_float(lval)
    if b is not None and l is not None:
        diff_pct = ((l - b) / b) * 100 if b > 0 else 0
        print(f"{name:<35} {b:>12.4f} {l:>12.4f} {diff_pct:>+9.1f}%")
    else:
        print(f"{name:<35} {'N/A':>12} {'N/A':>12} {'N/A':>10}")
print()
EOF

    return 0
}

verify_thresholds() {
    local lmcache_result="$LONG_DOC_QA_DIR/lmcache_result.json"
    local baseline_result="$LONG_DOC_QA_DIR/baseline_result.json"

    echo "=== Verifying LMCache performance vs baseline ==="
    echo "Max allowed TTFT slowdown: ${MAX_TTFT_SLOWDOWN_PCT}%"
    echo "Max allowed query round time slowdown: ${MAX_ROUND_TIME_SLOWDOWN_PCT}%"
    echo ""

    lmcache_query_ttft=$(extract_json_field "$lmcache_result" "query_ttft_per_prompt")
    lmcache_query_round_time=$(extract_json_field "$lmcache_result" "query_round_time_per_prompt")
    baseline_query_ttft=$(extract_json_field "$baseline_result" "query_ttft_per_prompt")
    baseline_query_round_time=$(extract_json_field "$baseline_result" "query_round_time_per_prompt")

    python3 << EOF
import sys

def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def check_metric(name, lmcache_val, baseline_val, max_slowdown_pct):
    lmc = safe_float(lmcache_val)
    base = safe_float(baseline_val)
    if lmc is None or base is None or base <= 0:
        print(f"{name}: unable to compare (lmcache={lmcache_val}, baseline={baseline_val}) -- FAIL")
        return False
    pct = ((lmc - base) / base) * 100
    label = f"{abs(pct):.1f}% faster" if pct < 0 else f"{pct:.1f}% slower"
    if max_slowdown_pct < 0:
        threshold_label = f"need >= {abs(max_slowdown_pct):.0f}% faster"
    else:
        threshold_label = f"max {max_slowdown_pct}% slower"
    if pct <= max_slowdown_pct:
        print(f"{name}: {lmc:.4f}s vs baseline {base:.4f}s ({label}, {threshold_label}) -- PASS")
        return True
    else:
        print(f"{name}: {lmc:.4f}s vs baseline {base:.4f}s ({label}, {threshold_label}) -- FAIL")
        return False

failed = False
if not check_metric("query_ttft_per_prompt",
                     "$lmcache_query_ttft", "$baseline_query_ttft",
                     float("$MAX_TTFT_SLOWDOWN_PCT")):
    failed = True
if not check_metric("query_round_time_per_prompt",
                     "$lmcache_query_round_time", "$baseline_query_round_time",
                     float("$MAX_ROUND_TIME_SLOWDOWN_PCT")):
    failed = True

if failed:
    print("\nThreshold verification FAILED")
    sys.exit(1)
else:
    print("\nAll thresholds passed")
    sys.exit(0)
EOF
}

# Run benchmark against baseline
echo "============================================"
echo "=== Benchmark: Baseline vLLM (without LMCache) ==="
echo "============================================"
run_long_doc_qa "$VLLM_BASELINE_PORT" "$LONG_DOC_QA_DIR/baseline_result.json" "baseline"

# Run benchmark against vLLM with LMCache
echo "============================================"
echo "=== Benchmark: vLLM with LMCache ==="
echo "============================================"
run_long_doc_qa "$VLLM_PORT" "$LONG_DOC_QA_DIR/lmcache_result.json" "lmcache"

# Compare
echo "============================================"
echo "=== Comparing Results ==="
echo "============================================"
if ! compare_results; then
    echo "Comparison failed"
    exit 1
fi

# Verify thresholds
echo "============================================"
echo "=== Verifying Performance Thresholds ==="
echo "============================================"
if ! verify_thresholds; then
    echo "Threshold verification failed"
    exit 1
fi

echo "============================================"
echo "=== Long Doc QA test completed ==="
echo "============================================"
