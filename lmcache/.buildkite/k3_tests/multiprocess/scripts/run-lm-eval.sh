#!/usr/bin/env bash
# Run lm_eval workload test against vLLM server.
# Sends the same requests twice to test LMCache caching behavior.
# Adapted from the old Docker-based run-lm-eval.sh -- no venv setup needed
# (setup-env.sh + extras already installed by run.sh).
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

# Configuration
VLLM_PORT="${VLLM_PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
NUM_CONCURRENT="${NUM_CONCURRENT:-50}"
LIMIT="${LIMIT:-300}"
BUILD_ID="${BUILD_ID:-local_$$}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
LM_EVAL_VERIFY_MODE="${LM_EVAL_VERIFY_MODE:-samples}"
SCORE_TOLERANCE="${SCORE_TOLERANCE:-0.05}"
SCORE_MIN="${SCORE_MIN:-0.80}"
VLLM_LOG="${VLLM_LOG:-/tmp/build_${BUILD_ID}_vllm.log}"

case "$LM_EVAL_VERIFY_MODE" in
    samples) LM_EVAL_DIR="$RESULTS_DIR/lm_eval" ;;
    preemption) LM_EVAL_DIR="$RESULTS_DIR/lm_eval_preemption" ;;
    *)
        echo "Unknown LM_EVAL_VERIFY_MODE: $LM_EVAL_VERIFY_MODE (valid: samples, preemption)"
        exit 1
        ;;
esac
FIRST_RUN_DIR="$LM_EVAL_DIR/first_run"
SECOND_RUN_DIR="$LM_EVAL_DIR/second_run"

echo "=== LM-Eval Workload Test ($LM_EVAL_VERIFY_MODE) ==="
echo "Model: $MODEL"
echo "vLLM Port: $VLLM_PORT"
echo "Concurrent requests: $NUM_CONCURRENT"
echo "Limit: $LIMIT"
echo "Results dir: $LM_EVAL_DIR"
echo ""

mkdir -p "$FIRST_RUN_DIR" "$SECOND_RUN_DIR"

# Run one gsm8k lm_eval pass, writing results/samples to output_dir.
run_lm_eval() {
    local run_name="$1"
    local output_dir="$2"

    echo "=== Running lm_eval ($run_name) ==="
    lm_eval --model local-completions --tasks gsm8k \
        --model_args "model=${MODEL},base_url=http://127.0.0.1:${VLLM_PORT}/v1/completions,num_concurrent=${NUM_CONCURRENT},max_retries=3,tokenized_requests=False" \
        --limit "$LIMIT" \
        --seed 0 \
        -s --output_path "$output_dir" \
        --gen_kwargs '{"temperature": 0.0}'

    echo "$run_name completed"
    echo ""
}

verify_samples_match() {
    local first_dir="$1"
    local second_dir="$2"

    echo "=== Verifying samples files match ==="

    first_samples=$(find "$first_dir" -name "samples_gsm8k_*.jsonl" -type f 2>/dev/null | head -1)
    second_samples=$(find "$second_dir" -name "samples_gsm8k_*.jsonl" -type f 2>/dev/null | head -1)

    if [ -z "$first_samples" ]; then
        echo "Could not find samples_gsm8k_*.jsonl in first run directory: $first_dir"
        find "$first_dir" -type f -name "*.jsonl" || true
        return 1
    fi

    if [ -z "$second_samples" ]; then
        echo "Could not find samples_gsm8k_*.jsonl in second run directory: $second_dir"
        find "$second_dir" -type f -name "*.jsonl" || true
        return 1
    fi

    echo "First run samples: $first_samples"
    echo "Second run samples: $second_samples"

    first_sorted=$(mktemp)
    second_sorted=$(mktemp)

    sort "$first_samples" > "$first_sorted"
    sort "$second_samples" > "$second_sorted"

    if diff -q "$first_sorted" "$second_sorted" > /dev/null 2>&1; then
        echo "Samples files are identical!"
        rm -f "$first_sorted" "$second_sorted"
        return 0
    else
        echo "Samples files differ!"
        echo ""
        echo "=== Diff (first 50 lines) ==="
        diff "$first_sorted" "$second_sorted" | head -50 || true
        rm -f "$first_sorted" "$second_sorted"
        return 1
    fi
}

count_preemptions() {
    [ -f "$VLLM_LOG" ] || { echo 0; return; }
    local count
    count=$(grep -c "<preempted>" "$VLLM_LOG" 2>/dev/null || true)
    echo "${count:-0}"
}

verify_preemption() {
    # Check score drift/floor and that each lm_eval run observed preemptions.
    python3 - "$1" "$2" "$SCORE_TOLERANCE" "$SCORE_MIN" "$3" "$4" "$5" <<'PYEOF'
import glob, json, os, sys

first_dir, second_dir, tolerance, score_min, before, after_first, after_second = sys.argv[1:8]
tolerance, score_min = float(tolerance), float(score_min)
before, after_first, after_second = int(before), int(after_first), int(after_second)
def score(results_dir):
    """Return gsm8k exact_match from newest results_*.json or exit if missing."""
    files = glob.glob(os.path.join(results_dir, "**", "results_*.json"), recursive=True)
    if not files:
        raise SystemExit(f"No results_*.json under {results_dir}")
    with open(max(files, key=os.path.getmtime)) as f:
        metrics = json.load(f)["results"]["gsm8k"]
    if "exact_match,strict-match" in metrics:
        return float(metrics["exact_match,strict-match"])
    for key in metrics:
        if key.startswith("exact_match,") and "stderr" not in key:
            return float(metrics[key])
    raise SystemExit(f"No exact_match metric in {sorted(metrics)}")
first, second = score(first_dir), score(second_dir)
drift = abs(first - second)
print(f"First run gsm8k exact_match: {first:.4f}")
print(f"Second run gsm8k exact_match: {second:.4f}")
print(f"vLLM preemptions logged: before={before}, after_first={after_first}, after_second={after_second}")
failures = []
if drift > tolerance:
    failures.append(f"score drift {drift:.4f} > tolerance {tolerance}")
if first < score_min or second < score_min:
    failures.append(f"scores below minimum {score_min}: first={first:.4f}, second={second:.4f}")
if after_first <= before:
    failures.append(f"no preemptions observed during first_run (before={before}, after_first={after_first})")
if after_second <= after_first:
    failures.append(f"no preemptions observed during second_run (after_first={after_first}, after_second={after_second})")
if failures:
    raise SystemExit("FAILED:\n  - " + "\n  - ".join(failures))
print(
    "Preemption verification passed; "
    f"first_run observed {after_first - before}, "
    f"second_run observed {after_second - after_first} preemptions"
)
PYEOF
}

# First run -- populates cache
echo "============================================"
echo "=== First lm_eval run (cache population) ==="
echo "============================================"
[ "$LM_EVAL_VERIFY_MODE" = "preemption" ] && preemptions_before=$(count_preemptions)
run_lm_eval "first_run" "$FIRST_RUN_DIR"
[ "$LM_EVAL_VERIFY_MODE" = "preemption" ] && preemptions_after_first=$(count_preemptions)

# Second run -- should use cached results
echo "============================================"
echo "=== Second lm_eval run (cache hit) ==="
echo "============================================"
run_lm_eval "second_run" "$SECOND_RUN_DIR"
[ "$LM_EVAL_VERIFY_MODE" = "preemption" ] && preemptions_after_second=$(count_preemptions)

# Verify consistency
echo "============================================"
echo "=== Verifying output consistency ==="
echo "============================================"
if [ "$LM_EVAL_VERIFY_MODE" = "preemption" ]; then
    verify_preemption "$FIRST_RUN_DIR" "$SECOND_RUN_DIR" \
        "$preemptions_before" "$preemptions_after_first" "$preemptions_after_second"
elif ! verify_samples_match "$FIRST_RUN_DIR" "$SECOND_RUN_DIR"; then
    echo "Verification failed: samples files do not match"
    exit 1
fi

echo "============================================"
echo "=== LM-Eval workload test completed ==="
echo "============================================"
