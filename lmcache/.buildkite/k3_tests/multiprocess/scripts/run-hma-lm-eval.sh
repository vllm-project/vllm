#!/usr/bin/env bash
# HMA (hybrid memory allocator) correctness test using a real hybrid model.
#
# Models (selected by run-single-test.sh):
#   - google/gemma-4-31B-it: sliding-window + full-attention hybrid whose full
#     layers have a larger head_dim, so vLLM gives the KV cache groups
#     different block sizes -- exercising per-group HMA store/retrieve.
#   - Qwen/Qwen3.5-0.8B: Mamba/GDN + full-attention hybrid, exercising the
#     registration-time cache re-views (kv_cache_group_edits.py).
#
# Flow (single GPU, no baseline server):
#   1. vLLM run: lm_eval (gsm8k) against vLLM+LMCache, populating LMCache.
#   2. Reset vLLM's *local* prefix cache (APC) only, leaving LMCache intact, via
#      the dev-mode endpoint POST /reset_prefix_cache (reset_external defaults to
#      false, so the LMCache-managed cache is preserved).
#   3. LMCache retrieve run: re-run lm_eval; vLLM's APC misses, so the prefix KV
#      is served by LMCache.
#   4. Assert the two runs' gsm8k scores match -- a broken LMCache would skew the
#      retrieved KV and make them diverge.
#   5. Assert LMCache actually served retrieves in the retrieve run (non-vacuous).
#
# The reset endpoint requires VLLM_SERVER_DEV_MODE=1 (set by launch-processes.sh).
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

# Configuration
VLLM_PORT="${VLLM_PORT:-8000}"
MODEL="${MODEL:-google/gemma-4-31B-it}"
NUM_CONCURRENT="${NUM_CONCURRENT:-50}"
# 31B has a large per-token KV footprint; cap the sample count so the working
# set fits the CPU pool (a too-large set thrashes and the retrieve run misses).
LIMIT="${LIMIT:-100}"
# Max abs difference allowed between the two runs' gsm8k scores; 0 requires an
# exact match. For non-bit-exact backends, raise LIMIT to shrink run-to-run
# drift (~1/sqrt(LIMIT)) rather than loosening this.
SCORE_TOLERANCE="${SCORE_TOLERANCE:-0}"
# Seconds to let async LMCache stores drain before the retrieve run.
STORE_DRAIN_SECONDS="${STORE_DRAIN_SECONDS:-20}"
BUILD_ID="${BUILD_ID:-local_$$}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
# LMCache MP server log, scanned to confirm the retrieve run hit LMCache.
LMCACHE_LOG="${LMCACHE_LOG:-/tmp/build_${BUILD_ID}_lmcache.log}"

HMA_DIR="$RESULTS_DIR/hma_lm_eval"
VLLM_RUN_DIR="$HMA_DIR/vllm_run"
RETRIEVE_RUN_DIR="$HMA_DIR/retrieve_run"

echo "=== HMA lm_eval correctness test ==="
echo "Model: $MODEL"
echo "vLLM (LMCache) port: $VLLM_PORT"
echo "Concurrent requests: $NUM_CONCURRENT"
echo "Limit: $LIMIT"
echo "Score tolerance: $SCORE_TOLERANCE"
echo "Results dir: $HMA_DIR"
echo ""

mkdir -p "$VLLM_RUN_DIR" "$RETRIEVE_RUN_DIR"

# Run one lm_eval gsm8k pass against a vLLM OpenAI-compatible server.
#
# Globals (read):
#   MODEL          - HuggingFace model id, echoed to lm_eval's model_args.
#   NUM_CONCURRENT - number of in-flight requests lm_eval issues.
#   LIMIT          - number of gsm8k samples to evaluate.
# Arguments:
#   $1 port       - TCP port of the vLLM /v1/completions endpoint to evaluate.
#   $2 output_dir - directory lm_eval writes results_*.json / samples_*.jsonl to.
#   $3 run_name   - human-readable label used only in progress log lines.
# Outputs:
#   Writes lm_eval result and per-sample files under output_dir; prints progress
#   to stdout.
# Returns:
#   lm_eval's exit status (non-zero if the evaluation run fails). Propagated to
#   the caller via ``set -e``.
run_lm_eval() {
    local port="$1"
    local output_dir="$2"
    local run_name="$3"

    echo "=== Running lm_eval ($run_name) on port $port ==="
    lm_eval --model local-completions --tasks gsm8k \
        --model_args "model=${MODEL},base_url=http://127.0.0.1:${port}/v1/completions,num_concurrent=${NUM_CONCURRENT},max_retries=3,tokenized_requests=False" \
        --limit "$LIMIT" \
        --seed 0 \
        -s --output_path "$output_dir" \
        --gen_kwargs '{"temperature": 0.0}'
    echo "$run_name completed"
    echo ""
}

# Reset a vLLM server's local prefix cache (APC) while preserving LMCache.
#
# POSTs to the dev-mode /reset_prefix_cache endpoint without reset_external
# (which defaults to false), so only vLLM's GPU-side automatic prefix cache is
# cleared and the LMCache-managed cache is left intact.
# Arguments:
#   $1 port - TCP port of the vLLM server whose local APC should be reset.
# Outputs:
#   Progress / failure detail to stdout.
# Returns:
#   0 if the server acknowledged with HTTP 200; 1 otherwise (e.g. the endpoint
#   is absent because VLLM_SERVER_DEV_MODE was not set when launching vLLM).
reset_vllm_prefix_cache() {
    local port="$1"
    echo "=== Resetting vLLM local prefix cache on port $port (LMCache preserved) ==="
    # reset_external defaults to false -> only vLLM's APC is cleared.
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
        "http://127.0.0.1:${port}/reset_prefix_cache")
    if [ "$code" != "200" ]; then
        echo "Failed to reset prefix cache (HTTP $code). Is VLLM_SERVER_DEV_MODE=1?"
        return 1
    fi
    echo "vLLM prefix cache reset."
    echo ""
}

# Count completed LMCache retrieves recorded in the server log so far.
#
# Used to prove run 2 was actually served by LMCache (the delta around run 2
# must be > 0), so the correctness comparison cannot pass vacuously by silently
# recomputing.
# Globals (read):
#   LMCACHE_LOG - path to the LMCache MP server log file.
# Arguments:
#   none.
# Outputs:
#   The integer count of "Retrieved" log lines to stdout (0 if the log file does
#   not exist yet).
count_retrieves() {
    # NB: ``grep -c`` prints 0 *and* exits 1 on no match, so guard the file
    # existence and use ``|| true`` (not ``|| echo 0``) to avoid emitting "0\n0".
    [ -f "$LMCACHE_LOG" ] || { echo 0; return; }
    grep -c "Retrieved" "$LMCACHE_LOG" 2>/dev/null || true
}

# ── 1. vLLM run: compute from scratch, populating LMCache ───
run_lm_eval "$VLLM_PORT" "$VLLM_RUN_DIR" "vLLM run"

# Let async stores drain to the LMCache server before invalidating the APC.
echo "Waiting ${STORE_DRAIN_SECONDS}s for LMCache stores to drain..."
sleep "$STORE_DRAIN_SECONDS"

retrieves_before=$(count_retrieves)

# ── 2. Invalidate vLLM's local prefix cache (keep LMCache) ──
reset_vllm_prefix_cache "$VLLM_PORT"

# ── 3. Retrieve run: vLLM APC misses -> LMCache serves the KV ─
run_lm_eval "$VLLM_PORT" "$RETRIEVE_RUN_DIR" "LMCache retrieve run"

retrieves_after=$(count_retrieves)

# ── 4. Compare scores and verify LMCache was actually used ──
echo "============================================"
echo "=== Verifying HMA store/retrieve correctness ==="
echo "============================================"
echo "LMCache retrieves logged: before=${retrieves_before}, after=${retrieves_after}"

python3 - "$VLLM_RUN_DIR" "$RETRIEVE_RUN_DIR" \
    "$SCORE_TOLERANCE" "$retrieves_before" "$retrieves_after" <<'PYEOF'
import glob
import json
import os
import sys

vllm_run_dir, retrieve_run_dir, tol_s, before_s, after_s = sys.argv[1:6]
tol = float(tol_s)
retrieves_before = int(before_s)
retrieves_after = int(after_s)


def gsm8k_score_and_stderr(results_dir: str) -> tuple[float, float]:
    """Return the gsm8k (exact_match, stderr) from an lm_eval results directory.

    Prefers the strict-match variant; falls back to any non-stderr
    ``exact_match`` metric key (paired with its ``exact_match_stderr`` twin).

    Args:
        results_dir: Directory passed to ``lm_eval --output_path``. Searched
            recursively for the newest ``results_*.json`` (lm_eval nests it
            under a per-model subdirectory and stamps the filename with a
            timestamp).

    Returns:
        ``(score, stderr)``: the gsm8k ``exact_match`` accuracy in
        ``[0.0, 1.0]`` and its reported sampling stderr (0.0 if absent).

    Raises:
        SystemExit: If no ``results_*.json`` exists under ``results_dir`` or the
            newest one contains no ``exact_match`` metric for the gsm8k task.
    """
    files = glob.glob(os.path.join(results_dir, "**", "results_*.json"), recursive=True)
    if not files:
        raise SystemExit(f"No results_*.json under {results_dir}")
    latest = max(files, key=os.path.getmtime)
    with open(latest) as f:
        data = json.load(f)
    metrics = data["results"]["gsm8k"]
    preferred = "exact_match,strict-match"
    if preferred in metrics:
        stderr = float(metrics.get("exact_match_stderr,strict-match", 0.0))
        return float(metrics[preferred]), stderr
    for key, value in metrics.items():
        if key.startswith("exact_match,") and "stderr" not in key:
            variant = key.split(",", 1)[1]
            stderr = float(metrics.get(f"exact_match_stderr,{variant}", 0.0))
            return float(value), stderr
    raise SystemExit(f"No exact_match metric in {latest}: {sorted(metrics)}")


s_vllm, e_vllm = gsm8k_score_and_stderr(vllm_run_dir)
s_retrieve, e_retrieve = gsm8k_score_and_stderr(retrieve_run_dir)

print(f"  vLLM run             gsm8k exact_match = {s_vllm:.4f} +/- {e_vllm:.4f}")
print(f"  LMCache retrieve run gsm8k exact_match = {s_retrieve:.4f} +/- {e_retrieve:.4f}")
print(f"  tolerance = {tol}")

failures = []
# The two runs must match -- a broken LMCache would skew the retrieved KV.
if abs(s_vllm - s_retrieve) > tol:
    failures.append(
        f"score drift between runs: |{s_vllm:.4f} - {s_retrieve:.4f}| = "
        f"{abs(s_vllm - s_retrieve):.4f} > {tol}"
    )
# Non-vacuous: the retrieve run must have been served by LMCache, not recompute.
if retrieves_after <= retrieves_before:
    failures.append(
        "LMCache served no retrieves during the retrieve run "
        f"(before={retrieves_before}, after={retrieves_after})"
    )

if failures:
    print("\nFAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)

print(
    f"\nPASS: vLLM and LMCache-retrieve gsm8k scores match (tol={tol}); "
    f"LMCache served {retrieves_after - retrieves_before} retrieves."
)
PYEOF

echo ""
echo "============================================"
echo "=== HMA lm_eval correctness test passed ==="
echo "============================================"
