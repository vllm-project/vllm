#!/usr/bin/env bash
set -euo pipefail

# Skip-Softmax LongBench-only sweep
# Same server orchestration as skip_softmax_run_all.sh, but runs ONLY
# the LongBench-E accuracy eval for each (model × threshold × kv-cache-dtype)
# combination. No perf benchmarks, no GSM8K, no MMLU Pro.
#
# Use this to re-measure long-context accuracy after a code change without
# re-running the full perf+accuracy sweep.
#
# Requires (in addition to lm_eval itself):
#   uv pip install jieba fuzzywuzzy rouge python-Levenshtein
# See tests/skip_softmax_test_plan.md for details.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure HF_TOKEN is exported so vllm serve (and huggingface_hub) can see it
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
fi

# ── tunables ──────────────────────────────────────────────────────
# Keep in sync with skip_softmax_run_all.sh so we're evaluating the
# same set of server configs.
MODELS=(
  "/models/Qwen3/Qwen3-30B-A3B-Instruct-2507"
)

# Threshold combinations to sweep, formatted as "<prefill>:<decode>".
# Use "none" for either side to mean "don't pass that flag at all".
THRESHOLD_PAIRS=(
  "none:none"
  "10000:500"
)

# KV cache data types to test
KV_CACHE_DTYPES=("auto")

PORT=8000
MAX_MODEL_LEN=140000 # must cover the largest ISL + OSL
HEALTH_TIMEOUT=1800  # seconds to wait for server health (large models take a while)
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"
# ──────────────────────────────────────────────────────────────────

wait_for_server() {
  local port="$1"
  local timeout="$2"
  echo ">>> Waiting for server on port ${port} (timeout ${timeout}s) …"
  for i in $(seq 1 "$timeout"); do
    if curl -sf "http://127.0.0.1:${port}/health" > /dev/null 2>&1; then
      echo ">>> Server ready after ~${i}s"
      return 0
    fi
    if (( i % 10 == 0 )); then
      echo ">>>   … still waiting (${i}/${timeout}s)"
    fi
    sleep 1
  done
  echo ">>> ERROR: Server did not become healthy within ${timeout}s" >&2
  return 1
}

for MODEL in "${MODELS[@]}"; do
  for KV_DTYPE in "${KV_CACHE_DTYPES[@]}"; do
    for PAIR in "${THRESHOLD_PAIRS[@]}"; do

      THRESH_PREFILL="${PAIR%%:*}"
      THRESH_DECODE="${PAIR##*:}"

      echo "============================================================"
      echo "MODEL         : $MODEL"
      echo "KV_DTYPE      : $KV_DTYPE"
      echo "THRESH prefill: $THRESH_PREFILL"
      echo "THRESH decode : $THRESH_DECODE"
      echo "TASK          : LongBench-E (only)"
      echo "============================================================"

      # ── build the extra flags ──
      EXTRA_FLAGS=()
      if [[ "$THRESH_PREFILL" != "none" ]]; then
        EXTRA_FLAGS+=(
          --attention-config.skip_softmax_threshold_scale_factor_prefill \
          "$THRESH_PREFILL"
        )
      fi
      if [[ "$THRESH_DECODE" != "none" ]]; then
        EXTRA_FLAGS+=(
          --attention-config.skip_softmax_threshold_scale_factor_decode \
          "$THRESH_DECODE"
        )
      fi

      # ── launch server in background, redirect output to log ──
      MODEL_TAG="${MODEL//\//_}"
      THRESH_TAG="thresh-pf${THRESH_PREFILL}-dc${THRESH_DECODE}"
      SERVER_LOG="${LOG_DIR}/${MODEL_TAG}_${THRESH_TAG}_kvdtype-${KV_DTYPE}_longbench_server.log"
      echo ">>> Server log: $SERVER_LOG"

      VLLM_ENGINE_READY_TIMEOUT_S=7200 vllm serve "$MODEL" \
        --kv-cache-dtype "$KV_DTYPE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --port "$PORT" \
        --trust-remote-code \
        --no-enable-log-requests \
        "${EXTRA_FLAGS[@]}" \
        > "$SERVER_LOG" 2>&1 &
      SERVER_PID=$!

      # ── wait for the server to be ready (fail if it never is) ──
      if ! wait_for_server "$PORT" "$HEALTH_TIMEOUT"; then
        echo ">>> Killing unhealthy server (PID $SERVER_PID) and skipping this config."
        echo ">>> Check $SERVER_LOG for details."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        continue
      fi

      # ── run LongBench-E only ──
      echo ">>> Running LongBench-E accuracy …"
      SKIP_SOFTMAX_ACCURACY_TASKS="longbench_e" \
        bash "${SCRIPT_DIR}/skip_softmax_accuracy.sh" \
          "$MODEL" "$THRESH_PREFILL" "$THRESH_DECODE" "$KV_DTYPE" "$PORT"

      # ── tear down ──
      kill "$SERVER_PID" 2>/dev/null || true
      wait "$SERVER_PID" 2>/dev/null || true
      echo ">>> Server stopped."
      echo ""

    done
  done
done
