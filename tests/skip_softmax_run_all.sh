#!/usr/bin/env bash
set -euo pipefail

# Skip-Softmax full test sweep
# Starts a server per (model × threshold × kv-cache-dtype) combination,
# runs perf + accuracy benchmarks, then tears the server down.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure HF_TOKEN is exported so vllm serve (and huggingface_hub) can see it
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
fi

# ── tunables ──────────────────────────────────────────────────────
MODELS=(
  "/models/Qwen3/Qwen3-30B-A3B-Instruct-2507"
)

# Threshold combinations to sweep, formatted as "<prefill>:<decode>".
# Use "none" for either side to mean "don't pass that flag at all".
#
# The non-baseline pairs below are the scale factors produced by
# `tools/calibrate_skip_softmax.py` for
# /models/Qwen3/Qwen3-30B-A3B-Instruct-2507 (samples=24, max_seqlen=16384,
# chunk_size=4096) — see calibration/qwen3_30b_a3b_instruct_2507.json.
# Fitted exponential model per phase (scale = a * exp(b * target)):
#   prefill: a=26.7158,  b=7.0113  (R^2=0.91)
#   decode:  a=1.837e-5, b=20.0    (R^2=0.67, b hit optimiser clamp)
# The decode fit's R^2 is mediocre, so at 30%/50% targets the value is a
# rough order-of-magnitude estimate (see skip_softmax_tuning_guide.md §4.4).
THRESHOLD_PAIRS=(
  "none:none"
  # target sparsity 30%  (prefill / decode)
  "218.91:0.00741"
  # target sparsity 50%
  "889.72:0.40463"
  # target sparsity 70%
  "3616.16:22.092"
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
    # Print a progress dot every 10 seconds so the user knows it's alive
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
      SERVER_LOG="${LOG_DIR}/${MODEL_TAG}_${THRESH_TAG}_kvdtype-${KV_DTYPE}_server.log"
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

      # ── run perf benchmarks ──
      echo ">>> Running perf benchmarks …"
      bash "${SCRIPT_DIR}/skip_softmax_perf.sh" \
        "$MODEL" "$THRESH_PREFILL" "$THRESH_DECODE" "$KV_DTYPE" "$PORT"

      # ── run accuracy benchmarks ──
      echo ">>> Running accuracy benchmarks …"
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
