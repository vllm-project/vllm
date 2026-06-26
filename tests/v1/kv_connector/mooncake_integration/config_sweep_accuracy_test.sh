#!/usr/bin/env bash
set -euo pipefail

# Utility to run Mooncake P/D integration tests sequentially with varying TP
# configurations.
SCRIPT="v1/kv_connector/mooncake_integration/run_accuracy_test.sh"

# Symmetric TP configurations (default model: Qwen/Qwen3-0.6B).
tp_configs=(
  "GPU_MEMORY_UTILIZATION=0.6 PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=1"
  "GPU_MEMORY_UTILIZATION=0.6 PREFILLER_TP_SIZE=2 DECODER_TP_SIZE=2"
)

configs=("${tp_configs[@]}")

run_tests() {
  local label=$1
  local extra_args=$2

  echo "=== Running tests (${label}) ==="
  for cfg in "${configs[@]}"; do
    local -a cfg_parts extra_args_parts
    read -r -a cfg_parts <<< "$cfg"
    read -r -a extra_args_parts <<< "$extra_args"

    echo "-> Running with ${cfg} ${extra_args:+and ${extra_args}}"
    # Use 'env' to safely set variables without eval
    # keep argv splitting safe and SC2086-clean via arrays.
    if ! env "${cfg_parts[@]}" bash "${SCRIPT}" "${extra_args_parts[@]}"; then
      echo "❌ Test failed for config: ${cfg} ${extra_args:+(${extra_args})}"
      exit 1
    fi
  done
  echo "✅ All ${label} tests passed!"
}

# Set backend
label="default backend"
cmdline_args=""
if [[ -n "${ATTENTION_BACKEND:-}" ]]; then
  echo "ATTENTION_BACKEND is set, running with --attention-backend ${ATTENTION_BACKEND}"
  label="${ATTENTION_BACKEND} backend"
  cmdline_args=" --attention-backend ${ATTENTION_BACKEND} "
else
  echo "running with default attention backend"
fi

# Run tests
run_tests "${label}" "${cmdline_args}"
