#!/usr/bin/env bash
set -euo pipefail

# Utility to run integration tests sequentially with varying TP configurations.
# If FLASHINFER is set, reruns all tests with VLLM_ATTENTION_BACKEND=FLASHINFER.

SCRIPT="tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh"

# Define test configurations
configs=(
  "PREFILLER_TP_SIZE=2 DECODER_TP_SIZE=2"
  "PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=2"
  "PREFILLER_TP_SIZE=2 DECODER_TP_SIZE=1"
  "GPU_MEMORY_UTILIZATION=0.6 MODEL_NAMES=deepseek-ai/DeepSeek-V2-Lite-Chat" # MLA case
  # TP greater than num heads
)

run_tests() {
  local label=$1
  local extra_env=$2

  echo "=== Running tests (${label}) ==="
  for cfg in "${configs[@]}"; do
    echo "-> Running with ${cfg} ${extra_env:+and ${extra_env}}"
    # Use 'env' to safely set variables without eval
    if ! env ${extra_env} ${cfg} bash "${SCRIPT}"; then
      echo "❌ Test failed for config: ${cfg} ${extra_env:+(${extra_env})}"
      exit 1
    fi
  done
  echo "✅ All ${label} tests passed!"
}

# Run base tests
run_tests "default backend" ""

# Check if FLASHINFER is set (non-empty)
if [[ -n "${FLASHINFER:-}" ]]; then
  echo "FLASHINFER is set, rerunning with VLLM_ATTENTION_BACKEND=FLASHINFER"
  run_tests "FLASHINFER backend" "VLLM_ATTENTION_BACKEND=FLASHINFER"
else
  echo "FLASHINFER not set, skipping FLASHINFER runs."
fi
