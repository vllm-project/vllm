#!/usr/bin/env bash
set -euo pipefail

# Utility to run integration tests sequentially with varying TP configurations.
SCRIPT="v1/kv_connector/nixl_integration/run_accuracy_test.sh"

# Define test configurations
configs=(
  "GPU_MEMORY_UTILIZATION=0.6 PREFILLER_TP_SIZE=2 DECODER_TP_SIZE=2"
  "GPU_MEMORY_UTILIZATION=0.6 PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=2"
  "GPU_MEMORY_UTILIZATION=0.8 MODEL_NAMES=deepseek-ai/deepseek-vl2-tiny" # MLA case
  "GPU_MEMORY_UTILIZATION=0.8 PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=2 MODEL_NAMES=deepseek-ai/deepseek-vl2-tiny"
  "DP_EP=1 GPU_MEMORY_UTILIZATION=0.8 PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=2 MODEL_NAMES=deepseek-ai/deepseek-vl2-tiny" # MLA+P-TP1, D-DPEP=2 (TP=1) 
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

# Run tests
run_tests "default backend" ""

# Check if FLASHINFER is set (non-empty)
if [[ -n "${FLASHINFER:-}" ]]; then
  echo "FLASHINFER is set, rerunning with VLLM_ATTENTION_BACKEND=FLASHINFER"
  run_tests "FLASHINFER backend" "VLLM_ATTENTION_BACKEND=FLASHINFER"
else
  echo "FLASHINFER not set, skipping FLASHINFER runs."
fi
