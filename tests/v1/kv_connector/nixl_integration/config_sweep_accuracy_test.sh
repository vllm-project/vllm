#!/usr/bin/env bash
set -euo pipefail

# Utility to run integration tests sequentially with varying TP configurations.
SCRIPT="v1/kv_connector/nixl_integration/run_accuracy_test.sh"

# Define test configurations
tp_configs=(
  "GPU_MEMORY_UTILIZATION=0.6 PREFILLER_TP_SIZE=2 DECODER_TP_SIZE=2"
  "GPU_MEMORY_UTILIZATION=0.6 PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=2"
  "GPU_MEMORY_UTILIZATION=0.6 PREFILLER_TP_SIZE=2 DECODER_TP_SIZE=1"
  "GPU_MEMORY_UTILIZATION=0.8 MODEL_NAMES=deepseek-ai/deepseek-vl2-tiny" # MLA case
  "GPU_MEMORY_UTILIZATION=0.8 PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=2 MODEL_NAMES=deepseek-ai/deepseek-vl2-tiny"
  "GPU_MEMORY_UTILIZATION=0.8 PREFILLER_TP_SIZE=2 DECODER_TP_SIZE=1 MODEL_NAMES=deepseek-ai/deepseek-vl2-tiny"
)
cross_layers_configs=(
  "CROSS_LAYERS_BLOCKS=True GPU_MEMORY_UTILIZATION=0.6 PREFILLER_TP_SIZE=2 DECODER_TP_SIZE=2"
  "CROSS_LAYERS_BLOCKS=True GPU_MEMORY_UTILIZATION=0.6 PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=2"
  "CROSS_LAYERS_BLOCKS=True GPU_MEMORY_UTILIZATION=0.6 PREFILLER_TP_SIZE=2 DECODER_TP_SIZE=1"
  "CROSS_LAYERS_BLOCKS=True GPU_MEMORY_UTILIZATION=0.8 MODEL_NAMES=deepseek-ai/deepseek-vl2-tiny" # MLA case
  "CROSS_LAYERS_BLOCKS=True GPU_MEMORY_UTILIZATION=0.8 PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=2 MODEL_NAMES=deepseek-ai/deepseek-vl2-tiny"
  "CROSS_LAYERS_BLOCKS=True GPU_MEMORY_UTILIZATION=0.8 PREFILLER_TP_SIZE=2 DECODER_TP_SIZE=1 MODEL_NAMES=deepseek-ai/deepseek-vl2-tiny"
)
dp_ep_configs=(
"DP_EP=1 GPU_MEMORY_UTILIZATION=0.8 PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=2 MODEL_NAMES=deepseek-ai/deepseek-vl2-tiny" # MLA+P-TP1, D-DPEP=2 (TP=1)
"DP_EP=1 GPU_MEMORY_UTILIZATION=0.8 PREFILLER_TP_SIZE=2 DECODER_TP_SIZE=2 MODEL_NAMES=deepseek-ai/deepseek-vl2-tiny" # MLA+P-TP2, D-DPEP=2 (TP=1)
)

# Select config array based on DP_EP env var
if [[ -n "${DP_EP:-}" ]]; then
  configs=("${dp_ep_configs[@]}")
  echo "DP_EP is set, using dp_ep_configs"
elif [[ -n "${CROSS_LAYERS_BLOCKS:-}" ]]; then
  configs=("${cross_layers_configs[@]}")
  echo "CROSS_LAYERS_BLOCKS is set, using cross_layers_configs"
else
  configs=("${tp_configs[@]}")
fi

run_tests() {
  local label=$1
  local extra_args=$2

  echo "=== Running tests (${label}) ==="
  for cfg in "${configs[@]}"; do
    echo "-> Running with ${cfg} ${extra_args:+and ${extra_args}}"
    # Use 'env' to safely set variables without eval
    if ! env ${cfg} bash "${SCRIPT}" ${extra_args}; then
      echo "❌ Test failed for config: ${cfg} ${extra_args:+(${extra_args})}"
      exit 1
    fi
  done
  echo "✅ All ${label} tests passed!"
}

# Run tests
if [[ -n "${ROCM_ATTN:-}" ]]; then
  echo "ROCM_ATTN is set, running with --attention-backend ROCM_ATTN"
  run_tests "ROCM_ATTN backend" "--attention-backend ROCM_ATTN"
else
  run_tests "default backend" ""
fi

# Check if FLASHINFER is set (non-empty)
if [[ -n "${FLASHINFER:-}" ]]; then
  echo "FLASHINFER is set, rerunning with --attention-backend FLASHINFER"
  run_tests "FLASHINFER backend" "--attention-backend FLASHINFER"
else
  echo "FLASHINFER not set, skipping FLASHINFER runs."
fi
