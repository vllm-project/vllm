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
  "GPU_MEMORY_UTILIZATION=0.8 MODEL_NAMES=google/gemma-3-4b-it VLLM_SERVE_EXTRA_ARGS=--max-model-len,8192" # SW model
)
dp_ep_configs=(
"DP_EP=1 GPU_MEMORY_UTILIZATION=0.8 PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=2 MODEL_NAMES=deepseek-ai/deepseek-vl2-tiny" # MLA+P-TP1, D-DPEP=2 (TP=1)
"DP_EP=1 GPU_MEMORY_UTILIZATION=0.8 PREFILLER_TP_SIZE=2 DECODER_TP_SIZE=2 MODEL_NAMES=deepseek-ai/deepseek-vl2-tiny" # MLA+P-TP2, D-DPEP=2 (TP=1)
)
hybrid_ssm_configs=(
  "ENABLE_HMA_FLAG=1 GPU_MEMORY_UTILIZATION=0.8 MODEL_NAMES=ibm-granite/granite-4.0-h-tiny VLLM_SERVE_EXTRA_ARGS=--max-model-len,8192,--trust-remote-code"
  # TODO: (NickLucche) Address async scheduling issue with TP>1 separately as this may impact other models.
  "ENABLE_HMA_FLAG=1 PREFILLER_TP_SIZE=2 DECODER_TP_SIZE=2 GPU_MEMORY_UTILIZATION=0.8 MODEL_NAMES=ibm-granite/granite-4.0-h-tiny VLLM_SERVE_EXTRA_ARGS=--max-model-len,8192,--trust-remote-code,--no-async-scheduling"
)
sw_attn_configs=(
  "ENABLE_HMA_FLAG=1 GPU_MEMORY_UTILIZATION=0.8 MODEL_NAMES=google/gemma-3-4b-it PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=2 VLLM_SERVE_EXTRA_ARGS=--max-model-len,8192"
  "ENABLE_HMA_FLAG=1 GPU_MEMORY_UTILIZATION=0.8 MODEL_NAMES=google/gemma-3-4b-it PREFILLER_TP_SIZE=2 DECODER_TP_SIZE=1 VLLM_SERVE_EXTRA_ARGS=--max-model-len,8192"
)

# Select config array based on DP_EP env var
if [[ -n "${DP_EP:-}" ]]; then
  configs=("${dp_ep_configs[@]}")
  echo "DP_EP is set, using dp_ep_configs"
elif [[ -n "${HYBRID_SSM:-}" ]]; then
  configs=("${hybrid_ssm_configs[@]}")
  echo "HYBRID_SSM is set, using hybrid_ssm_configs."
elif [[ -n "${SW_ATTN:-}" ]]; then
  configs=("${sw_attn_configs[@]}")
  echo "SW_ATTN is set, using sw_attn_configs."
else
  configs=("${tp_configs[@]}")
fi

if [[ -n "${ENABLE_HMA_FLAG:-}" ]]; then
  # Append ENABLE_HMA_FLAG=1 to each config in the selected array
  echo "ENABLE_HMA_FLAG is set, appending ENABLE_HMA_FLAG=1 to each config"
  for i in "${!configs[@]}"; do
    configs[$i]="ENABLE_HMA_FLAG=1 ${configs[$i]}"
  done
fi

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
if [[ -n "${ROCM_ATTN:-}" ]]; then
  echo "ROCM_ATTN is set, running with --attention-backend ROCM_ATTN"
  label="ROCM_ATTN backend"
  cmdline_args=" --attention-backend ROCM_ATTN "
elif [[ -n "${FLASHINFER:-}" ]]; then
  echo "FLASHINFER is set, running with --attention-backend FLASHINFER"
  label="FLASHINFER backend"
  cmdline_args=" --attention-backend FLASHINFER "
else
  echo "running with default attention backend"
fi

# Check if cross-layers is enabled (non-empty)
if [[ -n "${CROSS_LAYERS_BLOCKS:-}" ]]; then
  echo "CROSS_LAYERS_BLOCKS is set, running with --enable-cross-layers"
  label+=" - CROSS_LAYERS_BLOCKS enabled"
  cmdline_args+=" --enable-cross-layers "
fi

# Run tests
run_tests "${label}" "${cmdline_args}"
