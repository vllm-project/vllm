#!/usr/bin/env bash
set -euo pipefail

# Sweep wrapper for spec decode acceptance tests, following the same pattern
# as config_sweep_accuracy_test.sh. Runs spec_decode_acceptance_test.sh once
# per configuration.

SCRIPT="v1/kv_connector/nixl_integration/spec_decode_acceptance_test.sh"

# EAGLE3: Llama-3.1-8B-Instruct with EAGLE3 speculator.
eagle3_config="SD_METHOD=eagle3 MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct SD_MODEL=RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3 NUM_SPEC_TOKENS=3"

# MTP: Qwen3.5-0.8B-Base with hybrid SSM flags.
mtp_config="SD_METHOD=mtp MODEL_NAME=Qwen/Qwen3.5-0.8B-Base SD_MODEL=Qwen/Qwen3.5-0.8B-Base NUM_SPEC_TOKENS=1 BLOCK_SIZE=32 MAX_MODEL_LEN=4096 VLLM_SSM_CONV_STATE_LAYOUT=DS KV_BUFFER_DEVICES=cuda"

configs=(
  "$eagle3_config"
  "$mtp_config"
)

for cfg in "${configs[@]}"; do
  local_cfg_parts=()
  read -r -a local_cfg_parts <<< "$cfg"
  echo "-> Running with: ${cfg}"
  if ! env "${local_cfg_parts[@]}" bash "${SCRIPT}"; then
    echo "❌ Test failed for config: ${cfg}"
    exit 1
  fi
done

echo "✅ All spec decode acceptance tests passed!"
