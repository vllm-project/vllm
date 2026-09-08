#!/usr/bin/env bash
set -euo pipefail

# NixlPushConnector (push-mode PD) integration tests. Kept separate from the
# pull-mode sweep in ../nixl_integration so push coverage can grow on its own;
# the per-config runner is shared.
SCRIPT="v1/kv_connector/nixl_integration/run_accuracy_test.sh"
IMPORT_CANARY="v1/kv_connector/nixl_integration/test_nixl_imports.py"

echo "=== Running NIXL import canary ==="
python3 -m pytest -s -x "${IMPORT_CANARY}"

# PP=2 prefill -> TP=2 decode: each PP stage writes its layer slice into the
# decoder's matching region range (non-MLA and MLA contiguous-slice paths).
configs=(
  "KV_CONNECTOR=NixlPushConnector GPU_MEMORY_UTILIZATION=0.6 PREFILLER_PP_SIZE=2 PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=2" # non-MLA, Qwen3-0.6B
  "KV_CONNECTOR=NixlPushConnector GPU_MEMORY_UTILIZATION=0.8 PREFILLER_PP_SIZE=2 PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=2 MODEL_NAMES=deepseek-ai/deepseek-vl2-tiny" # MLA
)

run_tests() {
  local label=$1
  local extra_args=$2

  echo "=== Running push tests (${label}) ==="
  for cfg in "${configs[@]}"; do
    local -a cfg_parts extra_args_parts
    read -r -a cfg_parts <<< "$cfg"
    read -r -a extra_args_parts <<< "$extra_args"

    echo "-> Running with ${cfg} ${extra_args:+and ${extra_args}}"
    # Use 'env' to safely set variables without eval; keep argv splitting
    # safe and SC2086-clean via arrays.
    if ! env "${cfg_parts[@]}" bash "${SCRIPT}" "${extra_args_parts[@]}"; then
      echo "❌ Test failed for config: ${cfg} ${extra_args:+(${extra_args})}"
      exit 1
    fi
  done
  echo "✅ All ${label} push tests passed!"
}

run_tests "default backend" ""
