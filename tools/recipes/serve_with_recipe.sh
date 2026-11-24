#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER="${SCRIPT_DIR}/recipe_json_to_vllm_config.py"
OUTPUT_DIR="$(pwd -P)"
CONFIG_PATH="${OUTPUT_DIR}/config.yml"
ENV_PATH="${OUTPUT_DIR}/env.sh"

args=("$@")
hardware=""

for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
        --hardware)
            if ((i + 1 < ${#args[@]})); then
                hardware="${args[$((i + 1))]}"
            fi
            ;;
        --hardware=*)
            hardware="${args[$i]#*=}"
            ;;
    esac
done

has_detect_hardware=false
for arg in "${args[@]}"; do
    if [[ "$arg" == "--detect-hardware" ]]; then
        has_detect_hardware=true
        break
    fi
done

if [[ "$has_detect_hardware" == false && \
      ( "${hardware,,}" == "xeon6" || "${hardware,,}" == "arc_pro_b70" ) ]]; then
    args+=(--detect-hardware)
fi

# Explicit opt-in for XPU TP/DP fallback; no behavior change for other hardware.
if [[ "${hardware,,}" == "arc_pro_b70" &&
      ( "${RECIPE_AUTO_FIT_XPU_PARALLELISM:-false}" == "true" ||
        "${RECIPE_AUTO_FIT_XPU_PARALLELISM:-false}" == "1" ) ]]; then
    args+=(--auto-fit-xpu-parallelism)
fi

# Opt-in protocol for validation harnesses; normal CLI use is unchanged.
if [[ -n "${RECIPE_VALIDATION_RESULT_OUT:-}" ]]; then
    args+=(--validation-result-out "$RECIPE_VALIDATION_RESULT_OUT")
fi

echo "==> Generating vLLM configuration"
python3 "${CONVERTER}" "${args[@]}"

echo
echo "==> Preparing recipe assets"
python3 "${SCRIPT_DIR}/recipe_assets.py" \
    --config "${CONFIG_PATH}" \
    --cache-dir "${RECIPE_ASSET_CACHE_DIR:-${OUTPUT_DIR}/.recipe-assets}"

echo
echo "==> Loading recipe environment"
# shellcheck disable=SC1090
source "${ENV_PATH}"

# Released vLLM images keep recipe-relative assets such as examples/*.jinja
# under /vllm-workspace. Launch from there when available while keeping the
# generated config and environment files in the caller's output directory.
if [[ -d /vllm-workspace/examples ]]; then
    cd /vllm-workspace
fi

echo
echo "==> Starting vLLM"
exec vllm serve --config "${CONFIG_PATH}"
