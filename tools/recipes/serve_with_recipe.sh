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

if [[ "${hardware,,}" == "xeon6" ]]; then
    args+=(--detect-hardware)
fi

echo "==> Generating vLLM configuration"
python3 "${CONVERTER}" "${args[@]}"

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
