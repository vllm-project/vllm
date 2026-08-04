#!/usr/bin/env bash
# CacheBlend <-> LMCache compatibility check.
#
 
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

cd "${REPO_ROOT}"
source .buildkite/k3_tests/common_scripts/helpers.sh

# Fail fast if the GPUs are occupied by stale host processes.
check_gpu_health 80

: "${CB_PLUGIN_DIR:?CB_PLUGIN_DIR not set — source k3_harness/setup-blend-env.sh first}"

MODEL="${CB_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MAX_LEN="${CB_MAX_MODEL_LEN:-16384}"
LOG_DIR="${REPO_ROOT}/cb_compat_logs"      # workspace-relative so artifact_paths can collect it
mkdir -p "${LOG_DIR}"

# Resolve the venv that owns vllm/lmcache. In the CI pod this is the image's
# /opt/venv (on PATH via setup-env.sh); CB_VENV overrides for local runs.
if [ -n "${CB_VENV:-}" ]; then
    VENV="${CB_VENV}"
elif [ -n "${VIRTUAL_ENV:-}" ]; then
    VENV="${VIRTUAL_ENV}"
elif command -v vllm >/dev/null 2>&1; then
    VENV="$(cd "$(dirname "$(command -v vllm)")/.." && pwd)"
else
    VENV="/opt/venv"
fi

echo "+++ :jigsaw: vllm_compat_check  model=${MODEL}  venv=${VENV}  plugin=${CB_PLUGIN_DIR}"
python "${CB_PLUGIN_DIR}/.buildkite/harness/vllm_compat_check.py" \
    --model "${MODEL}" \
    --gpu-sync 0 \
    --skip-correctness \
    --venv "${VENV}" \
    --max-model-len "${MAX_LEN}" \
    --log-dir "${LOG_DIR}"
