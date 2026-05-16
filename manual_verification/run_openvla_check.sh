#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

workspace_root="${WORKSPACE_ROOT:-/workspace}"
run_dir="${OPENVLA_RUN_DIR:-${workspace_root}/openvla_check}"
log_dir="${workspace_root}/logs"
ref_venv="${workspace_root}/openvla_hf_ref_venv"
script_dir="${repo_root}/manual_verification"
log="${log_dir}/openvla_check_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${log_dir}" "${run_dir}"
export OPENVLA_RUN_DIR="${run_dir}"

{
  echo "OpenVLA check"
  echo "repo_root=${repo_root}"
  echo "run_dir=${run_dir}"
  echo "commit=$(git rev-parse --short HEAD)"
  echo "branch=$(git branch --show-current)"
  echo "start=$(date -Is)"
  echo

  if [ -f "${workspace_root}/.hf_env" ]; then
    . "${workspace_root}/.hf_env"
  fi
  export HF_HUB_ENABLE_HF_TRANSFER=1
  export VLLM_ALLOW_INSECURE_SERIALIZATION=1

  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
  fi

  echo "== system =="
  nvidia-smi || true
  df -h "${workspace_root}" || true
  free -h || true
  echo

  echo "== setup HF reference env =="
  if [ ! -x "${ref_venv}/bin/python" ]; then
    uv venv --python 3.12 "${ref_venv}"
  fi
  if "${ref_venv}/bin/python" - <<PY >/dev/null 2>&1
import accelerate
import huggingface_hub
import pyarrow
import timm
import tokenizers
import torch
import torchvision
import transformers
PY
  then
    echo "HF reference env already has required packages."
  else
    uv pip install --python "${ref_venv}/bin/python" \
      accelerate \
      hf_transfer \
      huggingface_hub \
      pillow \
      pyarrow \
      safetensors \
      timm==0.9.10 \
      tokenizers==0.19.1 \
      torch==2.6.0 \
      torchvision==0.21.0 \
      transformers==4.40.1
  fi
  uv pip show --python "${ref_venv}/bin/python" timm torch transformers
  echo

  echo "== setup vLLM env =="
  if [ ! -x ".venv/bin/python" ]; then
    uv venv --python 3.12 .venv
  fi
  if .venv/bin/python - <<PY >/dev/null 2>&1
import huggingface_hub
import pyarrow
import timm
import vllm
PY
  then
    echo "vLLM env already has required packages."
  else
    echo "vLLM editable install may take 10+ minutes on a fresh machine."
    VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
    uv pip install --python .venv/bin/python \
      hf_transfer \
      huggingface_hub \
      pillow \
      pyarrow \
      timm==0.9.10
  fi
  uv pip show --python .venv/bin/python timm torch transformers vllm
  echo

  echo "== sample LIBERO cases =="
  "${ref_venv}/bin/python" "${script_dir}/sample_libero_cases.py"
  echo

  echo "== run HF OpenVLA =="
  "${ref_venv}/bin/python" "${script_dir}/run_hf_openvla.py"
  echo

  echo "== run vLLM OpenVLA and write result =="
  .venv/bin/python "${script_dir}/run_vllm_openvla.py"
  echo

  echo "result=${run_dir}/result.json"
  echo "end=$(date -Is)"
} 2>&1 | tee "${log}"

echo "LOG=${log}"
