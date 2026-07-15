#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${repo_root}"

env_file="${RWKV7_ALBATROSS_ENV_FILE:-.env}"
if [[ -f "${env_file}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${env_file}"
  set +a
fi

if [[ "${1:-}" == "--server" ]]; then
  export RWKV7_RUN_SERVER_ALIGNMENT=1
  shift
else
  export RWKV7_RUN_SERVER_ALIGNMENT=0
fi

export ALBATROSS_ROOT="${ALBATROSS_ROOT:-${HOME}/Projects/MachineLearning/albatross}"
export ALBATROSS_IMPL="${ALBATROSS_IMPL:-faster3a_2607}"
export VLLM_RWKV7_WKV_MODE="${VLLM_RWKV7_WKV_MODE:-fp32io16}"
export RWKV7_ALBATROSS_MAX_MODEL_LEN="${RWKV7_ALBATROSS_MAX_MODEL_LEN:-1024}"
export RWKV7_ALBATROSS_GPU_MEMORY_UTILIZATION="${RWKV7_ALBATROSS_GPU_MEMORY_UTILIZATION:-0.70}"
export RWKV7_ALBATROSS_ENABLE_FLASHINFER_AUTOTUNE="${RWKV7_ALBATROSS_ENABLE_FLASHINFER_AUTOTUNE:-0}"
export RWKV7_ALBATROSS_EXECUTION_MODES="${RWKV7_ALBATROSS_EXECUTION_MODES:-eager,cudagraph}"
export RWKV7_ALBATROSS_TENSOR_PARALLEL_SIZE="${RWKV7_ALBATROSS_TENSOR_PARALLEL_SIZE:-1}"
export RWKV7_ALBATROSS_PIPELINE_PARALLEL_SIZE="${RWKV7_ALBATROSS_PIPELINE_PARALLEL_SIZE:-1}"
export ALBATROSS_PTH="${ALBATROSS_PTH:-${VLLM_RWKV7_MODEL:-}}"
albatross_impl_dir="${ALBATROSS_ROOT}/${ALBATROSS_IMPL}"

missing=()
for name in ALBATROSS_PTH VLLM_RWKV7_MODEL; do
  if [[ -z "${!name:-}" ]]; then
    missing+=("${name}")
  fi
done

if (( ${#missing[@]} > 0 )); then
  printf 'Missing required RWKV7 Albatross test environment variables:\n' >&2
  printf '  %s\n' "${missing[@]}" >&2
  printf '\nCreate .env or set RWKV7_ALBATROSS_ENV_FILE to a file with:\n' >&2
  printf '  VLLM_RWKV7_MODEL=/path/to/rwkv7-g1g-1.5b-20260526-ctx8192.pth\n' >&2
  printf '  # Optional: ALBATROSS_PTH=/path/to/same-rwkv7.pth\n' >&2
  printf '  # Optional: RWKV7_ALBATROSS_TENSOR_PARALLEL_SIZE=2\n' >&2
  printf '  # Optional: RWKV7_ALBATROSS_PIPELINE_PARALLEL_SIZE=2\n' >&2
  printf '\nUse --server to include the OpenAI server alignment test.\n' >&2
  exit 2
fi

missing_paths=()
for name in ALBATROSS_ROOT ALBATROSS_PTH; do
  if [[ ! -e "${!name}" ]]; then
    missing_paths+=("${name}=${!name}")
  fi
done
if [[ ! "${VLLM_RWKV7_MODEL}" =~ ^https?:// && ! -e "${VLLM_RWKV7_MODEL}" ]]; then
  missing_paths+=("VLLM_RWKV7_MODEL=${VLLM_RWKV7_MODEL}")
fi
if [[ ! -d "${albatross_impl_dir}" ]]; then
  missing_paths+=("ALBATROSS_ROOT/ALBATROSS_IMPL=${albatross_impl_dir}")
fi

if (( ${#missing_paths[@]} > 0 )); then
  printf 'Missing required RWKV7 Albatross test paths:\n' >&2
  printf '  %s\n' "${missing_paths[@]}" >&2
  printf '\nCheck .env or RWKV7_ALBATROSS_ENV_FILE before running pytest.\n' >&2
  printf 'Use --server to include the OpenAI server alignment test.\n' >&2
  exit 2
fi

python_bin="${PYTHON:-.venv/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  printf 'Python executable not found or not executable: %s\n' "${python_bin}" >&2
  printf 'Set PYTHON=/path/to/python or create the vLLM .venv first.\n' >&2
  exit 2
fi

exec "${python_bin}" -m pytest \
  tests/models/language/generation/test_rwkv7_albatross.py \
  -q -s "$@"
