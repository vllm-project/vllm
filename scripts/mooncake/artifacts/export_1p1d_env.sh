#!/usr/bin/env bash
# Source this file before running the 1P/1D helper scripts:
#   source recipes/crusoe/kimik25/low_latency/artifacts/export_1p1d_env.sh

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "export_1p1d_env.sh must be sourced from bash" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Note: run this with 'source' to export variables into your current shell." >&2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

prepend_path() {
  local var_name=$1
  local value=$2
  local current_value="${!var_name:-}"

  case ":${current_value}:" in
    *":${value}:"*) ;;
    *) export "${var_name}=${value}${current_value:+:${current_value}}" ;;
  esac
}

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.1}"
prepend_path PATH "${CUDA_HOME}/bin"
prepend_path PATH "/home/${USER}/.local/bin"
prepend_path PATH "/home/${USER}/code/uv_envs/py312/bin"

prepend_path LD_LIBRARY_PATH "${CUDA_HOME}/lib64"
prepend_path LD_LIBRARY_PATH "/home/${USER}/.local/share/uv/python/cpython-3.12.13-linux-aarch64-gnu/lib"
prepend_path LD_LIBRARY_PATH "/home/${USER}/.local/mooncake-deps/lib"

export VLLM_REPO="${VLLM_REPO:-/home/${USER}/vllm-mooncake}"
export VLLM_BENCH_REPO="${VLLM_BENCH_REPO:-/home/${USER}/vllm-dao}"
export ROUTER_REPO="${ROUTER_REPO:-/home/${USER}/router-internal}"

export HF_HOME="${HF_HOME:-/home/${USER}/hf-models}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-true}"
export MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_PATH:-${SCRIPT_DIR}/mooncake_config.json}"

export MODEL="${MODEL:-nvidia/Kimi-K2.5-NVFP4}"
export PORT="${PORT:-8000}"
export ROUTER_PORT="${ROUTER_PORT:-8100}"
export PROMETHEUS_PORT="${PROMETHEUS_PORT:-29000}"

echo "Exported 1P/1D environment:"
echo "  VLLM_REPO=${VLLM_REPO}"
echo "  VLLM_BENCH_REPO=${VLLM_BENCH_REPO}"
echo "  ROUTER_REPO=${ROUTER_REPO}"
echo "  MOONCAKE_CONFIG_PATH=${MOONCAKE_CONFIG_PATH}"
echo "  ROUTER_PORT=${ROUTER_PORT}"
echo "  PROMETHEUS_PORT=${PROMETHEUS_PORT}"
