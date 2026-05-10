#!/usr/bin/env bash
set -euo pipefail

# Manual decode-node bring-up extracted from:
# pd_tp_dep_mooncake_offload_nixl_400G_producer_random_python_bench.yaml
#
# Defaults are for a literal one-node D instance:
#   DP_SIZE=4, DATA_PARALLEL_SIZE_LOCAL=4, NODE_RANK=0
#
# To match the YAML's 2-node decode instance, run this script on both D nodes:
#   D_NNODES=2 DECODE_MASTER_ADDR=<first-D-node-ip> NODE_RANK=0 ./start_1p1d_decode.sh
#   D_NNODES=2 DECODE_MASTER_ADDR=<first-D-node-ip> NODE_RANK=1 ./start_1p1d_decode.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-nvidia/Kimi-K2.5-NVFP4}"
PORT="${PORT:-8000}"
VLLM_REPO="${VLLM_REPO:-/home/${USER}/vllm}"
MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_PATH:-${SCRIPT_DIR}/mooncake_config.json}"

DATA_PARALLEL_SIZE_LOCAL="${DATA_PARALLEL_SIZE_LOCAL:-4}"
D_NNODES="${D_NNODES:-1}"
DP_SIZE="${DP_SIZE:-$((D_NNODES * DATA_PARALLEL_SIZE_LOCAL))}"
NODE_RANK="${NODE_RANK:-0}"
DATA_PARALLEL_START_RANK="${DATA_PARALLEL_START_RANK:-$((NODE_RANK * DATA_PARALLEL_SIZE_LOCAL))}"

detect_ip() {
  ip route get 8.8.8.8 2>/dev/null | awk '{for (i = 1; i <= NF; i++) if ($i == "src") {print $(i + 1); exit}}'
}

DECODE_MASTER_ADDR="${DECODE_MASTER_ADDR:-$(detect_ip || true)}"
if [[ -z "${DECODE_MASTER_ADDR}" ]]; then
  echo "DECODE_MASTER_ADDR is required when this node's IP cannot be auto-detected" >&2
  exit 1
fi

mkdir -p "/tmp/${USER}"

if [[ -z "${VLLM_NIXL_SIDE_CHANNEL_HOST:-}" ]]; then
  VLLM_NIXL_SIDE_CHANNEL_HOST="$(detect_ip || true)"
  if [[ -n "${VLLM_NIXL_SIDE_CHANNEL_HOST}" ]]; then
    export VLLM_NIXL_SIDE_CHANNEL_HOST
  fi
fi

export VLLM_RPC_BASE_PATH="${VLLM_RPC_BASE_PATH:-/tmp/${USER}}"
export VLLM_MOE_DP_CHUNK_SIZE="${VLLM_MOE_DP_CHUNK_SIZE:-1024}"
export VLLM_FLASHINFER_MOE_BACKEND="${VLLM_FLASHINFER_MOE_BACKEND:-latency}"
export VLLM_RANDOMIZE_DP_DUMMY_INPUTS="${VLLM_RANDOMIZE_DP_DUMMY_INPUTS:-1}"

export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-600000}"
export VLLM_LOG_STATS_INTERVAL="${VLLM_LOG_STATS_INTERVAL:-1}"

if [[ -d "${VLLM_REPO}/.venv/bin" ]]; then
  export PATH="${VLLM_REPO}/.venv/bin:${PATH}"
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.1}"
export HF_HOME="${HF_HOME:-/home/${USER}/hf-models}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-true}"

export MOONCAKE_CONFIG_PATH
export MC_ENABLE_DEST_DEVICE_AFFINITY="${MC_ENABLE_DEST_DEVICE_AFFINITY:-1}"

export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
export UCX_MEMTYPE_REG_WHOLE="${UCX_MEMTYPE_REG_WHOLE:-n}"
export UCX_TLS="${UCX_TLS:-cuda_ipc,cuda_copy,tcp}"
export UCX_CUDA_IPC_ENABLE_MNNVL="${UCX_CUDA_IPC_ENABLE_MNNVL:-y}"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_1,mlx5_3,mlx5_4}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enP6p9s0np0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-5}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-300}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-300}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enP6p9s0np0}"

cd "${VLLM_REPO}"

rank_args=()
if (( DATA_PARALLEL_START_RANK > 0 )); then
  rank_args+=(--data-parallel-start-rank "${DATA_PARALLEL_START_RANK}")
fi

exec vllm serve "${MODEL}" \
  --port "${PORT}" \
  --trust-remote-code \
  --language-model-only \
  --load-format fastsafetensors \
  -dp "${DP_SIZE}" \
  -ep \
  -O3 \
  --block-size 64 \
  --max-model-len 118400 \
  --data-parallel-size-local "${DATA_PARALLEL_SIZE_LOCAL}" \
  --data-parallel-address "${DECODE_MASTER_ADDR}" \
  --data-parallel-hybrid-lb \
  --gpu-memory-utilization 0.85 \
  --attention_config.disable_flashinfer_prefill false \
  --kv-cache-dtype fp8 \
  --all2all-backend flashinfer_nvlink_one_sided \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --disable-hybrid-kv-cache-manager \
  --kv-transfer-config '{"kv_connector": "MultiConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"connectors": [{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail","kv_buffer_device":"cuda","kv_connector_extra_config":{"enforce_handshake_compat":false}}, {"kv_connector": "MooncakeStoreConnector", "kv_role": "kv_producer", "kv_connector_extra_config": {"load_async": true, "enable_cross_layers_blocks": true}}]}}' \
  --disable-uvicorn-access-log \
  --enable-sleep-mode \
  "${rank_args[@]}" \
  "$@"
