#!/usr/bin/env bash
# INNER script — runs inside Docker container
# Sets NPU env vars in the same process before exec vllm-hust
set -euo pipefail

# ── NPU device selection ──
export ASCEND_RT_VISIBLE_DEVICES="${VLLM_NPU_DEVICE:-0}"
export ASCEND_VISIBLE_DEVICES="${VLLM_NPU_DEVICE:-0}"
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# ── Isolate cache (avoid polluting host $HOME) ──
export HOME=/tmp/vllm-hust-home
export XDG_CACHE_HOME="$HOME/.cache"
export XDG_CONFIG_HOME="$HOME/.config"
export VLLM_CACHE_ROOT="$HOME/.cache/vllm"
export VLLM_CONFIG_ROOT="$HOME/.config/vllm"
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
         "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT"

# ── Conda env path ──
VLLM_BIN="/workspace/miniconda3/envs/vllm-hust-dev/bin/vllm-hust"

# ── Launch ──
exec "${VLLM_BIN}" serve \
  "${VLLM_MODEL:-/data/shared_models/Qwen--Qwen2.5-14B-Instruct}" \
  --served-model-name "${VLLM_MODEL_NAME:-qwen2.5-14b}" \
  --host 0.0.0.0 --port "${VLLM_PORT:-8000}" \
  --tensor-parallel-size "${VLLM_TP_SIZE:-1}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-8192}" \
  --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}" \
  --gpu-memory-utilization 0.9 \
  --dtype bfloat16 \
  --load-format auto \
  --trust-remote-code \
  --max-num-seqs 16 \
  --enforce-eager
