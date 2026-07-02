#!/usr/bin/env bash
# INNER script — runs inside Docker container
# Launches Qwen3-235B-A22B-W8A8 (MoE) on all 8 NPUs
set -euo pipefail

# ── NPU device selection (all 8 for MoE) ──
export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export ASCEND_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# ── vLLM plugin & offline flags ──
export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ── Ascend MoE performance optimizations (910B / A2) ──
# FlashComm1: optimize TP all-reduce communication for high concurrency
export VLLM_ASCEND_ENABLE_FLASHCOMM1="${VLLM_ASCEND_ENABLE_FLASHCOMM1:-1}"

# ── Source ATB env if available (matches hust-ascend-manager behavior) ──
if [[ -n "${HUST_ATB_SET_ENV:-}" && -f "${HUST_ATB_SET_ENV}" ]]; then
  set +u; source "${HUST_ATB_SET_ENV}" --cxx_abi=1; set -u
fi

# ── Isolate cache ──
export HOME=/tmp/vllm-hust-home
export XDG_CACHE_HOME="$HOME/.cache"
export XDG_CONFIG_HOME="$HOME/.config"
export VLLM_CACHE_ROOT="$HOME/.cache/vllm"
export VLLM_CONFIG_ROOT="$HOME/.config/vllm"
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
         "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT"

# ── Conda env path ──
VLLM_BIN="/workspace/miniconda3/envs/vllm-hust-dev/bin/vllm-hust"

# ── Launch MoE model ──
exec "${VLLM_BIN}" serve \
  "${VLLM_MODEL:-/data/shared_models/modelscope_cache/vllm-ascend/Qwen3-235B-A22B-W8A8}" \
  --served-model-name "${VLLM_MODEL_NAME:-qwen3-235b-a22b-w8a8}" \
  --host 0.0.0.0 --port "${VLLM_PORT:-8000}" \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --quantization ascend \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-8192}" \
  --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS:-4096}" \
  --gpu-memory-utilization 0.9 \
  --dtype bfloat16 \
  --load-format auto \
  --trust-remote-code \
  --max-num-seqs 16
