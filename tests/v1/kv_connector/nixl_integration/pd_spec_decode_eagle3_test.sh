#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# PD disaggregation + EAGLE3 speculative decoding acceptance length test.
#
# Starts prefill + decode vLLM servers with NixlConnector and EAGLE3,
# then runs test_pd_spec_decode_eagle3.py to validate acceptance length
# matches standalone SD baselines.
#
# Supports multiple model configs (qwen3-8b-eagle3, gpt-oss-20b-eagle3)
# and optional attention backend sweeping.
#
# Usage:
#   # Run single config with default backend:
#   CUDA_VISIBLE_DEVICES=0,1 bash tests/v1/kv_connector/nixl_integration/pd_spec_decode_eagle3_test.sh \
#       --config qwen3-8b-eagle3
#
#   # Run with explicit backend:
#   CUDA_VISIBLE_DEVICES=0,1 bash tests/v1/kv_connector/nixl_integration/pd_spec_decode_eagle3_test.sh \
#       --config gpt-oss-20b-eagle3 --attention-backend TRITON_ATTN
#
#   # Run all configs with their default backends:
#   CUDA_VISIBLE_DEVICES=0,1 bash tests/v1/kv_connector/nixl_integration/pd_spec_decode_eagle3_test.sh --all
#
# Environment variables:
#   GPU_MEMORY_UTILIZATION - (default: 0.7)
#   KV_BUFFER_DEVICE       - cuda or cpu (default: cuda)
set -x

if ! test -w /tmp/tiktoken-rs-cache 2>/dev/null; then
  export TMPDIR="${HOME}/.cache/tmp"
  mkdir -p "$TMPDIR"
fi

# ── Parse args ───────────────────────────────────────────────────────────

CONFIG=""
ATTENTION_BACKEND=""
RUN_ALL=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --config) CONFIG="$2"; shift 2 ;;
    --attention-backend) ATTENTION_BACKEND="$2"; shift 2 ;;
    --all) RUN_ALL=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Preset configs ───────────────────────────────────────────────────────

declare -A CFG_MODEL CFG_SD_MODEL CFG_BACKEND CFG_MAX_MODEL_LEN

CFG_MODEL[qwen3-8b-eagle3]="Qwen/Qwen3-8B"
CFG_SD_MODEL[qwen3-8b-eagle3]="RedHatAI/Qwen3-8B-speculator.eagle3"
CFG_BACKEND[qwen3-8b-eagle3]=""
CFG_MAX_MODEL_LEN[qwen3-8b-eagle3]=16384

CFG_MODEL[gpt-oss-20b-eagle3]="openai/gpt-oss-20b"
CFG_SD_MODEL[gpt-oss-20b-eagle3]="RedHatAI/gpt-oss-20b-speculator.eagle3"
CFG_BACKEND[gpt-oss-20b-eagle3]="TRITON_ATTN"
CFG_MAX_MODEL_LEN[gpt-oss-20b-eagle3]=16384

ALL_CONFIGS=("qwen3-8b-eagle3" "gpt-oss-20b-eagle3")

# ── Cluster layout ───────────────────────────────────────────────────────

PREFILLER_TP_SIZE=${PREFILLER_TP_SIZE:-1}
DECODER_TP_SIZE=${DECODER_TP_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.7}
BLOCK_SIZE=${BLOCK_SIZE:-16}
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-3}
KV_BUFFER_DEVICE=${KV_BUFFER_DEVICE:-cuda}

GIT_ROOT=$(git rev-parse --show-toplevel)

SMI_BIN=$(which nvidia-smi || which rocm-smi || echo "")

# ── Helpers ──────────────────────────────────────────────────────────────

cleanup_instances() {
  echo ""
  echo "Cleaning up..."
  kill $(jobs -pr) 2>/dev/null || true
  sleep 1
  kill -9 $(jobs -pr) 2>/dev/null || true
  pkill -9 -f "vllm serve" 2>/dev/null || true
  pkill -9 -f "toy_proxy_server.*8192" 2>/dev/null || true
  sleep 1
  echo "Cleanup done."
}
trap cleanup_instances EXIT
trap 'echo " Interrupted."; exit 130' INT TERM

wait_for_server() {
  local port=$1
  local deadline=600
  local elapsed=0
  echo "Waiting for server on port ${port}..."
  while [ $elapsed -lt $deadline ]; do
    if curl -s "localhost:${port}/v1/completions" > /dev/null 2>&1; then
      echo "Server on port ${port} ready (${elapsed}s)"
      return 0
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  echo "FAIL: Server on port ${port} did not start within ${deadline}s"
  exit 1
}

# ── Resolve GPU list ─────────────────────────────────────────────────────

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra ALL_GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
  ALL_GPUS=()
  if [[ "$SMI_BIN" == *"nvidia"* ]]; then
    num=$($SMI_BIN --query-gpu=name --format=csv,noheader | wc -l)
  elif [[ "$SMI_BIN" == *"rocm"* ]]; then
    num=$($SMI_BIN -l | grep -c GPU)
  else
    num=1
  fi
  for (( g=0; g<num; g++ )); do ALL_GPUS+=($g); done
fi

TOTAL_GPUS_NEEDED=$(( PREFILLER_TP_SIZE + DECODER_TP_SIZE ))
if [[ ${#ALL_GPUS[@]} -lt $TOTAL_GPUS_NEEDED ]]; then
  echo "FAIL: Need $TOTAL_GPUS_NEEDED GPUs but only have ${#ALL_GPUS[@]}"
  exit 1
fi

# ── Run a single config ─────────────────────────────────────────────────

run_config() {
  local config_name=$1
  local backend_override=$2

  local model="${CFG_MODEL[$config_name]}"
  local sd_model="${CFG_SD_MODEL[$config_name]}"
  local default_backend="${CFG_BACKEND[$config_name]}"
  local max_model_len="${CFG_MAX_MODEL_LEN[$config_name]}"
  local backend="${backend_override:-$default_backend}"

  if [[ -z "$model" ]]; then
    echo "Unknown config: $config_name"
    echo "Available: ${ALL_CONFIGS[*]}"
    exit 1
  fi

  local prefill_spec="{\"method\":\"eagle3\",\"model\":\"${sd_model}\",\"num_speculative_tokens\":1,\"max_model_len\":${max_model_len}}"
  local decode_spec="{\"method\":\"eagle3\",\"model\":\"${sd_model}\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS},\"max_model_len\":${max_model_len}}"

  if [[ "$KV_BUFFER_DEVICE" == "cuda" ]]; then
    local kv_config='{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
  else
    local kv_config="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"${KV_BUFFER_DEVICE}\"}"
  fi

  local backend_args=""
  if [[ -n "$backend" ]]; then
    backend_args="--attention-backend $backend"
  fi

  echo ""
  echo "================================================================"
  echo " PD + EAGLE3 — NixlConnector"
  echo "================================================================"
  echo " Config:         ${config_name}"
  echo " Model:          ${model}"
  echo " Drafter:        ${sd_model}"
  echo " Spec tokens:    prefill=1, decode=${NUM_SPEC_TOKENS}"
  echo " Max model len:  ${max_model_len}"
  echo " Backend:        ${backend:-'(default)'}"
  echo " KV buffer:      ${KV_BUFFER_DEVICE}"
  echo " GPUs:           ${ALL_GPUS[*]} (need ${TOTAL_GPUS_NEEDED})"
  echo "================================================================"

  # Assign GPUs
  local gpu_idx=0
  local prefill_gpu="${ALL_GPUS[$gpu_idx]}"
  gpu_idx=$((gpu_idx + 1))
  for (( j=1; j < PREFILLER_TP_SIZE; j++ )); do
    prefill_gpu="${prefill_gpu},${ALL_GPUS[$gpu_idx]}"
    gpu_idx=$((gpu_idx + 1))
  done

  local decode_gpu="${ALL_GPUS[$gpu_idx]}"
  gpu_idx=$((gpu_idx + 1))
  for (( j=1; j < DECODER_TP_SIZE; j++ )); do
    decode_gpu="${decode_gpu},${ALL_GPUS[$gpu_idx]}"
    gpu_idx=$((gpu_idx + 1))
  done

  local prefill_port=8100
  local decode_port=8200
  local proxy_port=8192

  # Start prefill
  echo "Starting prefill on GPU $prefill_gpu, port $prefill_port"
  CUDA_VISIBLE_DEVICES=$prefill_gpu \
  VLLM_KV_CACHE_LAYOUT='HND' \
  UCX_NET_DEVICES=all \
  VLLM_NIXL_SIDE_CHANNEL_PORT=5559 \
  vllm serve $model \
    --port $prefill_port \
    --enforce-eager \
    --max-model-len $max_model_len \
    --block-size ${BLOCK_SIZE} \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --tensor-parallel-size $PREFILLER_TP_SIZE \
    --kv-transfer-config "$kv_config" \
    --speculative-config "$prefill_spec" \
    --disable-log-requests \
    ${backend_args} &

  # Start decode
  echo "Starting decode on GPU $decode_gpu, port $decode_port"
  CUDA_VISIBLE_DEVICES=$decode_gpu \
  VLLM_KV_CACHE_LAYOUT='HND' \
  UCX_NET_DEVICES=all \
  VLLM_NIXL_SIDE_CHANNEL_PORT=5659 \
  vllm serve $model \
    --port $decode_port \
    --enforce-eager \
    --max-model-len $max_model_len \
    --block-size ${BLOCK_SIZE} \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --tensor-parallel-size $DECODER_TP_SIZE \
    --kv-transfer-config "$kv_config" \
    --speculative-config "$decode_spec" \
    --disable-log-requests \
    ${backend_args} &

  wait_for_server "$prefill_port"
  wait_for_server "$decode_port"

  # Start proxy
  echo "Starting proxy on port $proxy_port..."
  python3 "${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" \
    --port $proxy_port \
    --prefiller-hosts localhost \
    --prefiller-ports $prefill_port \
    --decoder-hosts localhost \
    --decoder-ports $decode_port &

  sleep 5

  # Run pytest
  echo "Running acceptance test for ${config_name}..."
  TEST_CONFIG="$config_name" \
  TEST_MODEL="$model" \
  DECODE_PORT="$decode_port" \
  PROXY_PORT="$proxy_port" \
  pytest -xvs \
    "${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/test_pd_spec_decode_eagle3.py"

  local rc=$?

  # Cleanup for next run
  cleanup_instances
  sleep 3

  return $rc
}

# ── Main ─────────────────────────────────────────────────────────────────

if [[ "$RUN_ALL" == "true" ]]; then
  for cfg in "${ALL_CONFIGS[@]}"; do
    run_config "$cfg" "$ATTENTION_BACKEND"
  done
elif [[ -n "$CONFIG" ]]; then
  run_config "$CONFIG" "$ATTENTION_BACKEND"
else
  echo "Usage:"
  echo "  $0 --config <config-name> [--attention-backend <backend>]"
  echo "  $0 --all [--attention-backend <backend>]"
  echo ""
  echo "Available configs: ${ALL_CONFIGS[*]}"
  exit 1
fi

echo ""
echo "All tests passed!"
