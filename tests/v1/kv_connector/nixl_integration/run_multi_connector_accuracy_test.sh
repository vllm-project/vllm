#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Integration accuracy test for MultiConnector (NixlConnector + OffloadingConnector).
#
# Launches a P/D setup where both prefill and decode instances use MultiConnector
# wrapping NixlConnector and OffloadingConnector, then runs gsm8k accuracy via
# test_accuracy.py.
#
# By default runs two configurations:
#   1. Normal KV layout (NixlConnector without cross-layer blocks)
#   2. Cross-layer KV layout (NixlConnector with enable_cross_layers_blocks)
#
# Usage:
#   bash tests/v1/kv_connector/nixl_integration/run_multi_connector_accuracy_test.sh
#
# Environment variables:
#   MODEL_NAMES              - model to test (default: Qwen/Qwen3-0.6B)
#   GPU_MEMORY_UTILIZATION   - GPU memory fraction (default: 0.6)
#   VLLM_SERVE_EXTRA_ARGS    - comma-separated extra args for vllm serve
#   SKIP_CROSS_LAYERS        - set to 1 to skip the cross-layer layout test
#   SKIP_NORMAL_LAYOUT       - set to 1 to skip the normal layout test
set -xe

# ── Configuration ────────────────────────────────────────────────────────

MODEL_NAMES=${MODEL_NAMES:-}
if [[ -n "$MODEL_NAMES" ]]; then
  MODELS=("$MODEL_NAMES")
else
  MODELS=("Qwen/Qwen3-0.6B")
fi

GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}
BLOCK_SIZE=${BLOCK_SIZE:-128}
VLLM_SERVE_EXTRA_ARGS=${VLLM_SERVE_EXTRA_ARGS:-}

GIT_ROOT=$(git rev-parse --show-toplevel)
SMI_BIN=$(which nvidia-smi || which rocm-smi || echo "")

# ── KV transfer configs ─────────────────────────────────────────────────

# Normal layout: OffloadingConnector prefers cross-layer but NixlConnector
# does not, so MultiConnector.prefer_cross_layer_blocks = False.
KV_CONFIG_NORMAL='{
  "kv_connector":"MultiConnector",
  "kv_role":"kv_both",
  "kv_connector_extra_config":{
    "connectors":[
      {"kv_connector":"NixlConnector","kv_role":"kv_both"},
      {"kv_connector":"OffloadingConnector","kv_role":"kv_both",
       "kv_connector_extra_config":{"cpu_bytes_to_use":1000000000}}
    ]
  }
}'
# Remove whitespace for CLI safety
KV_CONFIG_NORMAL=$(echo "$KV_CONFIG_NORMAL" | tr -d '[:space:]')

# Cross-layer layout: both connectors prefer cross-layer blocks.
KV_CONFIG_CROSS_LAYERS='{
  "kv_connector":"MultiConnector",
  "kv_role":"kv_both",
  "kv_connector_extra_config":{
    "connectors":[
      {"kv_connector":"NixlConnector","kv_role":"kv_both",
       "kv_connector_extra_config":{"enable_cross_layers_blocks":"True"}},
      {"kv_connector":"OffloadingConnector","kv_role":"kv_both",
       "kv_connector_extra_config":{"cpu_bytes_to_use":1000000000}}
    ]
  }
}'
KV_CONFIG_CROSS_LAYERS=$(echo "$KV_CONFIG_CROSS_LAYERS" | tr -d '[:space:]')

# ── Helpers ──────────────────────────────────────────────────────────────

trap 'kill $(jobs -pr) 2>/dev/null' SIGINT SIGTERM EXIT

wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

cleanup_instances() {
  echo "Cleaning up any running vLLM instances..."
  pkill -f "vllm serve" || true
  sleep 2
}

get_num_gpus() {
  if [[ "$SMI_BIN" == *"nvidia"* ]]; then
    $SMI_BIN --query-gpu=name --format=csv,noheader | wc -l
  elif [[ "$SMI_BIN" == *"rocm"* ]]; then
    $SMI_BIN -l | grep -c GPU
  else
    echo "1"
  fi
}

# ── Run tests for one model with a given KV config ───────────────────────

run_tests_for_model() {
  local model_name=$1
  local kv_config=$2
  local label=$3

  echo "================================================================"
  echo "Testing model: $model_name ($label)"
  echo "KV config: $kv_config"
  echo "================================================================"

  local PREFILL_PORT=8100
  local DECODE_PORT=8200
  local PREFILL_GPU=0
  local DECODE_GPU=1
  local PREFILL_SIDE_CHANNEL_PORT=5559
  local DECODE_SIDE_CHANNEL_PORT=5659

  # ── Start prefill instance ──
  echo "Starting prefill instance on GPU $PREFILL_GPU, port $PREFILL_PORT"
  BASE_CMD="CUDA_VISIBLE_DEVICES=$PREFILL_GPU \
    VLLM_KV_CACHE_LAYOUT='HND' \
    UCX_NET_DEVICES=all \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$PREFILL_SIDE_CHANNEL_PORT \
    vllm serve $model_name \
    --port $PREFILL_PORT \
    --enforce-eager \
    --block-size ${BLOCK_SIZE} \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --tensor-parallel-size 1 \
    --kv-transfer-config '$kv_config'"

  if [[ -n "$VLLM_SERVE_EXTRA_ARGS" ]]; then
    IFS=',' read -r -a extra_args <<< "$VLLM_SERVE_EXTRA_ARGS"
    for arg in "${extra_args[@]}"; do
      BASE_CMD="${BASE_CMD} $arg"
    done
  fi
  eval "$BASE_CMD &"

  # ── Start decode instance ──
  echo "Starting decode instance on GPU $DECODE_GPU, port $DECODE_PORT"
  BASE_CMD="CUDA_VISIBLE_DEVICES=$DECODE_GPU \
    VLLM_KV_CACHE_LAYOUT='HND' \
    UCX_NET_DEVICES=all \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$DECODE_SIDE_CHANNEL_PORT \
    vllm serve $model_name \
    --port $DECODE_PORT \
    --enforce-eager \
    --block-size ${BLOCK_SIZE} \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --tensor-parallel-size 1 \
    --kv-transfer-config '$kv_config'"

  if [[ -n "$VLLM_SERVE_EXTRA_ARGS" ]]; then
    IFS=',' read -r -a extra_args <<< "$VLLM_SERVE_EXTRA_ARGS"
    for arg in "${extra_args[@]}"; do
      BASE_CMD="${BASE_CMD} $arg"
    done
  fi
  eval "$BASE_CMD &"

  # ── Wait for servers ──
  echo "Waiting for prefill instance on port $PREFILL_PORT to start..."
  wait_for_server "$PREFILL_PORT"
  echo "Waiting for decode instance on port $DECODE_PORT to start..."
  wait_for_server "$DECODE_PORT"

  # ── Start proxy ──
  PROXY_CMD="python3 ${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py --port 8192"
  PROXY_CMD+=" --prefiller-hosts localhost"
  PROXY_CMD+=" --prefiller-ports $PREFILL_PORT"
  PROXY_CMD+=" --decoder-hosts localhost"
  PROXY_CMD+=" --decoder-ports $DECODE_PORT"

  echo "Starting proxy server with command: $PROXY_CMD"
  $PROXY_CMD &
  sleep 5

  # ── Run accuracy test ──
  echo "Running accuracy tests for $model_name ($label)"
  TEST_MODEL=$model_name python3 -m pytest -s -x \
    "${GIT_ROOT}"/tests/v1/kv_connector/nixl_integration/test_accuracy.py

  # ── Cleanup ──
  cleanup_instances
  sleep 3
}

# ── Main ─────────────────────────────────────────────────────────────────

for model in "${MODELS[@]}"; do
  if [[ -z "${SKIP_NORMAL_LAYOUT:-}" ]]; then
    run_tests_for_model "$model" "$KV_CONFIG_NORMAL" "MultiConnector normal layout"
  fi

  if [[ -z "${SKIP_CROSS_LAYERS:-}" ]]; then
    run_tests_for_model "$model" "$KV_CONFIG_CROSS_LAYERS" "MultiConnector cross-layer layout"
  fi
done

echo "All MultiConnector accuracy tests passed!"
