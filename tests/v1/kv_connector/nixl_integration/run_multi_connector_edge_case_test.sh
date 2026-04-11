#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Integration edge-case tests for MultiConnector (NixlConnector + OffloadingConnector).
#
# Launches a P/D setup where both prefill and decode instances use MultiConnector
# wrapping NixlConnector and OffloadingConnector, then runs scenario-based edge
# case tests including Prometheus metrics validation.
#
# Tests cover: block-size boundaries, decode-side cache-hit scenarios
# (cold / full / partial), direct decode (control), and prefill-side CPU
# offload recovery after GPU eviction.
#
# Usage:
#   bash tests/v1/kv_connector/nixl_integration/run_multi_connector_edge_case_test.sh
#
# Environment variables:
#   MODEL_NAMES              - model to test (default: Qwen/Qwen3-0.6B)
#   KV_CACHE_MEMORY_BYTES    - GPU KV cache size in bytes (default: 268435456 = 256 MiB)
#   BLOCK_SIZE               - KV cache block size (default: 128)
#   VLLM_SERVE_EXTRA_ARGS    - comma-separated extra args for vllm serve
set -xe

# ── Configuration ────────────────────────────────────────────────────────

MODEL_NAMES=${MODEL_NAMES:-}
if [[ -n "$MODEL_NAMES" ]]; then
  MODELS=("$MODEL_NAMES")
else
  MODELS=("Qwen/Qwen3-0.6B")
fi

KV_CACHE_MEMORY_BYTES=${KV_CACHE_MEMORY_BYTES:-268435456}  # 256 MiB
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
BLOCK_SIZE=${BLOCK_SIZE:-128}
VLLM_SERVE_EXTRA_ARGS=${VLLM_SERVE_EXTRA_ARGS:-}

GIT_ROOT=$(git rev-parse --show-toplevel)

# ── KV transfer config ──────────────────────────────────────────────────

KV_CONFIG='{
  "kv_connector":"MultiConnector",
  "kv_role":"kv_both",
  "kv_connector_extra_config":{
    "connectors":[
      {"kv_connector":"NixlConnector","kv_role":"kv_both"},
      {"kv_connector":"OffloadingConnector","kv_role":"kv_both",
       "kv_connector_extra_config":{"cpu_bytes_to_use":2147483648}}
    ]
  }
}'
KV_CONFIG=$(echo "$KV_CONFIG" | tr -d '[:space:]')

# ── Helpers ──────────────────────────────────────────────────────────────

trap 'kill $(jobs -pr) 2>/dev/null || true' SIGINT SIGTERM EXIT

wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

cleanup_instances() {
  echo "Cleaning up any running vLLM instances and proxy..."
  pkill -f "vllm serve" || true
  pkill -f "toy_proxy_server.py" || true
  sleep 2
}

# ── Run tests for one model ──────────────────────────────────────────────

run_tests_for_model() {
  local model_name=$1

  echo "================================================================"
  echo "Testing model: $model_name (MultiConnector edge cases)"
  echo "================================================================"

  local PREFILL_PORT=8100
  local DECODE_PORT=8200
  local PROXY_PORT=8192
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
    vllm serve \"$model_name\" \
    --port $PREFILL_PORT \
    --enforce-eager \
    --block-size ${BLOCK_SIZE} \
    --max-model-len $MAX_MODEL_LEN \
    --kv-cache-memory-bytes $KV_CACHE_MEMORY_BYTES \
    --tensor-parallel-size 1 \
    --kv-transfer-config '$KV_CONFIG'"

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
    vllm serve \"$model_name\" \
    --port $DECODE_PORT \
    --enforce-eager \
    --block-size ${BLOCK_SIZE} \
    --max-model-len $MAX_MODEL_LEN \
    --kv-cache-memory-bytes $KV_CACHE_MEMORY_BYTES \
    --tensor-parallel-size 1 \
    --kv-transfer-config '$KV_CONFIG'"

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
  echo "Starting proxy server on port $PROXY_PORT"
  python3 "${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" \
    --port "$PROXY_PORT" \
    --prefiller-hosts localhost \
    --prefiller-ports "$PREFILL_PORT" \
    --decoder-hosts localhost \
    --decoder-ports "$DECODE_PORT" &
  sleep 5

  # ── Run edge case tests ──
  echo "Running MultiConnector edge case tests for $model_name"
  PREFILL_PORT=$PREFILL_PORT \
  DECODE_PORT=$DECODE_PORT \
  PROXY_PORT=$PROXY_PORT \
  BLOCK_SIZE=$BLOCK_SIZE \
    python3 -m pytest -s -x \
    "${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/test_multi_connector_edge_cases.py"

  # ── Cleanup ──
  cleanup_instances
  sleep 3
}

# ── Main ─────────────────────────────────────────────────────────────────

for model in "${MODELS[@]}"; do
  run_tests_for_model "$model"
done

echo "All MultiConnector edge case tests passed!"
