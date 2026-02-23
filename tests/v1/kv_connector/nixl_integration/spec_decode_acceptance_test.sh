#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# NixlConnector PD + speculative decoding acceptance length test.
# Tests EAGLE3 acceptance length for both RDMA (cuda) and CPU host (cpu)
# KV buffer device paths.
#
# For each kv_buffer_device setting, starts prefill + decode vllm servers
# with NixlConnector, then runs test_spec_decode_acceptance.py to validate
# acceptance length matches the standalone SD baseline.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1 bash tests/v1/kv_connector/nixl_integration/spec_decode_acceptance_test.sh
#
# Environment variables:
#   KV_BUFFER_DEVICES   - space-separated list of devices to test
#                         (default: "cuda cpu")
#   SD_METHOD           - spec decode method (default: eagle3)
#   SD_MODEL            - drafter model path
#   MODEL_NAME          - target model (default: meta-llama/Llama-3.1-8B-Instruct)
#   NUM_SPEC_TOKENS     - number of speculative tokens (default: 3)
#   GPU_MEMORY_UTILIZATION - (default: 0.7)
set -x

# ── Model & spec decode config ──────────────────────────────────────────

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
SD_METHOD="${SD_METHOD:-eagle3}"
SD_MODEL="${SD_MODEL:-RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

PREFILL_SPEC_CONFIG="{\"method\":\"${SD_METHOD}\",\"model\":\"${SD_MODEL}\",\"num_speculative_tokens\":1,\"max_model_len\":${MAX_MODEL_LEN}}"
DECODE_SPEC_CONFIG="{\"method\":\"${SD_METHOD}\",\"model\":\"${SD_MODEL}\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS},\"max_model_len\":${MAX_MODEL_LEN}}"

# ── Test matrix ──────────────────────────────────────────────────────────

KV_BUFFER_DEVICES="${KV_BUFFER_DEVICES:-cuda cpu}"

# ── Cluster layout ───────────────────────────────────────────────────────

NUM_PREFILL_INSTANCES=${NUM_PREFILL_INSTANCES:-1}
NUM_DECODE_INSTANCES=${NUM_DECODE_INSTANCES:-1}
PREFILLER_TP_SIZE=${PREFILLER_TP_SIZE:-1}
DECODER_TP_SIZE=${DECODER_TP_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.7}
BLOCK_SIZE=${BLOCK_SIZE:-16}

GIT_ROOT=$(git rev-parse --show-toplevel)

SMI_BIN=$(which nvidia-smi || which rocm-smi || echo "")

cleanup_instances() {
  echo ""
  echo "Cleaning up..."
  kill $(jobs -pr) 2>/dev/null || true
  sleep 1
  kill -9 $(jobs -pr) 2>/dev/null || true
  pkill -9 -f "vllm serve.*${MODEL_NAME}" 2>/dev/null || true
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
      echo "Server on port ${port} ready"
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

TOTAL_GPUS_NEEDED=$(( (NUM_PREFILL_INSTANCES * PREFILLER_TP_SIZE) + (NUM_DECODE_INSTANCES * DECODER_TP_SIZE) ))
if [[ ${#ALL_GPUS[@]} -lt $TOTAL_GPUS_NEEDED ]]; then
  echo "FAIL: Need $TOTAL_GPUS_NEEDED GPUs but only have ${#ALL_GPUS[@]} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set})"
  exit 1
fi

# ── Run one test iteration ───────────────────────────────────────────────

run_test_for_device() {
  local kv_device=$1

  if [[ "$kv_device" == "cuda" ]]; then
    local kv_config='{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
  else
    local kv_config="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"${kv_device}\"}"
  fi

  echo ""
  echo "================================================================"
  echo "NixlConnector PD + Spec Decode Acceptance Test (kv_buffer_device=${kv_device})"
  echo "================================================================"
  echo "Model:            ${MODEL_NAME}"
  echo "SD method:        ${SD_METHOD}"
  echo "SD model:         ${SD_MODEL}"
  echo "Spec tokens:      ${NUM_SPEC_TOKENS}"
  echo "KV buffer device: ${kv_device}"
  echo "GPUs available:   ${ALL_GPUS[*]}"
  echo "================================================================"

  local PREFILL_HOSTS=()
  local PREFILL_PORTS=()
  local DECODE_HOSTS=()
  local DECODE_PORTS=()
  local GPU_IDX=0

  # Start prefill instances
  for i in $(seq 0 $((NUM_PREFILL_INSTANCES-1))); do
    local GPU_ID="${ALL_GPUS[$GPU_IDX]}"
    GPU_IDX=$((GPU_IDX + 1))
    for (( j=1; j < PREFILLER_TP_SIZE; j++ )); do
      GPU_ID="${GPU_ID},${ALL_GPUS[$GPU_IDX]}"
      GPU_IDX=$((GPU_IDX + 1))
    done

    local PORT=$((8100 + i))
    local SIDE_CHANNEL_PORT=$((5559 + i))

    echo "Starting prefill instance $i on GPU $GPU_ID, port $PORT"
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    VLLM_KV_CACHE_LAYOUT='HND' \
    UCX_NET_DEVICES=all \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT \
    vllm serve $MODEL_NAME \
      --port $PORT \
      --enforce-eager \
      --max-model-len $MAX_MODEL_LEN \
      --block-size ${BLOCK_SIZE} \
      --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
      --tensor-parallel-size $PREFILLER_TP_SIZE \
      --kv-transfer-config "$kv_config" \
      --speculative-config "$PREFILL_SPEC_CONFIG" \
      --attention-backend FLASH_ATTN \
      --disable-log-requests &

    PREFILL_HOSTS+=("localhost")
    PREFILL_PORTS+=("$PORT")
  done

  # Start decode instances
  for i in $(seq 0 $((NUM_DECODE_INSTANCES-1))); do
    local GPU_ID="${ALL_GPUS[$GPU_IDX]}"
    GPU_IDX=$((GPU_IDX + 1))
    for (( j=1; j < DECODER_TP_SIZE; j++ )); do
      GPU_ID="${GPU_ID},${ALL_GPUS[$GPU_IDX]}"
      GPU_IDX=$((GPU_IDX + 1))
    done

    local PORT=$((8200 + i))
    local SIDE_CHANNEL_PORT=$((5659 + i * $DECODER_TP_SIZE))

    echo "Starting decode instance $i on GPU $GPU_ID, port $PORT"
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    VLLM_KV_CACHE_LAYOUT='HND' \
    UCX_NET_DEVICES=all \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT \
    vllm serve $MODEL_NAME \
      --port $PORT \
      --enforce-eager \
      --max-model-len $MAX_MODEL_LEN \
      --block-size ${BLOCK_SIZE} \
      --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
      --tensor-parallel-size $DECODER_TP_SIZE \
      --kv-transfer-config "$kv_config" \
      --speculative-config "$DECODE_SPEC_CONFIG" \
      --attention-backend FLASH_ATTN \
      --disable-log-requests &

    DECODE_HOSTS+=("localhost")
    DECODE_PORTS+=("$PORT")
  done

  # Wait for servers
  for PORT in "${PREFILL_PORTS[@]}"; do
    wait_for_server "$PORT"
  done
  for PORT in "${DECODE_PORTS[@]}"; do
    wait_for_server "$PORT"
  done

  # Start proxy
  local PROXY_PORT=8192
  echo "Starting proxy server on port $PROXY_PORT..."
  python3 "${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" \
    --port $PROXY_PORT \
    --prefiller-hosts ${PREFILL_HOSTS[*]} \
    --prefiller-ports ${PREFILL_PORTS[*]} \
    --decoder-hosts ${DECODE_HOSTS[*]} \
    --decoder-ports ${DECODE_PORTS[*]} &

  sleep 5

  # Run test
  echo "Running spec decode acceptance test (kv_buffer_device=${kv_device})..."
  DECODE_PORT=${DECODE_PORTS[0]} \
  TEST_MODEL=$MODEL_NAME \
  python3 -m pytest -s -x "${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/test_spec_decode_acceptance.py"

  # Tear down before next iteration
  cleanup_instances
  sleep 3
}

# ── Main: loop over kv_buffer_device values ──────────────────────────────

for device in $KV_BUFFER_DEVICES; do
  run_test_for_device "$device"
done

echo "=== All spec decode acceptance tests passed ==="
