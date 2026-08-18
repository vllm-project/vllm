#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Integration accuracy test for MultiConnector (NixlConnector + OffloadingConnector)
# with speculative decoding enabled.
#
# Launches a P/D setup where both prefill and decode instances use MultiConnector
# wrapping NixlConnector and OffloadingConnector, with ngram speculative decoding
# turned on, then runs gsm8k accuracy via test_accuracy.py.
#
# Speculative decoding is output-preserving under greedy sampling, so gsm8k
# accuracy through this pipeline must stay within RTOL of the non-spec baseline;
# a drop signals speculative advance/rollback corrupting KV across the NIXL
# transfer and CPU offload path.
#
# Usage:
#   bash tests/v1/kv_connector/nixl_integration/run_multi_connector_spec_decode_test.sh
#
# Environment variables:
#   MODEL_NAME               - target model to test (default: Qwen/Qwen3-0.6B)
#   PREFILL_GPU              - GPU index for the prefill instance (default: 0)
#   DECODE_GPU               - GPU index for the decode instance (default: 1)
#   NUM_SPEC_TOKENS          - speculative tokens per decode step (default: 3)
#   PROMPT_LOOKUP_MAX        - max ngram proposer window (default: 5)
#   PROMPT_LOOKUP_MIN        - min ngram proposer window (default: 3)
#   GPU_MEMORY_UTILIZATION   - GPU memory fraction (default: 0.6)
#   CPU_OFFLOAD_BYTES        - CPU offload buffer size in bytes (default: 1000000000)
#   MAX_MODEL_LEN            - max sequence length (default: 8192)
set -ex

# ── Model & spec decode config ──────────────────────────────────────────
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-3}"
PROMPT_LOOKUP_MAX="${PROMPT_LOOKUP_MAX:-5}"
PROMPT_LOOKUP_MIN="${PROMPT_LOOKUP_MIN:-3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

# ngram: no "model" field, driven purely by prompt lookup. Prefill barely uses
# spec decode (it doesn't generate), so keep its window at 1.
PREFILL_SPEC_CONFIG="{\"method\":\"ngram\",\"num_speculative_tokens\":1,\"prompt_lookup_max\":${PROMPT_LOOKUP_MAX},\"prompt_lookup_min\":${PROMPT_LOOKUP_MIN}}"
DECODE_SPEC_CONFIG="{\"method\":\"ngram\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS},\"prompt_lookup_max\":${PROMPT_LOOKUP_MAX},\"prompt_lookup_min\":${PROMPT_LOOKUP_MIN}}"

# ── Cluster layout ───────────────────────────────────────────────────────
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
CPU_OFFLOAD_BYTES="${CPU_OFFLOAD_BYTES:-1000000000}"
PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_GPU="${DECODE_GPU:-1}"
PREFILL_PORT="${PREFILL_PORT:-8100}"
DECODE_PORT="${DECODE_PORT:-8200}"
PREFILL_SIDE_CHANNEL_PORT="${PREFILL_SIDE_CHANNEL_PORT:-5559}"
DECODE_SIDE_CHANNEL_PORT="${DECODE_SIDE_CHANNEL_PORT:-5659}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
NIXL_SIDE_CHANNEL_HOST="${NIXL_SIDE_CHANNEL_HOST:-$SERVER_HOST}"
PROXY_PORT=8192

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GIT_ROOT="${GIT_ROOT:-$(cd -- "${SCRIPT_DIR}/../../../.." && pwd -P)}"

# ── KV transfer config: MultiConnector[Nixl + Offloading], kv_both ───────
# Prefill and decode both wrap NixlConnector + OffloadingConnector.
KV_CONFIG="{
  \"kv_connector\":\"MultiConnector\",
  \"kv_role\":\"kv_both\",
  \"kv_connector_extra_config\":{
    \"connectors\":[
      {\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\"},
      {\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",
       \"kv_connector_extra_config\":{\"cpu_bytes_to_use\":${CPU_OFFLOAD_BYTES}}}
    ]
  }
}"
KV_CONFIG=$(echo "$KV_CONFIG" | tr -d '[:space:]')

# ── Helpers ──────────────────────────────────────────────────────────────
cleanup_instances() {
  echo ""
  echo "Cleaning up..."
  jobs -pr | xargs -r kill 2>/dev/null || true
  sleep 1
  jobs -pr | xargs -r kill -9 2>/dev/null || true
  pkill -9 -f "vllm serve.*${MODEL_NAME}" 2>/dev/null || true
  pkill -9 -f "toy_proxy_server.*${PROXY_PORT}" 2>/dev/null || true
  sleep 1
  echo "Cleanup done."
}
trap cleanup_instances EXIT
trap 'echo " Interrupted."; exit 130' INT TERM

wait_for_server() {
  local port=$1 server_pid=$2 server_name=$3
  local endpoint=${4:-/v1/completions} deadline=${5:-600} elapsed=0
  echo "Waiting for ${server_name} on port ${port}..."
  while [ "$elapsed" -lt "$deadline" ]; do
    if ! ps -p "$server_pid" > /dev/null 2>&1; then
      local status=0; wait "$server_pid" || status=$?
      echo "FAIL: ${server_name} pid ${server_pid} exited with ${status} before port ${port} was ready"
      exit 1
    fi
    if curl -s "http://${SERVER_HOST}:${port}${endpoint}" > /dev/null 2>&1; then
      echo "${server_name} on port ${port} ready"; return 0
    fi
    sleep 2; elapsed=$((elapsed + 2))
  done
  echo "FAIL: ${server_name} on port ${port} did not start within ${deadline}s"; exit 1
}

wait_for_nixl_side_channel() {
  local host=$1 port=$2 server_pid=$3 server_name=$4 deadline=120 elapsed=0
  echo "Waiting for ${server_name} NIXL side channel on ${host}:${port}..."
  while [ $elapsed -lt $deadline ]; do
    if ! ps -p "$server_pid" > /dev/null 2>&1; then
      local status=0; wait "$server_pid" || status=$?
      echo "FAIL: ${server_name} pid ${server_pid} exited with ${status} before side channel ${host}:${port} was ready"
      exit 1
    fi
    if python3 "${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/nixl_side_channel_probe.py" \
      --host "$host" --port "$port" --timeout-ms 1000 > /dev/null 2>&1; then
      echo "${server_name} NIXL side channel ${host}:${port} ready"; return 0
    fi
    sleep 2; elapsed=$((elapsed + 2))
  done
  echo "FAIL: ${server_name} NIXL side channel ${host}:${port} did not start within ${deadline}s"; exit 1
}

# ── Launch ───────────────────────────────────────────────────────────────
echo "================================================================"
echo "MultiConnector(Nixl+Offloading) PD + ngram spec decode -- gsm8k"
echo "Model:            ${MODEL_NAME}"
echo "Spec tokens:      prefill=1 decode=${NUM_SPEC_TOKENS} (lookup ${PROMPT_LOOKUP_MIN}-${PROMPT_LOOKUP_MAX})"
echo "KV config:        ${KV_CONFIG}"
echo "Prefill GPU/port: ${PREFILL_GPU}/${PREFILL_PORT}   Decode GPU/port: ${DECODE_GPU}/${DECODE_PORT}"
echo "================================================================"

echo "Starting prefill instance..."
env CUDA_VISIBLE_DEVICES="$PREFILL_GPU" \
  VLLM_KV_CACHE_LAYOUT='HND' \
  UCX_NET_DEVICES=all \
  VLLM_NIXL_SIDE_CHANNEL_HOST="$NIXL_SIDE_CHANNEL_HOST" \
  VLLM_NIXL_SIDE_CHANNEL_PORT="$PREFILL_SIDE_CHANNEL_PORT" \
  vllm serve "$MODEL_NAME" \
    --port "$PREFILL_PORT" \
    --enforce-eager \
    --max-model-len "$MAX_MODEL_LEN" \
    --block-size "$BLOCK_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --tensor-parallel-size 1 \
    --kv-transfer-config "$KV_CONFIG" \
    --speculative-config "$PREFILL_SPEC_CONFIG" &
PREFILL_PID=$!
wait_for_server "$PREFILL_PORT" "$PREFILL_PID" "prefill"
wait_for_nixl_side_channel "$NIXL_SIDE_CHANNEL_HOST" "$PREFILL_SIDE_CHANNEL_PORT" "$PREFILL_PID" "prefill"

echo "Starting decode instance..."
env CUDA_VISIBLE_DEVICES="$DECODE_GPU" \
  VLLM_KV_CACHE_LAYOUT='HND' \
  UCX_NET_DEVICES=all \
  VLLM_NIXL_SIDE_CHANNEL_HOST="$NIXL_SIDE_CHANNEL_HOST" \
  VLLM_NIXL_SIDE_CHANNEL_PORT="$DECODE_SIDE_CHANNEL_PORT" \
  vllm serve "$MODEL_NAME" \
    --port "$DECODE_PORT" \
    --enforce-eager \
    --max-model-len "$MAX_MODEL_LEN" \
    --block-size "$BLOCK_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --tensor-parallel-size 1 \
    --kv-transfer-config "$KV_CONFIG" \
    --speculative-config "$DECODE_SPEC_CONFIG" &
DECODE_PID=$!
wait_for_server "$DECODE_PORT" "$DECODE_PID" "decode"
wait_for_nixl_side_channel "$NIXL_SIDE_CHANNEL_HOST" "$DECODE_SIDE_CHANNEL_PORT" "$DECODE_PID" "decode"

echo "Starting proxy on port ${PROXY_PORT}..."
python3 "${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" \
  --port "$PROXY_PORT" \
  --prefiller-hosts "$SERVER_HOST" \
  --prefiller-ports "$PREFILL_PORT" \
  --decoder-hosts "$SERVER_HOST" \
  --decoder-ports "$DECODE_PORT" &
PROXY_PID=$!
wait_for_server "$PROXY_PORT" "$PROXY_PID" "proxy" "/healthcheck" 60

# ── Correctness: gsm8k accuracy stays within RTOL of the baseline ────────
echo "Running gsm8k accuracy test through the combined PD pipeline..."
TEST_MODEL=$MODEL_NAME python3 -m pytest -s -x \
  "${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/test_accuracy.py"

# ── Liveness: speculative decoding is still accepting drafts (accept len > 1.3) ─
echo "Asserting speculative decoding stayed engaged on the decode server..."
SERVER_HOST=$SERVER_HOST DECODE_PORT=$DECODE_PORT python3 -m pytest -s -x \
  "${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/test_spec_decode_metrics.py"

echo ""
echo "=== MultiConnector(Nixl+Offloading) + ngram spec decode test passed ==="
