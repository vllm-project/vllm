#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Full-stack EPD validation with ECMooncakeConnector (Mooncake TransferEngine):
#   1) Single-GPU baseline (multimodal) -> saves baseline JSON
#   2) 1 Encoder + 1 PD with Mooncake EC -> compare outputs to baseline
#
# Usage (from repo root):
#   ./tests/v1/ec_connector/integration/run_epd_mooncake_ec_full_pipeline.sh
#
# Env:
#   MODEL                    HF model id (default: Qwen/Qwen2.5-VL-3B-Instruct)
#   GPU_SINGLE / GPU_E / GPU_PD   GPU ids (defaults 0 / 0 / 1)
#   ENDPOINT_PORT, ENCODE_PORT, PREFILL_DECODE_PORT
#   EC_MOONCAKE_RESERVATION_HOST  consumer control host (default 127.0.0.1)
#   MOONCAKE_EC_PROTOCOL        tcp | rdma (default tcp)
#   USE_MM_PROMPTS              1 (default) or 0 for text-only quick sanity
#   TIMEOUT_SECONDS             wait_for_server timeout (default 1200)
#   SKIP_BASELINE               set to 1 to reuse existing BASELINE_FILE
#   CONCURRENCY / REPEAT        concurrent requests / rounds (defaults 3 / 2)
#   MAX_MODEL_LEN               context length (default 16384)

set -euo pipefail

GIT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
cd "$GIT_ROOT" || exit 1
export PYTHONPATH="${GIT_ROOT}:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-${GIT_ROOT}/.venv/bin/python}"

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
USE_MM_PROMPTS="${USE_MM_PROMPTS:-1}"
TEST_ARGS=(--concurrency "${CONCURRENCY:-3}" --repeat "${REPEAT:-2}")
if [[ "$USE_MM_PROMPTS" == "1" ]]; then
  TEST_ARGS+=(--use_mm_prompts --mm_smoke_test)
fi
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

GPU_SINGLE="${GPU_SINGLE:-0}"
GPU_E="${GPU_E:-0}"
GPU_PD="${GPU_PD:-1}"

ENCODE_PORT="${ENCODE_PORT:-19534}"
PREFILL_DECODE_PORT="${PREFILL_DECODE_PORT:-19537}"
ENDPOINT_PORT="${ENDPOINT_PORT:-10002}"
BASELINE_PORT="${BASELINE_PORT:-10003}"

EC_MOONCAKE_RESERVATION_HOST="${EC_MOONCAKE_RESERVATION_HOST:-127.0.0.1}"
EC_MOONCAKE_RESERVATION_PORT="${EC_MOONCAKE_RESERVATION_PORT:-19019}"
MOONCAKE_EC_PROTOCOL="${MOONCAKE_EC_PROTOCOL:-tcp}"
export EC_MOONCAKE_RESERVATION_HOST
export EC_MOONCAKE_RESERVATION_PORT
export MOONCAKE_EC_PROTOCOL
if [[ "$MOONCAKE_EC_PROTOCOL" == "tcp" ]]; then
  # TransferEngine may otherwise auto-select RDMA on hosts with an HCA.
  export MC_FORCE_TCP=1
else
  unset MC_FORCE_TCP
fi

LOG_PATH="${LOG_PATH:-/tmp}"
BASELINE_FILE="${BASELINE_FILE:-/tmp/vllm_epd_mooncake_baseline.txt}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1200}"
PIDS=()

mkdir -p "$LOG_PATH"

VLLM_SERVE=("$PYTHON_BIN" -m vllm.entrypoints.cli.main serve)

ENC_EC_JSON=$("$PYTHON_BIN" <<PY
import json, os
print(json.dumps({
    "ec_connector": "ECMooncakeConnector",
    "ec_role": "ec_producer",
    "ec_connector_extra_config": {
        "mooncake_protocol": os.environ.get("MOONCAKE_EC_PROTOCOL", "tcp"),
    },
}, separators=(",", ":")))
PY
)

PD_EC_JSON=$("$PYTHON_BIN" <<PY
import json, os
print(json.dumps({
    "ec_connector": "ECMooncakeConnector",
    "ec_role": "ec_consumer",
    "ec_ip": os.environ["EC_MOONCAKE_RESERVATION_HOST"],
    "ec_port": int(os.environ.get("EC_MOONCAKE_RESERVATION_PORT", "19019")),
    "ec_connector_extra_config": {
        "mooncake_protocol": os.environ.get("MOONCAKE_EC_PROTOCOL", "tcp"),
    },
}, separators=(",", ":")))
PY
)

EC_MOONCAKE_RESERVATION_ADDR=$("$PYTHON_BIN" <<'PY'
import os

from vllm.utils.network_utils import make_zmq_path

print(make_zmq_path(
    "tcp", os.environ["EC_MOONCAKE_RESERVATION_HOST"],
    int(os.environ["EC_MOONCAKE_RESERVATION_PORT"]),
))
PY
)

wait_for_server() {
  local port=$1
  local pid=$2
  local deadline=$((SECONDS + TIMEOUT_SECONDS))
  while ((SECONDS < deadline)); do
    kill -0 "$pid" 2>/dev/null || return 1
    if curl --max-time 2 -fsS "http://localhost:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

cleanup_instances() {
  if ((${#PIDS[@]} == 0)); then
    return
  fi
  echo "Cleaning up tracked vLLM / proxy processes..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  for _ in {1..10}; do
    local alive=0
    for pid in "${PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        alive=1
      fi
    done
    ((alive == 0)) && break
    sleep 1
  done
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
  done
  PIDS=()
}

finish() {
  local status=$?
  cleanup_instances
  if ((status != 0)); then
    for log in "${LOG_PATH}"/mooncake_epd_*.log; do
      [[ -f "$log" ]] || continue
      echo "=== $log (last 100 lines) ==="
      tail -100 "$log"
    done
  fi
  exit "$status"
}

trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_baseline() {
  echo "================================"
  echo "BASELINE (single vLLM, MM if enabled)"
  echo "================================"
  cleanup_instances
  local PORT=$BASELINE_PORT
  echo "Starting baseline on GPU $GPU_SINGLE port $PORT"
  CUDA_VISIBLE_DEVICES="$GPU_SINGLE" "${VLLM_SERVE[@]}" "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization 0.75 \
    --max-num-seqs 32 \
    --max-model-len "$MAX_MODEL_LEN" \
    --no-enable-prefix-caching \
    --limit-mm-per-prompt '{"image":2,"video":0}' \
    --allowed-local-media-path "${GIT_ROOT}/tests/v1/ec_connector/integration" \
    >"${LOG_PATH}/mooncake_epd_baseline.log" 2>&1 &
  local BASELINE_PID=$!
  PIDS+=("$BASELINE_PID")
  echo "Waiting for baseline..."
  wait_for_server "$PORT" "$BASELINE_PID" || { echo "Baseline failed to start"; return 1; }
  curl -s "http://127.0.0.1:${PORT}/v1/models" | head -c 200 || true
  echo ""
  "$PYTHON_BIN" "${GIT_ROOT}/tests/v1/ec_connector/integration/test_epd_correctness.py" \
    --service_url "http://localhost:$PORT" \
    --model_name "$MODEL" \
    --mode baseline \
    --baseline_file "$BASELINE_FILE" \
    "${TEST_ARGS[@]}"
  cleanup_instances
}

run_epd_mooncake() {
  echo "================================"
  echo "EPD 1E + 1PD with ECMooncakeConnector"
  echo "Reservation port on consumer: $EC_MOONCAKE_RESERVATION_PORT"
  echo "Mooncake protocol: $MOONCAKE_EC_PROTOCOL"
  echo "================================"
  cleanup_instances

  echo "Starting ENCODER on GPU $GPU_E port $ENCODE_PORT"
  CUDA_VISIBLE_DEVICES="$GPU_E" "${VLLM_SERVE[@]}" "$MODEL" \
    --port "$ENCODE_PORT" \
    --gpu-memory-utilization 0.35 \
    --mm-tensor-ipc torch_shm \
    --enable-request-id-headers \
    --no-enable-prefix-caching \
    --max-num-batched-tokens "$MAX_MODEL_LEN" \
    --max-num-seqs 32 \
    --max-model-len "$MAX_MODEL_LEN" \
    --limit-mm-per-prompt '{"image":2,"video":0}' \
    --allowed-local-media-path "${GIT_ROOT}/tests/v1/ec_connector/integration" \
    --ec-transfer-config "$ENC_EC_JSON" \
    >"${LOG_PATH}/mooncake_epd_encoder.log" 2>&1 &
  local ENCODER_PID=$!
  PIDS+=("$ENCODER_PID")

  echo "Starting PD on GPU $GPU_PD port $PREFILL_DECODE_PORT"
  CUDA_VISIBLE_DEVICES="$GPU_PD" "${VLLM_SERVE[@]}" "$MODEL" \
    --port "$PREFILL_DECODE_PORT" \
    --gpu-memory-utilization 0.75 \
    --mm-tensor-ipc torch_shm \
    --enable-mm-embeds \
    --enable-request-id-headers \
    --max-num-seqs 32 \
    --max-model-len "$MAX_MODEL_LEN" \
    --no-enable-prefix-caching \
    --limit-mm-per-prompt '{"image":2,"video":0}' \
    --allowed-local-media-path "${GIT_ROOT}/tests/v1/ec_connector/integration" \
    --ec-transfer-config "$PD_EC_JSON" \
    >"${LOG_PATH}/mooncake_epd_pd.log" 2>&1 &
  local PD_PID=$!
  PIDS+=("$PD_PID")

  echo "Waiting for encoder..."
  wait_for_server "$ENCODE_PORT" "$ENCODER_PID" || { echo "Encoder failed to start"; return 1; }
  echo "Waiting for PD..."
  wait_for_server "$PREFILL_DECODE_PORT" "$PD_PID" || { echo "PD failed to start"; return 1; }

  echo "Starting EPD proxy on $ENDPOINT_PORT"
  "$PYTHON_BIN" "${GIT_ROOT}/examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py" \
    --host "0.0.0.0" \
    --port "$ENDPOINT_PORT" \
    --encode-servers-urls "http://localhost:$ENCODE_PORT" \
    --prefill-servers-urls "disable" \
    --decode-servers-urls "http://localhost:$PREFILL_DECODE_PORT" \
    --ec-consumer-zmq-addrs \
      "$EC_MOONCAKE_RESERVATION_ADDR" \
    >"${LOG_PATH}/mooncake_epd_proxy.log" 2>&1 &
  local PROXY_PID=$!
  PIDS+=("$PROXY_PID")

  echo "Waiting for proxy..."
  wait_for_server "$ENDPOINT_PORT" "$PROXY_PID" || { echo "Proxy failed to start"; return 1; }
  curl -s "http://127.0.0.1:${ENDPOINT_PORT}/health" || true
  echo ""

  "$PYTHON_BIN" "${GIT_ROOT}/tests/v1/ec_connector/integration/test_epd_correctness.py" \
    --service_url "http://localhost:$ENDPOINT_PORT" \
    --model_name "$MODEL" \
    --mode disagg \
    --baseline_file "$BASELINE_FILE" \
    "${TEST_ARGS[@]}"

  cleanup_instances
}

echo "================================"
echo "1E + 1PD ECMooncake end-to-end correctness"
echo "MODEL=$MODEL"
echo "================================"

if [[ "${SKIP_BASELINE:-0}" != "1" ]]; then
  run_baseline
else
  echo "SKIP_BASELINE=1 -> using existing $BASELINE_FILE"
  [[ -f "$BASELINE_FILE" ]] || { echo "Missing baseline file"; exit 1; }
fi

run_epd_mooncake

echo "================================"
echo "PASS: Mooncake EC EPD matches baseline"
echo "Logs: ${LOG_PATH}/mooncake_epd_*.log"
echo "================================"
