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
#   GPU_SINGLE / GPU_E / GPU_PD   GPU ids (defaults 0 / 1 / 2)
#   ENDPOINT_PORT, ENCODE_PORT, PREFILL_DECODE_PORT
#   MOONCAKE_EC_PROTOCOL        rdma | tcp (default rdma)
#   USE_MM_PROMPTS              1 (default) or 0 for text-only quick sanity
#   TIMEOUT_SECONDS             wait_for_server timeout (default 1200)
#   SKIP_BASELINE               set to 1 to reuse existing BASELINE_FILE

set -euo pipefail

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT" || exit 1
export PYTHONPATH="${GIT_ROOT}:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-${GIT_ROOT}/.venv/bin/python}"

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
USE_MM_PROMPTS="${USE_MM_PROMPTS:-1}"
MM_FLAG=""
if [[ "$USE_MM_PROMPTS" == "1" ]]; then
  MM_FLAG="--use_mm_prompts"
fi

GPU_SINGLE="${GPU_SINGLE:-0}"
GPU_E="${GPU_E:-1}"
GPU_PD="${GPU_PD:-2}"

ENCODE_PORT="${ENCODE_PORT:-19534}"
PREFILL_DECODE_PORT="${PREFILL_DECODE_PORT:-19537}"
ENDPOINT_PORT="${ENDPOINT_PORT:-10002}"
BASELINE_PORT="${BASELINE_PORT:-10003}"

EC_MOONCAKE_RESERVATION_PORT="${EC_MOONCAKE_RESERVATION_PORT:-19019}"
MOONCAKE_EC_PROTOCOL="${MOONCAKE_EC_PROTOCOL:-rdma}"
export EC_MOONCAKE_RESERVATION_PORT
export MOONCAKE_EC_PROTOCOL
# Mooncake cannot register CUDA memory through the peer-memory path on hosts
# whose kernel lacks the OFED peer-memory API; every transfer then fails at
# setup with -202. Opting out selects the path that works there and is a no-op
# where GPUDirect is available.
export WITH_NVIDIA_PEERMEM="${WITH_NVIDIA_PEERMEM:-0}"

LOG_PATH="${LOG_PATH:-/tmp}"
BASELINE_FILE="${BASELINE_FILE:-/tmp/vllm_epd_mooncake_baseline.txt}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1200}"

mkdir -p "$LOG_PATH"

VLLM_SERVE=("$PYTHON_BIN" -m vllm.entrypoints.cli.main serve)

ENC_EC_JSON=$("$PYTHON_BIN" <<PY
import json, os
print(json.dumps({
    "ec_connector": "ECMooncakeConnector",
    "ec_role": "ec_producer",
    "ec_connector_extra_config": {
        "mooncake_protocol": os.environ.get("MOONCAKE_EC_PROTOCOL", "rdma"),
    },
}, separators=(",", ":")))
PY
)

PD_EC_JSON=$("$PYTHON_BIN" <<PY
import json, os
print(json.dumps({
    "ec_connector": "ECMooncakeConnector",
    "ec_role": "ec_consumer",
    "ec_connector_extra_config": {
        "mooncake_protocol": os.environ.get("MOONCAKE_EC_PROTOCOL", "rdma"),
        "reservation_zmq_port": int(
            os.environ.get("EC_MOONCAKE_RESERVATION_PORT", "19019")
        ),
    },
}, separators=(",", ":")))
PY
)

wait_for_server() {
  local port=$1
  timeout "$TIMEOUT_SECONDS" bash -c "
        until curl -s -o /dev/null -w '' localhost:${port}/v1/chat/completions; do
            sleep 2
        done" && return 0 || return 1
}

cleanup_instances() {
  echo "Cleaning up vLLM / proxy processes..."
  pkill -f "vllm serve" 2>/dev/null || true
  pkill -f "vllm.entrypoints.cli.main serve" 2>/dev/null || true
  pkill -f "disagg_epd_proxy.py" 2>/dev/null || true
  sleep 2
}

trap 'cleanup_instances; kill $(jobs -pr) 2>/dev/null || true' EXIT INT TERM

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
    --allowed-local-media-path "${GIT_ROOT}/tests/v1/ec_connector/integration" \
    >"${LOG_PATH}/mooncake_epd_baseline.log" 2>&1 &
  local BASELINE_PID=$!
  echo "Waiting for baseline..."
  wait_for_server "$PORT" || { echo "Baseline failed to start; tail log:"; tail -80 "${LOG_PATH}/mooncake_epd_baseline.log"; return 1; }
  curl -s "http://127.0.0.1:${PORT}/v1/models" | head -c 200 || true
  echo ""
  "$PYTHON_BIN" "${GIT_ROOT}/tests/v1/ec_connector/integration/test_epd_correctness.py" \
    --service_url "http://localhost:$PORT" \
    --model_name "$MODEL" \
    --mode baseline \
    --baseline_file "$BASELINE_FILE" \
    $MM_FLAG
  kill "$BASELINE_PID" 2>/dev/null || true
  sleep 2
  cleanup_instances
}

run_epd_mooncake() {
  echo "================================"
  echo "EPD 1E + 1PD with ECMooncakeConnector"
  echo "Reservation port on consumer: $EC_MOONCAKE_RESERVATION_PORT"
  echo "Mooncake protocol: $MOONCAKE_EC_PROTOCOL"
  echo "================================"
  cleanup_instances

  declare -a PIDS=()

  echo "Starting ENCODER on GPU $GPU_E port $ENCODE_PORT"
  CUDA_VISIBLE_DEVICES="$GPU_E" "${VLLM_SERVE[@]}" "$MODEL" \
    --port "$ENCODE_PORT" \
    --gpu-memory-utilization 0.35 \
    --mm-tensor-ipc torch_shm \
    --enable-request-id-headers \
    --no-enable-prefix-caching \
    --max-num-batched-tokens 114688 \
    --max-num-seqs 32 \
    --allowed-local-media-path "${GIT_ROOT}/tests/v1/ec_connector/integration" \
    --ec-transfer-config "$ENC_EC_JSON" \
    >"${LOG_PATH}/mooncake_epd_encoder.log" 2>&1 &
  PIDS+=($!)

  echo "Starting PD on GPU $GPU_PD port $PREFILL_DECODE_PORT"
  CUDA_VISIBLE_DEVICES="$GPU_PD" "${VLLM_SERVE[@]}" "$MODEL" \
    --port "$PREFILL_DECODE_PORT" \
    --gpu-memory-utilization 0.75 \
    --mm-tensor-ipc torch_shm \
    --enable-mm-embeds \
    --enable-request-id-headers \
    --max-num-seqs 32 \
    --allowed-local-media-path "${GIT_ROOT}/tests/v1/ec_connector/integration" \
    --ec-transfer-config "$PD_EC_JSON" \
    >"${LOG_PATH}/mooncake_epd_pd.log" 2>&1 &
  PIDS+=($!)

  echo "Waiting for encoder..."
  wait_for_server "$ENCODE_PORT" || { echo "Encoder log:"; tail -100 "${LOG_PATH}/mooncake_epd_encoder.log"; return 1; }
  echo "Waiting for PD..."
  wait_for_server "$PREFILL_DECODE_PORT" || { echo "PD log:"; tail -100 "${LOG_PATH}/mooncake_epd_pd.log"; return 1; }

  echo "Starting EPD proxy on $ENDPOINT_PORT"
  "$PYTHON_BIN" "${GIT_ROOT}/examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py" \
    --host "0.0.0.0" \
    --port "$ENDPOINT_PORT" \
    --encode-servers-urls "http://localhost:$ENCODE_PORT" \
    --prefill-servers-urls "disable" \
    --decode-servers-urls "http://localhost:$PREFILL_DECODE_PORT" \
    --decode-ec-transfer-zmq-addrs \
      "tcp://localhost:$EC_MOONCAKE_RESERVATION_PORT" \
    >"${LOG_PATH}/mooncake_epd_proxy.log" 2>&1 &
  PIDS+=($!)

  echo "Waiting for proxy..."
  wait_for_server "$ENDPOINT_PORT" || { echo "Proxy log:"; tail -80 "${LOG_PATH}/mooncake_epd_proxy.log"; return 1; }
  curl -s "http://127.0.0.1:${ENDPOINT_PORT}/health" || true
  echo ""

  "$PYTHON_BIN" "${GIT_ROOT}/tests/v1/ec_connector/integration/test_epd_correctness.py" \
    --service_url "http://localhost:$ENDPOINT_PORT" \
    --model_name "$MODEL" \
    --mode disagg \
    --baseline_file "$BASELINE_FILE" \
    $MM_FLAG

  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  sleep 2
  cleanup_instances
}

echo "================================"
echo "EPD + ECMooncake full pipeline"
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
