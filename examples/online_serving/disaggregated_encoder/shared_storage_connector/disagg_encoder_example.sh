#!/usr/bin/env bash
# shellcheck disable=SC2086,SC2155
set -euo pipefail

###############################################################################
# Configuration -- override via env before running
###############################################################################
MODEL="${MODEL:-/workspace/vllm/Qwen2.5-VL-3B-Instruct/}"

LOG_PATH="${LOG_PATH:-./logs}"
ENCODE_PORT="${ENCODE_PORT:-19534}"
PREFILL_DECODE_PORT="${PREFILL_DECODE_PORT:-19535}"
PROXY_PORT="${PROXY_PORT:-10001}"

GPU_E="${GPU_E:-6}"
GPU_PD="${GPU_PD:-7}"

SHARED_STORAGE_PATH="${SHARED_STORAGE_PATH:-/tmp/}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-12000}"   # wait_for_server timeout

###############################################################################
# Dependencies check ##########################################################
###############################################################################
ensure_python_library_installed() {
  local lib=$1
  echo -n "[check] Python package '$lib' … "
  if python3 - <<EOF >/dev/null 2>&1
import importlib, sys
sys.exit(0) if importlib.util.find_spec("${lib}") else sys.exit(1)
EOF
  then
      echo "OK"
  else
      echo "NOT FOUND"
      echo "Please install it, e.g.:  pip install $lib"
      exit 1
  fi
}

ensure_python_library_installed vllm
ensure_python_library_installed pandas
ensure_python_library_installed datasets

###############################################################################
# Helpers
###############################################################################
mkdir -p "$LOG_PATH"
START_TIME=$(date +"%Y%m%d_%H%M%S")
ENC_LOG="$LOG_PATH/encoder_$START_TIME.log"
PD_LOG="$LOG_PATH/pd_$START_TIME.log"
PROXY_LOG="$LOG_PATH/proxy_$START_TIME.log"

wait_for_server() {
  local port=$1
  timeout "$TIMEOUT_SECONDS" bash -c '
    until curl -s "http://localhost:'"$port"'/v1/chat/completions" > /dev/null; do
      sleep 1
    done
  '
}

# keep PIDs in memory
declare -a PIDS=()
CLEANED=0
cleanup() {
  (( CLEANED )) && return        # run only once
  CLEANED=1

  echo "Cleaning up…"
  # iterate backwards (proxy → PD → encoder)
  for (( idx=${#PIDS[@]}-1 ; idx>=0 ; idx-- )); do
    PID=${PIDS[idx]}
    if kill -0 "$PID" 2>/dev/null; then
      echo "  • Killing $PID"
      kill "$PID"
      sleep 2
      kill -9 "$PID" 2>/dev/null || true
    fi
  done
  echo "Done."
}

trap cleanup EXIT INT TERM ERR

###############################################################################
# Encoder worker
###############################################################################
CUDA_VISIBLE_DEVICES="$GPU_E" vllm serve "$MODEL" \
  --gpu-memory-utilization 0.0 \
  --port "$ENCODE_PORT" \
  --enable-request-id-headers \
  --no-enable-prefix-caching \
  --max-num-seqs 128 \
  --enforce-eager \
  --ec-transfer-config '{
      "ec_connector": "ECSharedStorageConnector",
      "ec_role": "ec_producer",
      "ec_connector_extra_config": {
          "shared_storage_path": "'"$SHARED_STORAGE_PATH"'",
          "ec_max_num_scheduled_tokens": "4096"
      }
  }' \
  >"$ENC_LOG" 2>&1 &

PIDS+=($!)

###############################################################################
# Prefill / decode worker
###############################################################################
CUDA_VISIBLE_DEVICES="$GPU_PD" vllm serve "$MODEL" \
  --gpu-memory-utilization 0.7 \
  --port "$PREFILL_DECODE_PORT" \
  --enable-request-id-headers \
  --max-num-seqs 128 \
  --enforce-eager \
  --ec-transfer-config '{
      "ec_connector": "ECSharedStorageConnector",
      "ec_role": "ec_consumer",
      "ec_connector_extra_config": {
          "shared_storage_path": "'"$SHARED_STORAGE_PATH"'"
      }
  }' \
  >"$PD_LOG" 2>&1 &

PIDS+=($!)

# Wait for workers
wait_for_server "$ENCODE_PORT"
wait_for_server "$PREFILL_DECODE_PORT"

###############################################################################
# Proxy
###############################################################################
python disagg_encoder_proxy.py \
  --host "127.0.0.1" \
  --port "$PROXY_PORT" \
  --encode-servers-urls "http://localhost:$ENCODE_PORT" \
  --prefill-decode-servers-urls "http://localhost:$PREFILL_DECODE_PORT" \
  >"$PROXY_LOG" 2>&1 &

PIDS+=($!)

wait_for_server "$PROXY_PORT"
echo "All services are up!"

###############################################################################
# Benchmark
cd ../../../../benchmarks
python benchmark_serving.py \
  --backend           openai-chat \
  --model             $MODEL \
  --dataset-name      hf \
  --dataset-path      /workspace/lmarena-ai/VisionArena-Chat \
  --seed              40 \
  --endpoint          /v1/chat/completions \
  --num-prompts       $1 \
  --port              $PROXY_PORT \
  --host              127.0.0.1 \
  --request-rate      $2
###############################################################################
cleanup