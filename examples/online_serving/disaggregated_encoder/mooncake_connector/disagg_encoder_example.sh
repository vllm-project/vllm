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

MOONCAKE_ZEROCOPY=$1
MOONCAKE_MASTER_PORT=50051
MOONCAKE_METADATA_PORT=8080
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

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
MOONCAKE_MASTER_LOG="$LOG_PATH/mooncake_master_$START_TIME.log"
MOONCAKE_METADATA_LOG="$LOG_PATH/mooncake_metadata_$START_TIME.log"
BENCHMARK_LOG="$SCRIPT_DIR/logs/benchmark_$START_TIME.log"

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

  kill -9 $(lsof -t -i :$MOONCAKE_MASTER_PORT)
  echo "Done."
}

trap cleanup EXIT INT TERM ERR

###############################################################################
# Initialize Mooncake
###############################################################################
mooncake_master \
  --rpc_port $MOONCAKE_MASTER_PORT \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 \
  --http_metadata_server_port=$MOONCAKE_METADATA_PORT \
  --rpc_thread_num 8\
  --default_kv_lease_ttl 0\
  >"$MOONCAKE_MASTER_LOG" 2>&1 &
PIDS+=($!)

export MC_MS_AUTO_DISC=0

sed -e "s/\${MOONCAKE_MASTER_PORT}/$MOONCAKE_MASTER_PORT/"\
    -e "s/\${MOONCAKE_METADATA_PORT}/$MOONCAKE_METADATA_PORT/"\
    -e "s/\${MOONCAKE_ZEROCOPY}/$MOONCAKE_ZEROCOPY/"\
    mooncake_config/producer_template.json > producer.json
sed -e "s/\${MOONCAKE_MASTER_PORT}/$MOONCAKE_MASTER_PORT/"\
    -e "s/\${MOONCAKE_METADATA_PORT}/$MOONCAKE_METADATA_PORT/"\
    -e "s/\${MOONCAKE_ZEROCOPY}/$MOONCAKE_ZEROCOPY/"\
    mooncake_config/consumer_template.json > consumer.json

###############################################################################
# Encoder worker
###############################################################################
CUDA_VISIBLE_DEVICES="$GPU_E" vllm serve "$MODEL" \
  --gpu-memory-utilization 0.7 \
  --port "$ENCODE_PORT" \
  --enable-request-id-headers \
  --no-enable-prefix-caching \
  --max-num-seqs 128 \
  --enforce-eager \
  --ec-transfer-config '{
        "ec_connector":"ECMooncakeStorageConnector",
        "ec_role":"ec_producer",
        "ec_connector_extra_config": {
            "ec_mooncake_config_file_path":"'${SCRIPT_DIR}'/producer.json",
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
        "ec_connector":"ECMooncakeStorageConnector",
        "ec_role":"ec_consumer",
        "ec_connector_extra_config": {
            "ec_mooncake_config_file_path":"'${SCRIPT_DIR}'/consumer.json"
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
python $2 \
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
  --num-prompts       $3 \
  --port              $PROXY_PORT \
  --host              127.0.0.1 \
  --request-rate      $4
###############################################################################
cleanup