#!/usr/bin/env bash
set -euo pipefail

wait_for_server() {
    local port=$1
    timeout 12000 bash -c '
        until curl -s "http://localhost:'"$port"'/v1/chat/completions" > /dev/null; do
            sleep 1
        done
    ' && return 0 || return 1
}

MODEL="/path/to/Qwen2.5-VL-3B-Instruct"

LOG_PATH=${LOG_PATH:-./logs}
mkdir -p "$LOG_PATH"

ENCODE_PORT=19534
PREFILL_DECODE_PORT=19535
PROXY_PORT=10001

GPU_E=1
GPU_PD=2

START_TIME=$(date +"%Y%m%d_%H%M%S")
ENC_LOG="$LOG_PATH/encoder_${START_TIME}.log"
PD_LOG="$LOG_PATH/pd_${START_TIME}.log"
PROXY_LOG="$LOG_PATH/proxy_${START_TIME}.log"
PID_FILE="./pid.txt"

SHARED_STORAGE_PATH="/tmp/your/path"

###############################################################################
# Encoder worker
###############################################################################
CUDA_VISIBLE_DEVICES="$GPU_E" vllm serve "$MODEL" \
    --gpu-memory-utilization 0.7 \
    --port "$ENCODE_PORT" \
    --enable-request-id-headers \
    --max-num-seqs 128 \
    --enforce-eager \
    --ec-transfer-config '{
        "ec_connector": "ECSharedStorageConnector",
        "ec_role": "ec_producer",
        "ec_connector_extra_config": {
            "shared_storage_path": "'"$SHARED_STORAGE_PATH"'"
        }
    }' \
    >"$ENC_LOG" 2>&1 &

echo $! >> "$PID_FILE"

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

echo $! >> "$PID_FILE"

# Wait until both workers are ready
wait_for_server "$ENCODE_PORT"
wait_for_server "$PREFILL_DECODE_PORT"

###############################################################################
# Proxy
###############################################################################
python /path/to/your/vllm/draft/proxy.py \
    --host "0.0.0.0" \
    --port "$PROXY_PORT" \
    --encode-servers-urls "http://localhost:$ENCODE_PORT" \
    --prefill-decode-servers-urls "http://localhost:$PREFILL_DECODE_PORT" \
    >"$PROXY_LOG" 2>&1 &

echo $! >> "$PID_FILE"

wait_for_server "$PROXY_PORT"
echo "All services are up!"