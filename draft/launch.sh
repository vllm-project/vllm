#!/bin/bash


wait_for_server() {
    local port=$1
    timeout 12000 bash -c "
        until curl -s localhost:$port/v1/chat/completions > /dev/null; do
            sleep 1
        done" && return 0 || return 1
}

MODEL="/home/n00909098/Qwen2.5-VL-3B-Instruct" 
LOG_PATH=$LOG_PATH
ENCODE_PORT=19534
ENCODE_RANK=0
PREFILL_DECODE_PORT=19535
PREFILL_DECODE_RANK=1
PROXY_PORT=10001
GPU_E="1"
GPU_PD="2"
START_TIME=$(date +"%Y%m%d_%H%M%S")
ENC_LOG=./logs/encoder.log
PD_LOG=./logs/pd.log
PROXY_LOG=./logs/proxy.log
PID_FILE="./pid.txt"

CUDA_VISIBLE_DEVICES="$GPU_E" vllm serve "$MODEL" \
    --gpu-memory-utilization 0.7 \
    --port "$ENCODE_PORT" \
    --enable-request-id-headers \
    --max-num-seqs 128 \
    --enforce-eager \
    --ec-transfer-config \
        '{"ec_connector":"ECSharedStorageConnector","ec_role":"ec_producer"}' \
    >"${ENC_LOG}" 2>&1 &

echo $! >> "$PID_FILE"

CUDA_VISIBLE_DEVICES="$GPU_PD" vllm serve "$MODEL" \
    --gpu-memory-utilization 0.7 \
    --port "$PREFILL_DECODE_PORT" \
    --enable-request-id-headers \
    --max-num-seqs 128 \
    --enforce-eager \
    --ec-transfer-config \
        '{"ec_connector":"ECSharedStorageConnector","ec_role":"ec_consumer"}' \
    >"${PD_LOG}" 2>&1 &
    
echo $! >> "$PID_FILE"

wait_for_server $ENCODE_PORT
wait_for_server $PREFILL_DECODE_PORT

python /home/n00909098/EPD/refactor/vllm/draft/proxy.py \
    --host "0.0.0.0" \
    --port "$PROXY_PORT" \
    --encode-servers-urls "http://localhost:$ENCODE_PORT" \
    --prefill-decode-servers-urls "http://localhost:$PREFILL_DECODE_PORT" \
    >"${PROXY_LOG}" 2>&1 &

echo $! >> "$PID_FILE"

wait_for_server $PROXY_PORT