#!/bin/bash
set -euo pipefail

declare -a PIDS=()

###############################################################################
# Configuration -- override via env before running
###############################################################################
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
LOG_PATH="${LOG_PATH:-./logs}"
mkdir -p $LOG_PATH

ENCODE_PORT="${ENCODE_PORT:-19534}"
PREFILL_DECODE_PORT="${PREFILL_DECODE_PORT:-19535}"
PROXY_PORT="${PROXY_PORT:-10001}"

GPU_E="${GPU_E:-6}"
GPU_PD="${GPU_PD:-7}"

TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-12000}"   # wait_for_server timeout
NUM_PROMPTS="${NUM_PROMPTS:-100}"             # number of prompts to send in benchmark

MOONCAKE_MASTER_PORT=50051
MOONCAKE_METADATA_PORT=8080
MOONCAKE_MASTER_IP="localhost"                              # producer
MOONCAKE_STORE_INSTANCE_IP="localhost"                      # consumer
MOONCAKE_GLOBAL_SEGMENT_SIZE=$((1 * 1073741824))            # Rcm: 30 GB
MOONCAKE_LOCAL_BUFFER_SIZE=$((1 * 1073741824))              # Rcm: 1 GB
MOONCAKE_REPLICA_NUM=1
MOONCAKE_FAST_TRANSFER=true
MOONCAKE_FAST_TRANSFER_BUFFER_SIZE=$((1 * 1073741824))      # Rcm: 30 GB

###############################################################################
# Helpers
###############################################################################
START_TIME=$(date +"%Y%m%d_%H%M%S")
ENC_LOG=$LOG_PATH/encoder_${START_TIME}.log
PD_LOG=$LOG_PATH/pd_${START_TIME}.log
PROXY_LOG=$LOG_PATH/proxy_${START_TIME}.log
MOONCAKE_MASTER_LOG="$LOG_PATH/mooncake_master_$START_TIME.log"

wait_for_server() {
    local port=$1
    timeout "$TIMEOUT_SECONDS" bash -c "
        until curl -s localhost:$port/v1/chat/completions > /dev/null; do
            sleep 1
        done" && return 0 || return 1
}

# Cleanup function
cleanup() {
    echo "Stopping everything…"
    trap - INT TERM USR1   # prevent re-entrancy
    
    # Kill all tracked PIDs
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing process $pid"
            kill "$pid" 2>/dev/null
        fi
    done
    
    # Wait a moment for graceful shutdown
    wait "${PIDS[@]}" 2>/dev/null || true
    
    # Force kill any remaining processes
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Force killing process $pid"
            kill -9 "$pid" 2>/dev/null
        fi
    done

    echo "Force killing mooncake processes"
    pkill -f "mooncake_master"

    echo "All processes stopped."
    exit 0
}

trap cleanup INT
trap cleanup USR1
trap cleanup TERM

###############################################################################
# Initialize Mooncake
# Read more about Mooncake config at 
# https://kvcache-ai.github.io/Mooncake/deployment/mooncake-store-deployment-guide.html
###############################################################################
mooncake_master \
  --rpc_port $MOONCAKE_MASTER_PORT \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 \
  --http_metadata_server_port=$MOONCAKE_METADATA_PORT \
  --rpc_thread_num 8 \
  --default_kv_lease_ttl 5000 \
  --eviction_ratio 0.05 \
  --eviction_high_watermark_ratio 0.9 \
  >"$MOONCAKE_MASTER_LOG" 2>&1 &

export MC_MS_AUTO_DISC=0

###############################################################################
# Encoder worker
###############################################################################
CUDA_VISIBLE_DEVICES="$GPU_E" vllm serve "$MODEL" \
    --gpu-memory-utilization 0.8 \
    --port "$ENCODE_PORT" \
    --enforce-eager \
    --enable-request-id-headers \
    --no-enable-prefix-caching \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 16 \
    --ec-transfer-config "{
        \"ec_connector\": \"ECMooncakeStorageConnector\",
        \"ec_role\": \"ec_producer\",
        \"ec_connector_extra_config\": {
            \"local_hostname\": \"$MOONCAKE_MASTER_IP\",
            \"metadata_server\": \"http://localhost:$MOONCAKE_METADATA_PORT/metadata\",
            \"global_segment_size\": $MOONCAKE_GLOBAL_SEGMENT_SIZE,
            \"local_buffer_size\": $MOONCAKE_LOCAL_BUFFER_SIZE,
            \"protocol\": \"tcp\",
            \"device_name\": \"\",
            \"master_server_address\": \"localhost:$MOONCAKE_MASTER_PORT\",
            \"replica_num\": $MOONCAKE_REPLICA_NUM,
            \"fast_transfer\": $MOONCAKE_FAST_TRANSFER,
            \"fast_transfer_buffer_size\": $MOONCAKE_FAST_TRANSFER_BUFFER_SIZE
        }
    }" \
    >"${ENC_LOG}" 2>&1 &

PIDS+=($!)

###############################################################################
# Prefill+Decode worker
###############################################################################
CUDA_VISIBLE_DEVICES="$GPU_PD" vllm serve "$MODEL" \
    --gpu-memory-utilization 0.8 \
    --port "$PREFILL_DECODE_PORT" \
    --enforce-eager \
    --enable-request-id-headers \
    --max-num-seqs 16 \
    --ec-transfer-config "{
        \"ec_connector\": \"ECMooncakeStorageConnector\",
        \"ec_role\": \"ec_consumer\",
        \"ec_connector_extra_config\": {
            \"local_hostname\": \"$MOONCAKE_STORE_INSTANCE_IP\",
            \"metadata_server\": \"http://localhost:$MOONCAKE_METADATA_PORT/metadata\",
            \"global_segment_size\": 0,
            \"local_buffer_size\": $MOONCAKE_LOCAL_BUFFER_SIZE,
            \"protocol\": \"tcp\",
            \"device_name\": \"\",
            \"master_server_address\": \"localhost:$MOONCAKE_MASTER_PORT\",
            \"replica_num\": $MOONCAKE_REPLICA_NUM,
            \"fast_transfer\": $MOONCAKE_FAST_TRANSFER,
            \"fast_transfer_buffer_size\": $MOONCAKE_FAST_TRANSFER_BUFFER_SIZE
        }
    }" \
    >"${PD_LOG}" 2>&1 &

PIDS+=($!)

# Wait for workers
wait_for_server $ENCODE_PORT
wait_for_server $PREFILL_DECODE_PORT

###############################################################################
# Proxy
###############################################################################
python ../disagg_epd_proxy.py \
    --host "0.0.0.0" \
    --port "$PROXY_PORT" \
    --encode-servers-urls "http://localhost:$ENCODE_PORT" \
    --prefill-servers-urls "disable" \
    --decode-servers-urls "http://localhost:$PREFILL_DECODE_PORT" \
    >"${PROXY_LOG}" 2>&1 &

PIDS+=($!)

wait_for_server $PROXY_PORT
echo "All services are up!"

###############################################################################
# Benchmark
###############################################################################
vllm bench serve \
    --model $MODEL \
    --dataset-name random-mm \
    --num-prompts 100 \
    --random-input-len 150 \
    --random-output-len 100 \
    --random-range-ratio 0.0 \
    --random-mm-base-items-per-request 1 \
    --random-mm-num-mm-items-range-ratio 0 \
    --random-mm-limit-mm-per-prompt '{"image":2,"video":0}' \
    --random-mm-bucket-config '{(700, 728, 1): 1.0}' \
    --ignore-eos \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --port $PROXY_PORT

# cleanup
echo "cleanup..."
cleanup