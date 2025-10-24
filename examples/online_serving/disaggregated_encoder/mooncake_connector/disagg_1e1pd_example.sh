#!/bin/bash
set -euo pipefail

declare -a PIDS=()

###############################################################################
# Configuration -- override via env before running
###############################################################################
MODEL="${MODEL:-/models/Qwen2.5-VL-3B-Instruct}"
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
MOONCAKE_REPLICA_NUM=1
MOONCAKE_FAST_TRANSFER=true
MOONCAKE_FAST_TRANSFER_BUFFER_SIZE=3 # GB

SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

###############################################################################
# Helpers
###############################################################################
START_TIME=$(date +"%Y%m%d_%H%M%S")
ENC_LOG=$LOG_PATH/encoder_${START_TIME}.log
PD_LOG=$LOG_PATH/pd_${START_TIME}.log
PROXY_LOG=$LOG_PATH/proxy_${START_TIME}.log
MOONCAKE_MASTER_LOG="$LOG_PATH/mooncake_master_$START_TIME.log"
MOONCAKE_METADATA_LOG="$LOG_PATH/mooncake_metadata_$START_TIME.log"

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
    sleep 2
    
    # Force kill any remaining processes
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Force killing process $pid"
            kill -9 "$pid" 2>/dev/null
        fi
    done
    
    # Kill the entire process group as backup
    kill -- -$$ 2>/dev/null
    
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
  --default_kv_lease_ttl 0 \
  --eviction_ratio 0.05 \
  --eviction_high_watermark_ratio 0.9 \
  >"$MOONCAKE_MASTER_LOG" 2>&1 &
PIDS+=($!)

export MC_MS_AUTO_DISC=0

sed -e "s/\${MOONCAKE_MASTER_PORT}/$MOONCAKE_MASTER_PORT/"\
    -e "s/\${MOONCAKE_METADATA_PORT}/$MOONCAKE_METADATA_PORT/"\
    -e "s/\${MOONCAKE_REPLICA_NUM}/$MOONCAKE_REPLICA_NUM/"\
    -e "s/\${MOONCAKE_FAST_TRANSFER}/$MOONCAKE_FAST_TRANSFER/"\
    -e "s/\${MOONCAKE_FAST_TRANSFER_BUFFER_SIZE}/$MOONCAKE_FAST_TRANSFER_BUFFER_SIZE/"\
    mooncake_config/producer_template.json > producer.json
sed -e "s/\${MOONCAKE_MASTER_PORT}/$MOONCAKE_MASTER_PORT/"\
    -e "s/\${MOONCAKE_METADATA_PORT}/$MOONCAKE_METADATA_PORT/"\
    -e "s/\${MOONCAKE_REPLICA_NUM}/$MOONCAKE_REPLICA_NUM/"\
    -e "s/\${MOONCAKE_FAST_TRANSFER}/$MOONCAKE_FAST_TRANSFER/"\
    -e "s/\${MOONCAKE_FAST_TRANSFER_BUFFER_SIZE}/$MOONCAKE_FAST_TRANSFER_BUFFER_SIZE/"\
    mooncake_config/consumer_template.json > consumer.json

###############################################################################
# Encoder worker
###############################################################################
CUDA_VISIBLE_DEVICES="$GPU_E" vllm serve "$MODEL" \
    --gpu-memory-utilization 0.7 \
    --port "$ENCODE_PORT" \
    --enforce-eager \
    --enable-request-id-headers \
    --no-enable-prefix-caching \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 128 \
    --ec-transfer-config '{
        "ec_connector":"ECMooncakeStorageConnector",
        "ec_role":"ec_producer",
        "ec_connector_extra_config": {
            "ec_mooncake_config_file_path":"'${SCRIPT_DIR}'/producer.json",
            "ec_max_num_scheduled_tokens": "1000000000000000000"
        }
    }' \
    >"${ENC_LOG}" 2>&1 &

PIDS+=($!)

###############################################################################
# Prefill+Decode worker
###############################################################################
CUDA_VISIBLE_DEVICES="$GPU_PD" VLLM_NIXL_SIDE_CHANNEL_PORT=6000 vllm serve "$MODEL" \
    --gpu-memory-utilization 0.7 \
    --port "$PREFILL_DECODE_PORT" \
    --enforce-eager \
    --enable-request-id-headers \
    --max-num-seqs 128 \
    --ec-transfer-config '{
        "ec_connector":"ECMooncakeStorageConnector",
        "ec_role":"ec_consumer",
        "ec_connector_extra_config": {
            "ec_mooncake_config_file_path":"'${SCRIPT_DIR}'/consumer.json"
        }
    }' \
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
# vllm bench serve \
#   --model               $MODEL \
#   --backend             openai-chat \
#   --endpoint            /v1/chat/completions \
#   --dataset-name        hf \
#   --dataset-path        lmarena-ai/VisionArena-Chat \
#   --seed                0 \
#   --num-prompts         $NUM_PROMPTS \
#   --port                $PROXY_PORT

# vllm bench serve \
#     --model $MODEL \
#     --dataset-name random-mm \
#     --num-prompts 100 \
#     --random-input-len 100 \
#     --random-output-len 128 \
#     --random-range-ratio 0.0 \
#     --random-mm-base-items-per-request 1 \
#     --random-mm-num-mm-items-range-ratio 0 \
#     --random-mm-limit-mm-per-prompt '{"image":2,"video":0}' \
#     --random-mm-bucket-config '{(700, 728, 1): 1.0}' \
#     --request-rate 32 \
#     --ignore-eos \
#     --backend openai-chat \
#     --endpoint /v1/chat/completions \
#     --seed 60 \
#     --port $PROXY_PORT

# PIDS+=($!)

cd /workspace/mistral-evals/
python -m eval.run eval_vllm \
    --model_name $MODEL \
    --url http://localhost:$PROXY_PORT \
    --output_dir ./outputs \
    --eval_name "chartqa"
###############################################################################

# cleanup
echo "cleanup..."
cleanup