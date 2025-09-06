#!/bin/bash


wait_for_server() {
    local port=$1
    timeout 12000 bash -c "
        until curl -s localhost:$port/v1/chat/completions > /dev/null; do
            sleep 1
        done" && return 0 || return 1
}

MODEL="/workspace/helper/Qwen2.5-VL-3B-Instruct" 
LOG_PATH=$LOG_PATH
ENCODE_PORT=19534
PREFILL_DECODE_PORT=19535
PROXY_PORT=10001
GPU="5"
export REDIS_HOST="localhost"
export REDIS_PORT="6379"
START_TIME=$(date +"%Y%m%d_%H%M%S")

redis-server --bind "$REDIS_HOST" --port "$REDIS_PORT" &

CUDA_VISIBLE_DEVICES="$GPU" vllm serve "$MODEL" \
    --gpu-memory-utilization 0.2 \
    --port "$ENCODE_PORT" \
    --enable-request-id-headers \
    --max-num-seqs 32 \
    --instance-type "encode" \
    --connector-workers-num 8 \
    --epd-rank 0 &

wait_for_server $ENCODE_PORT

CUDA_VISIBLE_DEVICES="$GPU" vllm serve "$MODEL" \
    --gpu-memory-utilization 0.7 \
    --port "$PREFILL_DECODE_PORT" \
    --enable-request-id-headers \
    --max-num-seqs 128 \
    --instance-type "prefill+decode" \
    --connector-workers-num 8 \
    --epd-rank 1 &

wait_for_server $PREFILL_DECODE_PORT

python examples/online_serving/separated_encode/proxy/proxy_aiohttp.py \
    --host "0.0.0.0" \
    --port "$PROXY_PORT" \
    --encode-servers-urls "http://localhost:$ENCODE_PORT" \
    --prefill-decode-servers-urls "http://localhost:$PREFILL_DECODE_PORT" \
    --encode-servers-ranks "0" \
    --prefill-decode-servers-ranks "1" &

wait_for_server $PROXY_PORT