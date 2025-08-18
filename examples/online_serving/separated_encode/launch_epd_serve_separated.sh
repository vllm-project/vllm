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
GPU_E="6"
GPU_PD="7"

START_TIME=$(date +"%Y%m%d_%H%M%S")

redis-server &

CUDA_VISIBLE_DEVICES="$GPU_E" vllm serve "$MODEL" \
    --gpu-memory-utilization 0.9 \
    --port "$ENCODE_PORT" \
    --enable-request-id-headers \
    --max-num-seqs 128 \
    --instance-type "encode" \
    --connector-workers-num 8 \
    --epd-rank 0 &

wait_for_server $ENCODE_PORT

CUDA_VISIBLE_DEVICES="$GPU_PD" vllm serve "$MODEL" \
    --gpu-memory-utilization 0.9 \
    --port "$PREFILL_DECODE_PORT" \
    --enable-request-id-headers \
    --max-num-seqs 128 \
    --instance-type "prefill+decode" \
    --connector-workers-num 8 \
    --epd-rank 1 &

wait_for_server $PREFILL_DECODE_PORT

python examples/online_serving/separated_encode/proxy/proxy1e1pd_aiohttp.py \
    --host "0.0.0.0" \
    --port "$PROXY_PORT" \
    --encode-server-url "http://localhost:$ENCODE_PORT" \
    --prefill-decode-server-url "http://localhost:$PREFILL_DECODE_PORT" \
    --e-rank 0 \
    --pd-rank 1 &

wait_for_server $PROXY_PORT