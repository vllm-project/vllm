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
PREFILL_DECODE_PORT_F=19535
PREFILL_DECODE_PORT_S=19536

ENCODE_RANK=0
PREFILL_DECODE_RANK_F=1
PREFILL_DECODE_RANK_S=2

GPU_E="3"
GPU_PD_F="4"
GPU_PD_S="5"

PROXY_PORT=10001

export REDIS_HOST="localhost"
export REDIS_PORT="6379"

START_TIME=$(date +"%Y%m%d_%H%M%S")

redis-server --bind "$REDIS_HOST" --port "$REDIS_PORT" &

CUDA_VISIBLE_DEVICES="$GPU_E" vllm serve "$MODEL" \
    --gpu-memory-utilization 0.9 \
    --port "$ENCODE_PORT" \
    --enable-request-id-headers \
    --max-num-seqs 128 \
    --instance-type "encode" \
    --connector-workers-num 8 \
    --epd-rank "$ENCODE_RANK" &

CUDA_VISIBLE_DEVICES="$GPU_PD_F" vllm serve "$MODEL" \
    --gpu-memory-utilization 0.9 \
    --port "$PREFILL_DECODE_PORT_F" \
    --enable-request-id-headers \
    --max-num-seqs 128 \
    --instance-type "prefill+decode" \
    --connector-workers-num 8 \
    --epd-rank "$PREFILL_DECODE_RANK_F" &

CUDA_VISIBLE_DEVICES="$GPU_PD_S" vllm serve "$MODEL" \
    --gpu-memory-utilization 0.9 \
    --port "$PREFILL_DECODE_PORT_S" \
    --enable-request-id-headers \
    --max-num-seqs 128 \
    --instance-type "prefill+decode" \
    --connector-workers-num 8 \
    --epd-rank "$PREFILL_DECODE_RANK_S" &

wait_for_server $ENCODE_PORT
wait_for_server $PREFILL_DECODE_PORT_F
wait_for_server $PREFILL_DECODE_PORT_S

python examples/online_serving/separated_encode/proxy/proxy_aiohttp.py \
    --host "0.0.0.0" \
    --port "$PROXY_PORT" \
    --encode-servers-urls "http://localhost:$ENCODE_PORT" \
    --prefill-decode-servers-urls "http://localhost:$PREFILL_DECODE_PORT_F,http://localhost:$PREFILL_DECODE_PORT_S" \
    --encode-servers-ranks "$ENCODE_RANK" \
    --prefill-decode-servers-ranks "$PREFILL_DECODE_RANK_F,$PREFILL_DECODE_RANK_S" &

wait_for_server $PROXY_PORT