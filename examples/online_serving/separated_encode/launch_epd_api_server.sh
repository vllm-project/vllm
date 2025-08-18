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
PROXY_PORT=10001
GPU="7"
START_TIME=$(date +"%Y%m%d_%H%M%S")

redis-server &

CUDA_VISIBLE_DEVICES="$GPU" python examples/online_serving/separated_encode/api_server/api_server_1e1pd.py \
    --port "$PROXY_PORT" \
    --e-rank 0 \
    --pd-rank 1 > "$LOG_PATH/proxy_$START_TIME.log" &

wait_for_server $PROXY_PORT