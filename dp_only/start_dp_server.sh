#!/bin/bash

export VLLM_MLA_DISABLE_REQUANTIZATION=1
export VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0
export PT_HPU_ENABLE_LAZY_COLLECTIVES="true"
export PT_HPU_WEIGHT_SHARING=0

export VLLM_EP_SIZE=8
export PT_HPU_LAZY_MODE=1
export VLLM_SKIP_WARMUP=True
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export MAX_MODEL_LEN=8192
export MODEL_PATH=/data/models/DeepSeek-R1-static/
export VLLM_USE_V1=0


for i in `seq 0 7`; do
    VLLM_USE_FP8_MATMUL=true \
    VLLM_DELAYED_SAMPLING=true \
    VLLM_GRAPH_RESERVED_MEM=0.05 \
    VLLM_DP_RANK=${i} VLLM_DP_SIZE=8 VLLM_DP_MASTER_IP=127.0.0.1 VLLM_DP_MASTER_PORT=25940 \
    python -m vllm.entrypoints.openai.api_server \
        --model $MODEL_PATH  \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization 0.85 \
        --max-num-seqs 256 \
        --kv_cache_dtype "fp8_inc" \
        --trust-remote-code \
        --port 880$((i+1)) 2>&1 | tee dp_server_logs/dp_server${i}.log &
    sleep 5
done

until [[ $ready == true ]]; do
    n=$((n+1))
    i=7
    if grep -q "Started server process" dp_server_logs/dp_server${i}.log; then
        break
    fi
    sleep 5s
done
sleep 10s

for i in `seq 0 7`; do
    echo "Checking server $i"
    grep "Started server process" dp_server_logs/dp_server${i}.log
done

echo "ALL servers are ready"

python3 examples/online_serving/disagg_examples/disagg_proxy_demo.py \
    --model $MODEL_PATH \
    --prefill 127.0.0.1:8100 \
    --decode 127.0.0.1:8801 127.0.0.1:8802 127.0.0.1:8803 127.0.0.1:8804 127.0.0.1:8805 127.0.0.1:8806 127.0.0.1:8807 127.0.0.1:8808\
    --port 8123 2>&1 | tee dp_server_logs/disagg_proxy_demo.log

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
