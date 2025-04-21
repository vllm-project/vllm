#!/bin/bash
in_len=1024
out_len=1024

bs=192
request_rate=16
model="/data/models/DeepSeek-R1-static/"
log_name="[DP]online-gaudi3-nprompt${num_prompts}_rrate${request_rate}_bs${bs}_i${in_len}_o${out_len}-nowarmup-2"
num_hpu=8
mkdir -p pd_benchmark_logs/${log_name}

#!/bin/bash

export VLLM_MLA_DISABLE_REQUANTIZATION=1
export VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0
export PT_HPU_ENABLE_LAZY_COLLECTIVES="true"
export PT_HPU_WEIGHT_SHARING=0

export VLLM_EP_SIZE=${num_hpu}
export PT_HPU_LAZY_MODE=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export MAX_MODEL_LEN=8192
export MODEL_PATH=/data/models/DeepSeek-R1-static/
export VLLM_USE_V1=0

# start all servers
max_cid=$((num_hpu - 1))
for i in `seq 0 ${max_cid}`; do
    VLLM_SKIP_WARMUP=true \
    VLLM_USE_FP8_MATMUL=true \
    VLLM_DELAYED_SAMPLING=true \
    VLLM_GRAPH_RESERVED_MEM=0.05 \
    VLLM_DP_RANK=${i} VLLM_DP_SIZE=$num_hpu VLLM_DP_MASTER_IP=127.0.0.1 VLLM_DP_MASTER_PORT=25940 \
    python -m vllm.entrypoints.openai.api_server \
        --model $MODEL_PATH  \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization 0.85 \
        --disable-log-requests \
        --max-num-seqs ${bs} \
        --kv_cache_dtype "fp8_inc" \
        --trust-remote-code \
        --port 880$((i+1)) 2>&1 | tee pd_benchmark_logs/${log_name}/dp_server${i}.log &
    sleep 5
done

until [[ $ready == true ]]; do
    n=$((n+1))
    i=7
    if grep -q "Started server process" pd_benchmark_logs/${log_name}/dp_server${i}.log; then
        break
    fi
    sleep 5s
done
sleep 10s

for i in `seq 0 ${max_cid}`; do
    echo "Checking server $i"
    grep "Started server process" pd_benchmark_logs/${log_name}/dp_server${i}.log
done

echo "ALL servers are ready"

# start proxy router
python3 examples/online_serving/dp_proxy.py \
    --model $MODEL_PATH \
    --decode 127.0.0.1:8801 127.0.0.1:8802 127.0.0.1:8803 127.0.0.1:8804 127.0.0.1:8805 127.0.0.1:8806 127.0.0.1:8807 127.0.0.1:8808\
    --port 8123 2>&1 | tee pd_benchmark_logs/${log_name}/dp_proxy.log &

sleep 10

# START BENCHMARK

max_concurrency=$((num_hpu*bs))
num_prompts=$((max_concurrency))
start_time=$(date +%s)

echo "Start to warmup"
python benchmarks/benchmark_serving.py --backend vllm --model ${model} --dataset-name sonnet --dataset-path benchmarks/sonnet.txt --request-rate ${request_rate}  --ignore-eos --num-prompts ${num_prompts} --max_concurrency ${max_concurrency} --port 8123 --sonnet-input-len ${in_len} --sonnet-output-len ${out_len} --sonnet-prefix-len 100 2>&1 | tee pd_benchmark_logs/${log_name}/warmup.log
end_time=$(date +%s)
echo "Time elapsed: $((end_time - start_time))s"
sleep 10


start_time=$(date +%s)
echo "Start to benchmark"
python benchmarks/benchmark_serving.py --backend vllm --model ${model} --dataset-name sonnet --dataset-path benchmarks/sonnet.txt --request-rate ${request_rate}  --ignore-eos --num-prompts ${num_prompts} --max_concurrency ${max_concurrency} --port 8123 --sonnet-input-len ${in_len} --sonnet-output-len ${out_len} --sonnet-prefix-len 100 2>&1 | tee pd_benchmark_logs/${log_name}/run.log
end_time=$(date +%s)
echo "Time elapsed: $((end_time - start_time))s"

sleep 10

pkill -9 python
