#!/bin/bash
in_len=1024
out_len=1024

bs=192
request_rate=16
model="/data/models/DeepSeek-R1-static/"
log_name="[DP]online-gaudi3-nprompt${num_prompts}_rrate${request_rate}_bs${bs}_i${in_len}_o${out_len}"
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

total_len=$((in_len + out_len))

in_len_aligned=$((in_len + 127 / 128 * 128))
total_len_aligend=$((total_len + 127 / 128 * 128))
VLLM_DECODE_BLOCK_BUCKET_MIN=$((in_len_aligned * bs / 128))
VLLM_DECODE_BLOCK_BUCKET_MIN=$(((VLLM_DECODE_BLOCK_BUCKET_MIN + 127) / 128 * 128))
VLLM_DECODE_BLOCK_BUCKET_MAX=$((total_len_aligend * bs / 128))
VLLM_DECODE_BLOCK_BUCKET_MAX=$(((VLLM_DECODE_BLOCK_BUCKET_MAX + 127) / 128 * 128))

# start all servers
max_cid=$((num_hpu - 1))
for i in `seq 0 ${max_cid}`; do
    VLLM_SKIP_WARMUP=false \
    VLLM_PROMPT_BS_BUCKET_MIN=1 \
    VLLM_PROMPT_BS_BUCKET_MAX=8 \
    VLLM_PROMPT_SEQ_BUCKET_MIN=${in_len_aligned} \
    VLLM_PROMPT_SEQ_BUCKET_MAX=${in_len_aligned} \
    VLLM_DECODE_BS_BUCKET_MIN=${bs} \
    VLLM_DECODE_BS_BUCKET_MAX=${bs} \
    VLLM_DECODE_BLOCK_BUCKET_MIN=${VLLM_DECODE_BLOCK_BUCKET_MIN} \
    VLLM_DECODE_BLOCK_BUCKET_MAX=${VLLM_DECODE_BLOCK_BUCKET_MAX} \
    VLLM_USE_FP8_MATMUL=true \
    VLLM_DELAYED_SAMPLING=true \
    VLLM_GRAPH_RESERVED_MEM=0.05 \
    VLLM_DP_RANK=${i} VLLM_DP_SIZE=$num_hpu VLLM_DP_MASTER_IP=127.0.0.1 VLLM_DP_MASTER_PORT=25940 \
    python -m vllm.entrypoints.openai.api_server \
        --model $MODEL_PATH  \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization 0.85 \
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
python3 examples/online_serving/disagg_examples/disagg_proxy_demo.py \
    --model $MODEL_PATH \
    --decode 127.0.0.1:8801 127.0.0.1:8802 127.0.0.1:8803 127.0.0.1:8804 127.0.0.1:8805 127.0.0.1:8806 127.0.0.1:8807 127.0.0.1:8808\
    --port 8123 2>&1 | tee pd_benchmark_logs/${log_name}/disagg_proxy_demo.log &

sleep 10

# START BENCHMARK

max_concurrency=$((num_hpu*bs))
num_prompts=$((max_concurrency))
start_time=$(date +%s)
echo "Start to benchmark"
python benchmarks/benchmark_serving.py --backend vllm --model ${model} --dataset-name sonnet --dataset-path benchmarks/sonnet.txt --request-rate ${request_rate}  --ignore-eos --num-prompts ${num_prompts} --max-concurrency ${max_concurrency} --port 8123 --sonnet-input-len ${in_len} --sonnet-output-len ${out_len} --sonnet-prefix-len 100 2>&1 | tee pd_benchmark_logs/${log_name}/run.log
end_time=$(date +%s)
echo "Time elapsed: $((end_time - start_time))s"

sleep 10

pkill -9 python
