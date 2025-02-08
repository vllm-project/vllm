#! /bin/bash

model_path=/models/Qwen2-7B-Instruct
host=127.0.0.1
port=30001
data_path=/models/ShareGPT_V3_unfiltered_cleaned_split.json
num_prompts=1000
request_rate=inf

model_name=$( echo $model_path | awk -F/ '{print $NF}' )
echo "Benchmarking ${model_path} for vllm server '${host}:${port}' with ${num_prompts} prompts from ${data_path} and request_rate=${request_rate}"
log_name=benchmark_serving_${model_name}_sharegpt_rate-${request_rate}_prompts-${num_prompts}_$(date +%F-%H-%M-%S)

python ../../benchmarks/benchmark_serving.py \
    --backend vllm \
    --model $model_path \
    --trust-remote-code \
    --host $host \
    --port $port \
    --dataset-name sharegpt \
    --dataset-path $data_path \
    --num-prompts $num_prompts \
    --request-rate $request_rate \
    --seed 0 \
    --save-result \
    --result-filename "${log_name}".json \
    --ignore-eos \
    |& tee "${log_name}".log 2>&1
