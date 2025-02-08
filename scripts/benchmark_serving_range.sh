#! /bin/bash

model_path=/models/Qwen2-72B-Instruct
host=127.0.0.1
port=30001
input=1000
output=500
ratio=0.8
num_prompts=1000
request_rate=inf

model_name=$( echo $model_path | awk -F/ '{print $NF}' )
echo "Benchmarking ${model_path} for vllm server '${host}:${port}' with input_max=${input}, output_max=${output}, ratio=${ratio}, num_prompts=${num_prompts}, request_rate=${request_rate}"

log_name=benchmark_serving_${model_name}_random_in-${input}_out-${output}_ratio-${ratio}_rate-${request_rate}_prompts-${num_prompts}_$(date +%F-%H-%M-%S)

python ../../benchmarks/benchmark_serving.py \
    --backend vllm \
    --model $model_path \
    --trust-remote-code \
    --host $host \
    --port $port \
    --dataset-name random \
    --random-input-len $input \
    --random-output-len $output \
    --random-range-ratio $ratio \
    --num-prompts $num_prompts \
    --request-rate $request_rate \
    --seed 0 \
    --save-result \
    --result-filename "${log_name}".json \
    --ignore-eos \
    |& tee "${log_name}".log 2>&1
