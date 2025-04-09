#!/bin/bash

model_path=/data/hf_models/DeepSeek-R1-G2-static
ip_addr=127.0.0.1
port=8688


test_benchmark_serving_range() {
    local_input=$1
    local_output=$2
    local_max_concurrency=$3
    local_num_prompts=$4
    local_len_ratio=$5

    echo "running benchmark serving range test, input len: $local_input, output len: $local_output, len ratio: $local_len_ratio, concurrency: $local_max_concurrency"

    log_name=benchmark_serving_DeekSeek-R1_cardnumber_16_datatype_bfloat16_random_batchsize_${local_max_concurrency}_in_${local_input}_out_${local_output}_ratio_${local_len_ratio}_rate_inf_prompts_${local_num_prompts}_$(TZ='Asia/Shanghai' date +%F-%H-%M-%S)
    python3 benchmark_serving.py --backend vllm --model $model_path --trust-remote-code --host $ip_addr --port $port \
    --dataset-name random --random-input-len $local_input --random-output-len $local_output --random-range-ratio $local_len_ratio --max_concurrency $local_max_concurrency\
    --num-prompts $local_num_prompts --request-rate inf --seed 0 --ignore_eos \
    --save-result --result-filename ${log_name}.json 

}


test_benchmark_serving_range 1024 1024 1 3 1
test_benchmark_serving_range 1024 1024 32 96 1
