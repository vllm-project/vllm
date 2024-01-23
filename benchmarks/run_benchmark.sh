#!/bin/bash

# Cause the script to exit if a single command fails
set -eo pipefail

# Get gpu usage
# nvidia-smi

start=`date`

log_path=`pwd`"/bench_results/"`date "+%Y%m%d_%H%M%S"`
model="/bigdata/shared/models/huggingface/starcoder/"
tp=1

if [ ! -d ${log_path} ]; then
    mkdir -p ${log_path}
fi


function bench_kernel_latency_deepseek() {
    echo "###################### vllm-benchmark_kernel_latency ######################"
    local batch_size=(1 4 16 64 256)
    local num_query_heads=(56)
    local num_kv_heads=(8)
    local head_size=(128)
    local version=("v1" "v2" "flash-attn")
    local dtype=("bfloat16" "half")
    log_file=${log_path}/paged_attention_kernel_deepseek.log

    for d in "${dtype[@]}"; do
        for bs in "${batch_size[@]}"; do
            for query_heads in "${num_query_heads[@]}"; do
                for kv_heads in "${num_kv_heads[@]}"; do
                    for h_size in "${head_size[@]}"; do
                        for v in "${version[@]}"; do
                            python3 benchmarks/kernels/benchmark_paged_attention.py \
                                --batch-size ${bs} \
                                --num-query-heads ${query_heads} \
                                --num-kv-heads ${kv_heads} \
                                --head-size ${h_size} \
                                --dtype ${d} \
                                --version ${v} >> ${log_file}
                        done
                    done
                done
            done
        done
    done
}

function bench_kernel_latency_qwen() {
    echo "###################### vllm-benchmark_kernel_latency ######################"
    local batch_size=(1 4 16 64 256)
    local num_query_heads=(64)
    local num_kv_heads=(64)
    local head_size=(128)
    local version=("v1" "v2" "flash-attn")
    local dtype=("bfloat16" "half")
    log_file=${log_path}/paged_attention_kernel_qwen.log

    for d in "${dtype[@]}"; do
        for bs in "${batch_size[@]}"; do
            for query_heads in "${num_query_heads[@]}"; do
                for kv_heads in "${num_kv_heads[@]}"; do
                    for h_size in "${head_size[@]}"; do
                        for v in "${version[@]}"; do
                            python3 benchmarks/kernels/benchmark_paged_attention.py \
                                --batch-size ${bs} \
                                --num-query-heads ${query_heads} \
                                --num-kv-heads ${kv_heads} \
                                --head-size ${h_size} \
                                --dtype ${d} \
                                --version ${v} >> ${log_file}
                        done
                    done
                done
            done
        done
    done
}

function bench_latency() {
    echo "###################### vllm-benchmark_latency ######################"
    local batch_size=(1 4 16 64 256)
    local in_out_lens=(512)
    local iters=10
    log_file=${log_path}/latency.log

    for tp in ${tp}; do
        for bs in "${batch_size[@]}"; do
            for lens in "${in_out_lens[@]}"; do
                # nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o nsight_report -f true --capture-range=cudaProfilerApi --cudabacktrace=true \
                python3 benchmarks/benchmark_latency.py \
                    --model ${model} \
                    --tensor-parallel-size ${tp} \
                    --input-len ${lens} \
                    --output-len ${lens} \
                    --batch-size ${bs} \
                    --num-iters ${iters} >> ${log_file}

                python3 benchmarks/benchmark_latency.py \
                    --model ${model} \
                    --tensor-parallel-size ${tp} \
                    --input-len ${lens} \
                    --output-len ${lens} \
                    --batch-size ${bs} \
                    --use-flash-attn \
                    --num-iters ${iters} >> ${log_file}
            done
        done
    done
}

function bench_throughput_offline() {
    echo "###################### vllm-benchmark_throughput_offline ######################"
    local backends=("vllm")
    local in_out_lens=(512)
    log_file=${log_path}/throughput_offline.log

    for backend in "${backends[@]}"; do
        for num_prompts in 1000; do
            for lens in "${in_out_lens[@]}"; do
                python3 benchmarks/benchmark_throughput.py \
                    --backend ${backend} \
                    --model ${model} \
                    --tensor-parallel-size ${tp} \
                    --input-len ${lens} \
                    --output-len ${lens} \
                    --num-prompts ${num_prompts} >> ${log_file}

                python3 benchmarks/benchmark_throughput.py \
                    --backend ${backend} \
                    --model ${model} \
                    --tensor-parallel-size ${tp} \
                    --input-len ${lens} \
                    --output-len ${lens} \
                    --use-flash-attn \
                    --num-prompts ${num_prompts} >> ${log_file}
            done
        done
    done
}

function main() {
    # bench_kernel_latency_deepseek
    # bench_kernel_latency_qwen
    # bench_latency
    bench_throughput_offline

    end=`date`
    echo 'Start:' ${start}
    echo '  End:' ${end}
}

main $@
