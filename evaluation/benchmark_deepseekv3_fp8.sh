#!/bin/bash

# model_path="/data/pretrained-models/deepseek-ai/DeepSeek-V3.2-Exp"
model=deepseek-ai/DeepSeek-V3

timestamp=$(date +%Y%m%d_%H%M%S)
log_file="./ds3-log/benchmark_base_results_${timestamp}.log"
csv_file="./ds3-log/benchmark_base_summary_${timestamp}.csv"
# log_file="./ds3-log/benchmark_fuse_shared_exp_results_${timestamp}.log"
# csv_file="./ds3-log/benchmark_fuse_shared_exp_summary_${timestamp}.csv"

echo "Benchmark started at $(date)" | tee -a ${log_file}
echo "========================================" | tee -a ${log_file}

echo "Input_Tokens,Output_Tokens,Max_Concurrency,Num_Prompts,Mean_TTFT_ms,Mean_TPOT_ms,Req_per_sec,QPM,Total_Token_Throughput" > ${csv_file}

input_tokens_array=(1024 8192)
max_concurrency_array=(4 8 16 32 64)
output_tokens=1024

for input_tokens in "${input_tokens_array[@]}"; do
    for max_concurrency in "${max_concurrency_array[@]}"; do
        num_prompts=$((max_concurrency * 4))
        
        echo "" | tee -a ${log_file}
        echo "========================================" | tee -a ${log_file}
        echo "Running benchmark:" | tee -a ${log_file}
        echo "  Input tokens: ${input_tokens}" | tee -a ${log_file}
        echo "  Output tokens: ${output_tokens}" | tee -a ${log_file}
        echo "  Max concurrency: ${max_concurrency}" | tee -a ${log_file}
        echo "  Num prompts: ${num_prompts}" | tee -a ${log_file}
        echo "  Started at: $(date)" | tee -a ${log_file}
        echo "========================================" | tee -a ${log_file}
        
        temp_output=$(mktemp)
        vllm bench serve \
            --host localhost \
            --port 8000 \
            --model ${model} \
            --dataset-name random \
            --random-input-len ${input_tokens} \
            --random-output-len ${output_tokens} \
            --max-concurrency ${max_concurrency} \
            --num-prompts ${num_prompts} \
            --percentile-metrics ttft,tpot,itl,e2el \
            --ignore-eos \
            --seed 123 2>&1 | tee -a ${log_file} | tee ${temp_output}
        
        mean_ttft=$(grep "Mean TTFT (ms):" ${temp_output} | tail -1 | awk '{print $4}')
        mean_tpot=$(grep "Mean TPOT (ms):" ${temp_output} | tail -1 | awk '{print $4}')
        req_per_sec=$(grep "Request throughput (req/s):" ${temp_output} | tail -1 | awk '{print $4}')
        total_token_throughput=$(grep "Total Token throughput (tok/s):" ${temp_output} | tail -1 | awk '{print $5}')
        
        if [ ! -z "$req_per_sec" ]; then
            qpm=$(awk "BEGIN {print $req_per_sec * 60}")
        else
            qpm="N/A"
        fi
        
        echo "${input_tokens},${output_tokens},${max_concurrency},${num_prompts},${mean_ttft},${mean_tpot},${req_per_sec},${qpm},${total_token_throughput}" >> ${csv_file}
        
        rm -f ${temp_output}
        
        echo "" | tee -a ${log_file}
        echo "Completed at: $(date)" | tee -a ${log_file}
        echo "========================================" | tee -a ${log_file}
        
        sleep 5
    done
done

echo "" | tee -a ${log_file}
echo "All benchmarks completed at $(date)" | tee -a ${log_file}
echo "Results saved to: ${log_file}" | tee -a ${log_file}
echo "Summary CSV saved to: ${csv_file}" | tee -a ${log_file}
echo ""
echo "CSV Summary:"
cat ${csv_file}
