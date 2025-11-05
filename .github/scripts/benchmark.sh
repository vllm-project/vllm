#!/bin/bash
set -euo pipefail

# Usage function
usage() {
    echo "Usage: $0 <model_name> <output_path>"
    exit 1
}

# Check arguments
if [ $# -lt 2 ]; then
    usage
fi

model_name="${1:-deepseekr1_ptpc_fp8}"
output_path="${2:-benchmark_results.json}"

model_path=$(jq -r ".${model_name}.path" .github/scripts/models_datas.json)
baseline=$(jq -r ".${model_name}.baseline" .github/scripts/models_datas.json)
baseline_strict_match_value=$(echo $baseline | jq -r ".[0].value")
baseline_flexible_extract_value=$(echo $baseline | jq -r ".[1].value")
echo "Model name: $model_name"
echo "Model path: $model_path"
echo "Output path: $output_path"
echo "Baseline strict match value: $baseline_strict_match_value"
echo "Baseline flexible extract value: $baseline_flexible_extract_value"

# Launch vLLM server
echo
echo "========== LAUNCHING vLLM SERVER =============="
./.github/scripts/launch_models.sh $model_name $model_path &

vllm_pid=$!

echo
echo "========== WAITING FOR SERVER TO BE READY ========"
max_retries=60
retry_interval=60
for ((i=1; i<=max_retries; i++)); do
    if curl -s http://localhost:8000/v1/completions -o /dev/null; then
        echo "vLLM server is up."
        break
    fi
    echo "Waiting for vLLM server to be ready... ($i/$max_retries)"
    sleep $retry_interval
done

if ! curl -s http://localhost:8000/v1/completions -o /dev/null; then
    echo "vLLM server did not start after $((max_retries * retry_interval)) seconds."
    kill $vllm_pid
    exit 1
fi

echo
echo "========== CURLING THE REQUEST ================"
curl -X POST "http://localhost:8000/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "The capital of China", "temperature": 0, "top_p": 1, "top_k": 0, "repetition_penalty": 1.0, "presence_penalty": 0, "frequency_penalty": 0, "stream": false, "ignore_eos": false, "n": 1, "seed": 123 
    }' || true

echo
echo "========== STARTING THE TEXT MODEL EVALUATION =========="
# Run lm_eval and capture its output
lm_eval \
    --model local-completions \
    --tasks gsm8k \
    --model_args model="$model_path",base_url=http://127.0.0.1:8000/v1/completions \
    --batch_size 100 \
    --output_path "models_performance_test/$output_path"

# Parse lm_eval output and compare metrics to baseline
result_file=$(ls -1t models_performance_test/*.json 2>/dev/null | head -n 1)
if [ -z "$result_file" ] || [ ! -f "$result_file" ]; then
    echo "ERROR: No results JSON file found in models_performance_test/"
    kill $vllm_pid
    exit 2
else
    echo "RESULT_FILE: $result_file"
fi

# Extract metrics from the output json using jq
strict_match_value=$(jq '.results.gsm8k["exact_match,strict-match"]' "$result_file")
flexible_extract_value=$(jq '.results.gsm8k["exact_match,flexible-extract"]' "$result_file")


{
    echo
    echo "========== RESULTS COMPARISON =============="
    echo "Strict Match:    $strict_match_value (baseline: $baseline_strict_match_value)"
    echo "Flexible Match:  $flexible_extract_value (baseline: $baseline_flexible_extract_value)"

    # Calculate delta with baseline
    delta_strict=$(awk -v current="$strict_match_value" -v base="$baseline_strict_match_value" 'BEGIN { d=current-base; printf "%+.6f", d }')
    delta_flexible=$(awk -v current="$flexible_extract_value" -v base="$baseline_flexible_extract_value" 'BEGIN { d=current-base; printf "%+.6f", d }')

    echo "Delta Strict:    $delta_strict"
    echo "Delta Flexible:  $delta_flexible"

    # If either delta_strict or delta_flexible is greater than 0.05, then fail/exit
    awk -v d1="$delta_strict" -v d2="$delta_flexible" '
        BEGIN {
            if (d1 < -0.03 || d1 > 0.03 || d2 < -0.03 || d2 > 0.03) {
                print "vLLM BENCHMARK FAILED: the delta of strict match or flexible match exceeds 0.03";
                exit 1;
            }
            print "vLLM BENCHMARK PASSED: the delta of strict match or flexible match is within 0.03";
        }'
} | tee comparison-$model_name.log

exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo
    echo "========== vLLM BENCHMARK COMPLETED SUCCESSFULLY =========="
else
    echo
    echo "========== vLLM BENCHMARK FAILED WITH EXIT CODE $exit_code =========="
    exit $exit_code
fi
