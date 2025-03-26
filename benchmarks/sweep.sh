MODEL=meta-llama/Llama-3.1-8B-Instruct
REQUEST_RATES=(1 10 15 20)
INPUT_LEN=1000
OUTPUT_LEN=100
TOTAL_SECONDS=120

for REQUEST_RATE in "${REQUEST_RATES[@]}";
do
    NUM_PROMPTS=$(($TOTAL_SECONDS * $REQUEST_RATE))
    
    echo ""
    echo "===== RUNNING $MODEL FOR $NUM_PROMPTS PROMPTS WITH $REQUEST_RATE QPS ====="
    echo ""

    python3 vllm/benchmarks/benchmark_serving.py \
        --model $MODEL \
        --dataset-name random \
        --random-input-len $INPUT_LEN \
        --random-output-len $OUTPUT_LEN \
        --request-rate $REQUEST_RATE \
        --num-prompts $NUM_PROMPTS \
        --seed $REQUEST_RATE \
        --ignore-eos \
        --result-filename "$FRAMEWORK-results.json" \
        --save-result

done
