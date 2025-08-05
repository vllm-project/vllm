#!/bin/bash

#@VARS

# Wait for vLLM server to be ready
until curl -s http://localhost:8000${ENDPOINT} > /dev/null; do
    echo "Waiting for vLLM server to be ready..."
    sleep 15
done
echo "vLLM server is ready. Starting benchmark..."


SONNET_ARGS=""
if [[ "$DATASET_NAME" == "sonnet" ]]; then
    SONNET_ARGS="--sonnet-prefix-len $PREFIX_LEN --sonnet-input-len $INPUT_TOK --sonnet-output-len $OUTPUT_TOK"
fi

HF_ARGS=""
if [[ "$DATASET_NAME" == "hf" ]]; then
    HF_ARGS="--hf-split train"
fi

## Start benchmarking vLLM serving
python3 /workspace/vllm/benchmarks/benchmark_serving.py \
                --model $MODEL \
                --base-url http://localhost:8000 \
                --endpoint $ENDPOINT \
                --backend $BACKEND \
                --dataset-name $DATASET_NAME \
                --dataset-path $DATASET\
                $SONNET_ARGS \
                $HF_ARGS \
                --num-prompts $NUM_PROMPTS \
                --max-concurrency $CONCURRENT_REQ \
                --metric-percentiles 90 \
                --ignore-eos \
                --trust-remote-code \
2>&1 | tee -a logs/perftest_inp${INPUT_TOK}_out${OUTPUT_TOK}_user${CONCURRENT_REQ}.log