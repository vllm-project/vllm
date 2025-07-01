#!/bin/bash

#@VARS
# Wait for vLLM server to be ready
until curl -s http://localhost:8000/v1/completions > /dev/null; do
    echo "Waiting for vLLM server to be ready..."
    sleep 15
done
echo "vLLM server is ready. Starting benchmark..."
## Start benchmarking vLLM serving
python3 /workspace/vllm/benchmarks/benchmark_serving.py \
                 --model $MODEL \
                 --base-url http://localhost:8000 \
                 --backend vllm \
                 --dataset-name sonnet \
                 --dataset-path /workspace/vllm/benchmarks/sonnet.txt \
                 --sonnet-prefix-len 100 \
                 --sonnet-input-len $INPUT_TOK \
                 --sonnet-output-len $OUTPUT_TOK \
                 --ignore-eos \
                 --trust-remote-code \
                 --num-prompts $NUM_PROMPTS \
                 --max-concurrency $CONCURRENT_REQ \
                 --metric-percentiles 90 \
2>&1 | tee -a logs/perftest_inp${INPUT_TOK}_out${OUTPUT_TOK}_user${CONCURRENT_REQ}.log