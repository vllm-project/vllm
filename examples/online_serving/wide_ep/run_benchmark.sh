#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Error: Experiment tag required"
    echo "Usage: $0 <experiment_tag>"
    echo "Example: $0 dpep16"
    exit 1
fi

EXPERIMENT_TAG="$1"

# Configuration
HOST_URL="http://127.0.0.1:8000"
MODEL_NAME="dsv3"
CONCURRENCY_LEVELS=(4096) # 512 conccurency per rank, 8 ranks per node
PROMPTS_PER_CONCURRENCY=10
RESULT_DIR="results/${EXPERIMENT_TAG}"
MODEL_PATH="deepseek-ai/DeepSeek-V3-0324"

# Create output directory
mkdir -p $RESULT_DIR
timestamp=$(date +%Y%m%d_%H%M%S)

# Workload configuration
ITL=2 # ITL > 1 required for DeepSeek BOS token
OTL=256

echo "Starting benchmark sweep: $EXPERIMENT_TAG at $(date)"
echo "Workload: ITL=$ITL, OTL=$OTL"
echo "=========================================="

for concurrency in "${CONCURRENCY_LEVELS[@]}"; do
    output_file="c${concurrency}_${timestamp}.json"

    echo "Running concurrency=${concurrency} users..."
    NUM_REQUESTS=$((PROMPTS_PER_CONCURRENCY * concurrency))

    vllm bench serve \
        --base-url "$HOST_URL" \
        --model "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" \
        --random-input-len "$ITL" \
        --random-output-len "$OTL" \
        --num-prompts "$NUM_REQUESTS" \
        --max-concurrency "$concurrency" \
        --ready-check-timeout-sec 0 \
        --save-result \
        --result-dir "$RESULT_DIR" \
        --request-id-prefix "$EXPERIMENT_TAG" \
        --result-filename "$output_file"

    sleep 2
done


echo ""
echo "=========================================="
echo "Benchmark completed at $(date)"
echo "Results saved to: $RESULT_DIR"


