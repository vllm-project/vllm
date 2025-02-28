#!/bin/bash

# Define the model to use
MODEL=${1:-"Qwen/Qwen2.5-7B-Instruct"}

# Define the backend to use
BACKEND=${2:-"vllm"}

# Define the dataset to use
DATASET=${3:-"xgrammar_bench"}

# Define the guided decoding backend
GUIDED_BACKEND=${4:-"xgrammar"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR=${5:-"$SCRIPT_DIR/benchmark_results"}

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Define QPS values to test
QPS_VALUES=(1 5 10 15 20 25)

# Common parameters
COMMON_PARAMS="--backend $BACKEND \
               --model $MODEL \
               --dataset $DATASET \
               --guided-decoding-backend $GUIDED_BACKEND \
               --save-results \
               --result-dir $OUTPUT_DIR"

echo "Starting guided decoding benchmark with model: $MODEL"
echo "Backend: $BACKEND"
echo "Dataset: $DATASET"
echo "Guided decoding backend: $GUIDED_BACKEND"
echo "Results will be saved to: $OUTPUT_DIR"
echo "----------------------------------------"

# Run benchmarks with different QPS values
for qps in "${QPS_VALUES[@]}"; do
  echo "Running benchmark with QPS: $qps"

  # Get git hash and branch for the filename
  GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
  GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

  # Construct filename for this run
  FILENAME="${GUIDED_BACKEND}_${BACKEND}_${qps}qps_$(basename $MODEL)_${DATASET}_${GIT_BRANCH}_${GIT_HASH}.txt"

  # Run the benchmark
  python "$SCRIPT_DIR/benchmark_serving_guided.py" $COMMON_PARAMS \
    --request-rate $qps \
    --result-filename "$FILENAME"

  echo "Completed benchmark with QPS: $qps"
  echo "----------------------------------------"
done

echo "All benchmarks completed!"
echo "Results saved to: $OUTPUT_DIR"
