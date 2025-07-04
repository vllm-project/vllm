#!/bin/bash

# default values
MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
BACKEND=${BACKEND:-"vllm"}
DATASET=${DATASET:-"xgrammar_bench"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR=${OUTPUT_DIR:-"$SCRIPT_DIR/structured_output_benchmark_results"}
PORT=${PORT:-8000}
STRUCTURED_OUTPUT_RATIO=${STRUCTURED_OUTPUT_RATIO:-1}
TOTAL_SECONDS=${TOTAL_SECONDS:-90}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-300}
TOKENIZER_MODE=${TOKENIZER_MODE:-"auto"}

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --model MODEL                  Model to benchmark (default: $MODEL)"
    echo "  --backend BACKEND              Backend to use (default: $BACKEND)" 
    echo "  --dataset DATASET              Dataset to use (default: $DATASET)"
    echo "  --max-new-tokens N             Maximum number of tokens to generate (default: $MAX_NEW_TOKENS)"
    echo "  --output-dir DIR               Output directory for results (default: $OUTPUT_DIR)"
    echo "  --port PORT                    Port to use (default: $PORT)"
    echo "  --structured-output-ratio N    Ratio of structured outputs (default: $STRUCTURED_OUTPUT_RATIO)"
    echo "  --tokenizer-mode MODE          Tokenizer mode to use (default: $TOKENIZER_MODE)"
    echo "  --total-seconds N              Total seconds to run the benchmark (default: $TOTAL_SECONDS)"
    echo "  -h, --help                     Show this help message and exit"
    exit 0
}

# parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --structured-output-ratio)
      STRUCTURED_OUTPUT_RATIO="$2"
      shift 2
      ;;
    --tokenizer-mode)
      TOKENIZER_MODE="$2"
      shift 2
      ;;
    --total-seconds)
      TOTAL_SECONDS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown argument: $1\n"
      usage
      ;;
  esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Define QPS values to test
QPS_VALUES=(25 20 15 10 5 1)

# Common parameters
COMMON_PARAMS="--backend $BACKEND \
               --model $MODEL \
               --dataset $DATASET \
               --structured-output-ratio $STRUCTURED_OUTPUT_RATIO \
               --save-results \
               --result-dir $OUTPUT_DIR \
               --output-len $MAX_NEW_TOKENS \
               --port $PORT \
               --tokenizer-mode $TOKENIZER_MODE"

echo "Starting structured output benchmark with model: $MODEL"
echo "Backend: $BACKEND"
echo "Dataset: $DATASET"
echo "Results will be saved to: $OUTPUT_DIR"
echo "----------------------------------------"

# Run benchmarks with different QPS values
for qps in "${QPS_VALUES[@]}"; do
  echo "Running benchmark with QPS: $qps"

  # Get git hash and branch for the filename
  GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
  GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

  # Construct filename for this run
  FILENAME="${BACKEND}_${qps}qps_$(basename $MODEL)_${DATASET}_${GIT_HASH}.json"

  NUM_PROMPTS=$(echo "$TOTAL_SECONDS * $qps" | bc)
  NUM_PROMPTS=${NUM_PROMPTS%.*}  # Remove fractional part
  echo "Running benchmark with $NUM_PROMPTS prompts"

  # Run the benchmark
  python "$SCRIPT_DIR/benchmark_serving_structured_output.py" $COMMON_PARAMS \
    --request-rate $qps \
    --result-filename "$FILENAME" \
    --num-prompts $NUM_PROMPTS

  echo "Completed benchmark with QPS: $qps"
  echo "----------------------------------------"
done

echo "All benchmarks completed!"
echo "Results saved to: $OUTPUT_DIR"
