#!/bin/bash
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

DEFAULT_MODEL=meta-llama/Llama-3.1-8B
DEFAULT_VLLM_PORT=12346

NUM_PROMPTS=1000
REQUEST_RATE=10

MODEL=${1:-$DEFAULT_MODEL}
VLLM_PORT=${2:-$DEFAULT_VLLM_PORT}

check_and_download_sharegpt() {
    pushd $SCRIPT_DIR
    if [ ! -f "ShareGPT_V3_unfiltered_cleaned_split.json" ]; then
        wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
    fi
    popd
}

check_and_download_sharegpt

source "$ENGINE_DIR/.venv/bin/activate"
uv pip install pandas datasets

pushd $ENGINE_DIR/benchmarks
    python benchmark_serving.py --backend=vllm \
      --model $MODEL \
      --dataset-name sharegpt \
      --dataset-path $SCRIPT_DIR/ShareGPT_V3_unfiltered_cleaned_split.json \
      --request-rate $REQUEST_RATE \
      --num-prompts $NUM_PROMPTS \
      --port $VLLM_PORT
deactivate
popd
