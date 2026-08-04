#!/usr/bin/env bash
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
DATASET_PATH=~/CacheBlend/inputs/musique_s.json
PROMPT_BUILD_METHOD=QA
QPS=3.5
END_INDEX=32
BASE_URL="http://localhost:8000/v1"
DATASET_NAME=$(echo $DATASET_PATH | awk -F'/' '{print $NF}' | awk -F'.' '{print $1}')
OUTPUT_FILE="$DATASET_NAME"_vllm_qps_"$QPS".csv

python3 rag.py --qps $QPS\
 --model "$MODEL_NAME" --dataset "$DATASET_PATH" \
 --end-index "$END_INDEX" --warmup \
 --prompt-build-method $PROMPT_BUILD_METHOD --base-url $BASE_URL \
 --max-tokens 32 --output "$OUTPUT_FILE"