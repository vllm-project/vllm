#!/usr/bin/env bash
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
DATASET_PATH=~/CacheBlend/inputs/musique_s.json
PROMPT_BUILD_METHOD=QA
KV_STORAGE_SIZE=30GB
KV_CHUNK_SIZE=256
QPS=3.5
BASE_URL="http://localhost:8000/v1"
DATASET_NAME=$(echo $DATASET_PATH | awk -F'/' '{print $NF}' | awk -F'.' '{print $1}')
OUTPUT_FILE="$DATASET_NAME"_lmcache_qps_"$QPS".csv

export LMCACHE_CONFIG_FILE="example_blending.yaml"

log_str=$(python3 precompute.py --model "$MODEL_NAME"\
    --dataset "$DATASET_PATH" \
    --prompt-build-method $PROMPT_BUILD_METHOD \
    --kv-storage-size $KV_STORAGE_SIZE --kv-chunk-size $KV_CHUNK_SIZE \
    --base-url $BASE_URL)
echo "$log_str"
RETURNED_END_INDEX=$(echo "$log_str" | awk '{print $5}')
# Assert non-empty.
if [ -z "$RETURNED_END_INDEX" ]; then
    echo "Precompute returns empty end index"
    exit 1
fi
python3 rag.py --qps $QPS\
 --model "$MODEL_NAME" --dataset "$DATASET_PATH" \
 --end-index "$RETURNED_END_INDEX" --separator "[BLEND_SEP]"\
  --prompt-build-method $PROMPT_BUILD_METHOD --base-url $BASE_URL \
  --max-tokens 32 --output "$OUTPUT_FILE"