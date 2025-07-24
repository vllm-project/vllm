#!/bin/bash

set -euo pipefail

VLLM_LOG="$WORKSPACE/vllm_log.txt"
BM_LOG="$WORKSPACE/bm_log.txt"

if [ -n "$TARGET_COMMIT" ]; then
  head_hash=$(git rev-parse HEAD)
  if [ "$TARGET_COMMIT" != "$head_hash" ]; then
    echo "Error: target commit $TARGET_COMMIT does not match HEAD: $head_hash"
    exit 1
  fi
fi

echo "model: $MODEL"
echo

#
# create a log folder
#
mkdir "$WORKSPACE/log"

# TODO: Move to image building.
pip install pandas
pip install datasets

#
# create sonnet_4x
#
echo "Create sonnet_4x.txt"
echo "" > benchmarks/sonnet_4x.txt
for _ in {1..4}
 do
  cat benchmarks/sonnet.txt >> benchmarks/sonnet_4x.txt
done

#
# start vllm service in backend
#
echo "lanching vllm..."
echo "logging to $VLLM_LOG"
echo

VLLM_USE_V1=1 vllm serve $MODEL \
 --seed 42 \
 --disable-log-requests \
 --max-num-seqs $MAX_NUM_SEQS \
 --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
 --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
 --no-enable-prefix-caching \
 --download_dir $DOWNLOAD_DIR \
 --max-model-len $MAX_MODEL_LEN > "$VLLM_LOG" 2>&1 &


echo "wait for 20 minutes.."
echo
# sleep 1200
# wait for 10 minutes...
for i in {1..120}; do
    # TODO: detect other type of errors.
    if grep -Fq "raise RuntimeError" "$VLLM_LOG"; then
        echo "Detected RuntimeError, exiting."
        exit 1
    elif grep -Fq "Application startup complete" "$VLLM_LOG"; then
        echo "Application started"
        break
    else
        echo "wait for 10 seconds..."
        sleep 10
    fi
done

#
# run test
#
echo "run benchmark test..."
echo "logging to $BM_LOG"
echo
python benchmarks/benchmark_serving.py \
    --backend vllm \
    --model $MODEL  \
    --dataset-name sonnet \
    --dataset-path benchmarks/sonnet_4x.txt \
    --sonnet-input-len $INPUT_LEN \
    --sonnet-output-len $OUTPUT_LEN \
    --ignore-eos > "$BM_LOG"

echo "completed..."
echo

throughput=$(grep "Request throughput (req/s):" "$BM_LOG" | sed 's/[^0-9.]//g')
echo "throughput: $throughput"
echo
