#!/bin/bash

mkdir -p log

MODEL_LOG_NAME="opt-13b"
MODEL="facebook/opt-13b"

for BATCH_SIZE in 8 32 128; do
  for INPUT_LEN in 1 32 256 1024; do
    for OUTPUT_LEN in 1 16 128; do
      for TENSOR_PARALLEL_SIZE in 1 2 4; do
        python benchmark_latency.py \
            --model $MODEL \
            --batch-size $BATCH_SIZE \
            --input-len $INPUT_LEN \
            --output-len $OUTPUT_LEN \
            --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
            | tee -a log/model_${MODEL_LOG_NAME}_bs_${BATCH_SIZE}_in_${INPUT_LEN}_out_${OUTPUT_LEN}_tp_${TENSOR_PARALLEL_SIZE}.log
        sleep 0.1
      done
    done
  done
done
