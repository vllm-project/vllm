#!/bin/bash

#@VARS

## Start server
vllm serve $MODEL \
        --block-size $block_size \
        --dtype $dtype \
        --tensor-parallel-size $tensor_parallel_size \
        --download_dir $HF_HOME \
        --max-model-len $max_model_len \
        --gpu-memory-utilization $gpu_memory_utilization \
        --use-padding-aware-scheduling \
        --max-num-seqs $max_num_seqs \
        --max-num-prefill-seqs $max_num_prefill_seqs \
        --num-scheduler-steps 1 \
        --disable-log-requests
