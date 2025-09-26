#!/bin/bash

# MODEL_NAME="deepseek-ai/DeepSeek-V3.1"
MODEL_NAME="Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"
# MODEL_NAME="Qwen/Qwen3-235B-A22B-Thinking-2507-FP8"
HOST="0.0.0.0"
PORT=8006

DATA_PARALLEL_SIZE=2
DATA_PARALLEL_SIZE_LOCAL=2
LEADER_ADDRESS="192.168.5.45"
# LEADER_ADDRESS="172.18.0.3"

NUM_REDUNDANT_EXPERTS=16
EPLB_WINDOW_SIZE=1000
EPLB_STEP_INTERVAL=3000
MAX_MODEL_LEN=16384
GPU_MEMORY_UTILIZATION=0.9

export DG_JIT_NVCC_COMPILER=/usr/local/cuda-12.8/bin/nvcc
export CUDA_HOME='/usr/local/cuda-12.8'

export VLLM_USE_V1=1
export VLLM_ALL2ALL_BACKEND="pplx"
# export VLLM_ALL2ALL_BACKEND="deepep_low_latency"
export VLLM_USE_DEEP_GEMM=1
# export VLLM_ATTENTION_BACKEND="TRITON_MLA"

# Launch the vLLM server
vllm serve $MODEL_NAME --trust-remote-code \
    --disable-log-requests \
    --host $HOST \
    --port $PORT \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN \
    --no-enable-prefix-caching \
    --enable-expert-parallel \
    --enable-elastic-ep \
    --enable-eplb \
    --eplb-config.num_redundant_experts $NUM_REDUNDANT_EXPERTS \
    --eplb-config.window_size $EPLB_WINDOW_SIZE \
    --eplb-config.step_interval $EPLB_STEP_INTERVAL \
    --data-parallel-backend ray \
    --data-parallel-size $DATA_PARALLEL_SIZE \
    --data-parallel-size-local $DATA_PARALLEL_SIZE_LOCAL \
    --data-parallel-address $LEADER_ADDRESS \
    --data-parallel-rpc-port 9876 \
    --data-parallel-start-rank 0