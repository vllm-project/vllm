#!/bin/bash

export VLLM_USE_V1=1
export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_USE_TRITON_FLASH_ATTN=0
export NCCL_DEBUG=WARN
export VLLM_RPC_TIMEOUT=1800000
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_ROCM_USE_TRITON_ROPE=1 # add for acc
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 # add for acc, perf is not good for some cases

# for profiling
export VLLM_TORCH_PROFILER_WITH_STACK=1
export VLLM_TORCH_PROFILER_RECORD_SHAPES=1
export VLLM_TORCH_PROFILER_DIR=/root/profiler

# cache dirs
export VLLM_CACHE_ROOT=/root/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor

rm -rf /root/.cache/

model_path=deepseek-ai/DeepSeek-V3
vllm serve $model_path \
    --tensor-parallel-size 8 \
    --max-num-batched-tokens 32768 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --disable-log-requests \
    --gpu_memory_utilization 0.9 \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --block-size 1 \
    --async-scheduling \
    2>&1 | tee server-deepseek-ai_DeepSeek-V3.log
