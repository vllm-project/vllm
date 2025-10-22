#!/bin/bash

rm -rf /root/.cache/vllm

pip install flash-linear-attention
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
python3 setup.py install


# cudagraph|tp=4|bf16
VLLM_USE_V1=1 \
VLLM_ROCM_USE_AITER=1 \
SAFETENSORS_FAST_GPU=1 \
VLLM_ROCM_USE_AITER_MHA=0 \
HIP_VISIBLE_DEVICES=0,1,2,3 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct \
    --port 8000 --tensor-parallel-size 4 --max-model-len 262114


# cudagraph|tp=8&ep|bf16
VLLM_USE_V1=1 \
VLLM_ROCM_USE_AITER=1 \
SAFETENSORS_FAST_GPU=1 \
VLLM_ROCM_USE_AITER_MHA=0 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct \
    --port 8000 --tensor-parallel-size 8 --max-model-len 262114 --enable-expert-parallel


# eager|tp=4|MTP=2|bf16
VLLM_USE_V1=1 \
VLLM_ROCM_USE_AITER=1 \
SAFETENSORS_FAST_GPU=1 \
VLLM_ROCM_USE_AITER_MHA=0 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct \
    --port 8000 --tensor-parallel-size 4 --max-model-len 262114 \
    --force-eager \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'