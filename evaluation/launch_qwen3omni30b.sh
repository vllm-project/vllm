#!/bin/bash
rm -rf /root/.cache/vllm

VLLM_RPC_TIMEOUT=1800000 \
VLLM_USE_V1=1 \
VLLM_ROCM_USE_AITER=1 \
SAFETENSORS_FAST_GPU=1 \
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
 -tp 2 \
 --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
 --mm-encoder-tp-mode "data" \
 --gpu-memory-utilization 0.8 \
 --trust_remote_code \
> server_Qwen_Qwen3-Omni-30B-A3B-Instruct-tp2.log 2>&1