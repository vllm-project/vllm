#!/bin/bash
rm -rf /root/.cache/vllm

VLLM_RPC_TIMEOUT=1800000 \
VLLM_USE_V1=1 \
VLLM_ROCM_USE_AITER=1 \
SAFETENSORS_FAST_GPU=1 \
vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
 -tp 4 \
 --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
 --mm-encoder-tp-mode "data" \
 --trust_remote_code \
> server_Qwen_Qwen2.5-VL-72B-Instruct-syncupstream.log 2>&1