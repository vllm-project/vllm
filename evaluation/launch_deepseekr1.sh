#!/bin/bash

rm -rf /root/.cache/vllm

export GPU_ARCHS=gfx942

MODEL=deepseek-ai/DeepSeek-R1

AITER_ENABLE_VSKIP=0 \
VLLM_USE_V1=1 \
VLLM_ROCM_USE_AITER=1 \
vllm serve $MODEL \
--tensor-parallel-size 8 \
--disable-log-requests \
--compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
--trust-remote-code \
--block-size 1 \
--port 6789 \
> server-deepseek-ai_DeepSeek-R1.log 2>&1