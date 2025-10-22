#!/bin/bash
rm -rf /root/.cache/vllm

export GPU_ARCHS=gfx942

MODEL=EmbeddedLLM/Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic

AITER_ENABLE_VSKIP=0 \
AITER_ONLINE_TUNE=1 \
VLLM_USE_V1=1 \
VLLM_ROCM_USE_AITER=1 \
vllm serve $MODEL \
--tensor-parallel-size 8 \
--max-model-len 65536 \
--disable-log-requests \
--compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
--trust-remote-code \
--port 6789 \
> server-EmbeddedLLM_Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic.log 2>&1
