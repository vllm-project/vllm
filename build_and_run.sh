#!/bin/bash

set -e
export VLLM_TARGET_DEVICE=metal
export PYTORCH_ENABLE_MPS_FALLBACK=1
# export VLLM_TARGET_DEVICE=cpu
pip uninstall vllm
python setup.py install
vllm serve Qwen/Qwen2.5-0.5B-Instruct
