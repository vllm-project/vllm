#!/usr/bin/env bash
set -euo pipefail

echo '+ python3 repro_persistent_topk.py --backend persistent --full --repeats 20'
python3 repro_persistent_topk.py \
  --backend persistent --full --repeats 20

echo '+ TORCH_CUDA_ARCH_LIST=10.3 MAX_JOBS=4 PATCHED_TOPK_SOURCE=$PWD/patched_persistent_topk_ext.cu PATCHED_TOPK_INCLUDE=$PWD python3 repro_persistent_topk.py --backend overflow-extension --full --repeats 20'
TORCH_CUDA_ARCH_LIST=10.3 MAX_JOBS=4 \
PATCHED_TOPK_SOURCE="$PWD/patched_persistent_topk_ext.cu" \
PATCHED_TOPK_INCLUDE="$PWD" \
python3 repro_persistent_topk.py \
  --backend overflow-extension --full --repeats 20
