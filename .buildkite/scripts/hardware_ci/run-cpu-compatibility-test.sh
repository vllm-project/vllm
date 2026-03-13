#!/bin/bash
set -euox pipefail

export VLLM_CPU_KVCACHE_SPACE=1 
export VLLM_CPU_CI_ENV=1
# Reduce sub-processes for acceleration
export TORCH_COMPILE_DISABLE=1 
export VLLM_ENABLE_V1_MULTIPROCESSING=0

wget https://downloadmirror.intel.com/913594/sde-external-10.7.0-2026-02-18-lin.tar.xz
mkdir -p sde
tar -xvf ./sde-external-10.7.0-2026-02-18-lin.tar.xz --strip-components=1 -C ./sde/

# Test Sky Lake (AVX512F)
./sde/sde64 -skl -- python3 examples/basic/offline_inference/generate.py --model facebook/opt-125m --dtype bfloat16

# Test Cascade Lake (AVX512F + VNNI)
./sde/sde64 -clx -- python3 examples/basic/offline_inference/generate.py --model facebook/opt-125m --dtype bfloat16

# Test Cooper Lake (AVX512F + VNNI + BF16)
./sde/sde64 -clx -- python3 examples/basic/offline_inference/generate.py --model facebook/opt-125m --dtype bfloat16
