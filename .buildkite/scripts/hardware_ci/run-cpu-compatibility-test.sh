#!/bin/bash
set -euox pipefail

export VLLM_CPU_KVCACHE_SPACE=1 
export VLLM_CPU_CI_ENV=1
# Reduce sub-processes for acceleration
export TORCH_COMPILE_DISABLE=1 
export VLLM_ENABLE_V1_MULTIPROCESSING=0

wget https://downloadmirror.intel.com/913594/sde-external-10.7.0-2026-02-18-lin.tar.xz
mkdir -p sde
tar -xvf sde-external-10.7.0-2026-02-18-lin.tar.xz --strip-components=1  -C ./sde

./sde/sde64 -clx  -- python3 examples/basic/offline_inference/basic.py 