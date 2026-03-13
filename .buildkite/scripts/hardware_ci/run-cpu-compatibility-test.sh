#!/bin/bash
set -euox pipefail

export VLLM_CPU_KVCACHE_SPACE=1 
export VLLM_CPU_CI_ENV=1
# Reduce sub-processes for acceleration
export TORCH_COMPILE_DISABLE=1 
export VLLM_ENABLE_V1_MULTIPROCESSING=0

SDE_ARCHIVE="sde-external-10.7.0-2026-02-18-lin.tar.xz"
SDE_CHECKSUM="<EXPECTED_SHA256_CHECKSUM>" # TODO: Add the correct checksum
wget "https://downloadmirror.intel.com/913594/${SDE_ARCHIVE}"
echo "${SDE_CHECKSUM}  ${SDE_ARCHIVE}" | sha256sum --check
mkdir -p sde
tar -xvf "./${SDE_ARCHIVE}" --strip-components=1 -C ./sde/

# Test Sky Lake (AVX512F)
./sde/sde64 -skl -- python3 examples/basic/offline_inference/generate.py --model facebook/opt-125m --dtype bfloat16

# Test Cascade Lake (AVX512F + VNNI)
./sde/sde64 -clx -- python3 examples/basic/offline_inference/generate.py --model facebook/opt-125m --dtype bfloat16

# Test Cooper Lake (AVX512F + VNNI + BF16)
./sde/sde64 -cpx -- python3 examples/basic/offline_inference/generate.py --model facebook/opt-125m --dtype bfloat16
