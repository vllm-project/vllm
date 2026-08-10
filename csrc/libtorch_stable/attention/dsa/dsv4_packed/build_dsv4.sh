#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Build the in-tree DSv4 bf16 packed sparse-attention TVM-FFI module.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
BUILD_DIR=${LITEDSA_DSV4_BUILD_DIR:-/tmp/litedsa_dsv4_build}
OUTPUT_SO=${LITEDSA_DSV4_SO:-$BUILD_DIR/dsa_dsv4.so}

PYTHON_INCLUDE=$(
  "$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_path("include"))'
)
TVM_FFI_INCLUDE=$(
  "$PYTHON_BIN" -c \
    'from pathlib import Path; import tvm_ffi; print(Path(tvm_ffi.__file__).parent / "include")'
)
FLASHINFER_DATA=$(
  "$PYTHON_BIN" -c \
    'from pathlib import Path; import flashinfer; print(Path(flashinfer.__file__).parent / "data")'
)
CXX11_ABI=$(
  "$PYTHON_BIN" -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))'
)

mkdir -p "$BUILD_DIR"
mkdir -p "$(dirname -- "$OUTPUT_SO")"
"$CUDA_HOME/bin/nvcc" \
  --compiler-options=-fPIC \
  --expt-relaxed-constexpr \
  --expt-extended-lambda \
  -static-global-template-stub=false \
  -std=c++17 --threads=1 -use_fast_math -Xfatbin=-compress-all -O3 \
  -gencode=arch=compute_100a,code=sm_100a \
  -D_GLIBCXX_USE_CXX11_ABI="$CXX11_ABI" \
  -DFLASHINFER_ENABLE_F16 -DFLASHINFER_ENABLE_BF16 \
  -DFLASHINFER_ENABLE_FP8_E4M3 -DFLASHINFER_ENABLE_FP8_E5M2 \
  -I"$SCRIPT_DIR" \
  -I"$FLASHINFER_DATA/cccl/cub" \
  -I"$FLASHINFER_DATA/cccl/libcudacxx/include" \
  -I"$FLASHINFER_DATA/cccl/thrust" \
  -I"$PYTHON_INCLUDE" \
  -I"$TVM_FFI_INCLUDE" \
  -I"$FLASHINFER_DATA/cutlass/include" \
  -I"$FLASHINFER_DATA/cutlass/tools/util/include" \
  -I"$FLASHINFER_DATA/include" \
  -I"$FLASHINFER_DATA/csrc" \
  -I"$FLASHINFER_DATA/spdlog/include" \
  -c "$SCRIPT_DIR/litedsa_dsv4_binding.cu" \
  -o "$BUILD_DIR/dsa_dsv4_binding.cuda.o"

TMP_SO="$BUILD_DIR/dsa_dsv4.so.tmp.$$"
trap 'rm -f "$TMP_SO"' EXIT HUP INT TERM
"${CXX:-c++}" -shared "$BUILD_DIR/dsa_dsv4_binding.cuda.o" \
  -L"$CUDA_HOME/lib64" -L"$CUDA_HOME/lib64/stubs" \
  -lcudart -lcuda -o "$TMP_SO"
mv "$TMP_SO" "$OUTPUT_SO"
trap - EXIT HUP INT TERM

echo "$OUTPUT_SO"
