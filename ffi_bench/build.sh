#!/usr/bin/env bash
# Build all three binding variants and report build time + .so size.
# Run from the ffi_bench directory.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

PY="$HERE/../.venv/bin/python"
NVCC=/usr/local/cuda-13.0/bin/nvcc
ARCH=sm_103

TORCH_INC=$($PY -c "import torch.utils.cpp_extension as ce; print(' '.join(f'-I{p}' for p in ce.include_paths()))")
TORCH_LIB=$($PY -c "import torch.utils.cpp_extension as ce; print(' '.join(f'-L{p}' for p in ce.library_paths()))")
TORCH_ABI_FLAG="-D_GLIBCXX_USE_CXX11_ABI=1"

TVMFFI_CXX=$($HERE/../.venv/bin/tvm-ffi-config --cxxflags)
TVMFFI_LD=$($HERE/../.venv/bin/tvm-ffi-config --ldflags)
TVMFFI_LIBS=$($HERE/../.venv/bin/tvm-ffi-config --libs)

COMMON="-O3 -shared -Xcompiler -fPIC -arch=$ARCH -std=c++17 $TORCH_ABI_FLAG -DUSE_CUDA"

mkdir -p build

build_one() {
  local src=$1
  local out=$2
  shift 2
  echo ">>> Building $out"
  /usr/bin/time -f "  build_time_sec=%e  rss_mb=%M" \
    $NVCC $COMMON "$src" -o "build/$out" "$@" 2>&1 | tail -3
  ls -la "build/$out" | awk '{printf "  size_bytes=%s\n", $5}'
}

TORCH_LIBDIR=$($PY -c "import torch.utils.cpp_extension as ce; print(ce.library_paths()[0])")
TVMFFI_LIBDIR=$($HERE/../.venv/bin/tvm-ffi-config --libdir)

build_one binding_unstable.cu unstable.so \
  $TORCH_INC $TORCH_LIB \
  -lc10 -ltorch -ltorch_cpu -lc10_cuda -ltorch_cuda \
  -Xlinker -rpath -Xlinker "$TORCH_LIBDIR"

build_one binding_stable.cu stable.so \
  $TORCH_INC $TORCH_LIB \
  -lc10 -ltorch -ltorch_cpu -lc10_cuda -ltorch_cuda \
  -Xlinker -rpath -Xlinker "$TORCH_LIBDIR"

build_one binding_tvmffi.cu tvmffi.so \
  $TVMFFI_CXX $TVMFFI_LD $TVMFFI_LIBS \
  -Xlinker -rpath -Xlinker "$TVMFFI_LIBDIR"

echo
echo "=== summary ==="
ls -la build/
