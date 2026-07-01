#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Rebuild flash-attn from source for ROCm/torch ABI compatibility.
#
# The rocm/vllm-dev base image ships a flash-attn wheel compiled against the
# base image's PyTorch (release/2.11). When this image reinstalls a newer torch
# (e.g. the 2.13 test-channel wheels), that prebuilt flash_attn_2_cuda.so no
# longer matches the torch C++ ABI and fails to import with, e.g.:
#   undefined symbol: _ZN3c104impl3cow23materialize_cow_storageERNS_11StorageImplE
# (c10::impl::cow::materialize_cow_storage, changed by pytorch #179063).
# Recompiling flash-attn against the currently-installed torch fixes it. This
# mirrors install_torchcodec_rocm.sh and the flash-attn build in
# docker/Dockerfile.rocm_base.

set -e

# Pin to the same commit the base image builds (docker/Dockerfile.rocm_base
# FA_BRANCH); override via env if the base image pin changes.
FA_REPO="${FA_REPO:-https://github.com/Dao-AILab/flash-attention.git}"
FA_BRANCH="${FA_BRANCH:-0e60e394}"
# Cache directory for the rebuilt wheel to avoid redundant recompilation.
FA_WHEEL_CACHE="${FA_WHEEL_CACHE:-/root/.cache/flash-attn-wheels}"

echo "=== flash-attn ROCm rebuild ==="

# Skip if the installed flash-attn already imports (extension loads against the
# current torch). In the broken-ABI state this import raises, so we rebuild.
if python3 -c "import flash_attn, flash_attn_2_cuda" 2>/dev/null; then
    echo "flash-attn already imports against the current torch. Skipping."
    exit 0
fi

echo "flash-attn missing or ABI-incompatible with the current torch; rebuilding."

# Only build for the archs this image targets; the default (all archs) is slow.
# Match the base image: drop gfx11xx (consumer) archs from the CI build set.
GPU_ARCHS=$(echo "${PYTORCH_ROCM_ARCH}" | sed -e 's/;gfx1[0-9]\{3\}//g')
if [ -z "$GPU_ARCHS" ]; then
    GPU_ARCHS="gfx942;gfx950"
fi
echo "Building flash-attn for GPU_ARCHS=${GPU_ARCHS}"

ARCH_TAG="${GPU_ARCHS//;/_}"
CACHED_WHEEL="${FA_WHEEL_CACHE}/flash_attn-${FA_BRANCH}-${ARCH_TAG}.whl"

install_wheel() {
    # Replace the stale base-image flash-attn without disturbing torch or other
    # deps (the wheel is already built against the installed torch).
    python3 -m pip install --force-reinstall --no-deps "$1"
    python3 -c "import flash_attn, flash_attn_2_cuda; print('flash-attn imports OK:', flash_attn.__version__)"
}

if [ -f "$CACHED_WHEEL" ]; then
    echo "Found cached wheel: $CACHED_WHEEL"
    if install_wheel "$CACHED_WHEEL"; then
        echo "=== flash-attn install complete (cached) ==="
        exit 0
    fi
    echo "Cached wheel failed to import; rebuilding from source."
fi

BUILD_DIR=$(mktemp -d -t flash-attn-XXXXXX)
cleanup() { rm -rf "$BUILD_DIR"; }
trap cleanup EXIT

cd "$BUILD_DIR"
echo "Cloning flash-attention from ${FA_REPO} (checkout ${FA_BRANCH})..."
git clone "${FA_REPO}" flash-attention
cd flash-attention
git checkout "${FA_BRANCH}"
git submodule update --init

export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
if command -v ccache &>/dev/null; then
    export CMAKE_C_COMPILER_LAUNCHER=ccache
    export CMAKE_CXX_COMPILER_LAUNCHER=ccache
fi

echo "Building flash-attn wheel (MAX_JOBS=${MAX_JOBS})..."
GPU_ARCHS="${GPU_ARCHS}" python3 setup.py bdist_wheel --dist-dir=dist

BUILT_WHEEL=$(ls dist/flash_attn-*.whl 2>/dev/null | head -1)
if [ -z "$BUILT_WHEEL" ]; then
    echo "Error: no flash-attn wheel produced"
    exit 1
fi

install_wheel "$BUILT_WHEEL"

mkdir -p "$FA_WHEEL_CACHE"
cp "$BUILT_WHEEL" "$CACHED_WHEEL"
echo "Cached wheel to: $CACHED_WHEEL"
echo "=== flash-attn install complete ==="
