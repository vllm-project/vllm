#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Script to install TorchCodec from source (required for ROCm compatibility)
# The PyPI wheel is built against upstream PyTorch and has ABI mismatches with
# ROCm's custom torch build, so we must compile from source.

set -e

TORCHCODEC_REPO="${TORCHCODEC_REPO:-https://github.com/pytorch/torchcodec.git}"
# Pin to a specific release for reproducibility; update as needed.
TORCHCODEC_BRANCH="${TORCHCODEC_BRANCH:-v0.10.0}"
# Cache directory for pre-built wheels to avoid redundant recompilation.
TORCHCODEC_WHEEL_CACHE="${TORCHCODEC_WHEEL_CACHE:-/root/.cache/torchcodec-wheels}"

echo "=== TorchCodec Installation Script ==="

# Check if torchcodec is already installed and working
if python3 -c "from torchcodec.decoders import VideoDecoder" 2>/dev/null; then
    echo "TorchCodec is already installed and working. Skipping."
    exit 0
fi

# Try to install from cached wheel first
ARCH_TAG="${PYTORCH_ROCM_ARCH:-all}"
# Normalize arch tag (replace ; with _) for use in filename
ARCH_TAG="${ARCH_TAG//;/_}"
CACHED_WHEEL="${TORCHCODEC_WHEEL_CACHE}/torchcodec-${TORCHCODEC_BRANCH}-${ARCH_TAG}.whl"

if [ -f "$CACHED_WHEEL" ]; then
    echo "Found cached wheel: $CACHED_WHEEL"
    pip install "$CACHED_WHEEL" && {
        echo "Installed from cached wheel."
        echo "=== TorchCodec installation complete ==="
        exit 0
    }
    echo "Cached wheel installation failed, rebuilding from source..."
fi

echo "TorchCodec not found. Installing from source..."

# Install system dependencies (FFmpeg + pkg-config) if not already present.
# The Docker test image pre-installs these, so this is a fallback for other envs.
install_system_deps() {
    if command -v apt-get &> /dev/null; then
        echo "Installing system dependencies..."
        apt-get update && apt-get install -y --no-install-recommends \
            pkg-config \
            ffmpeg libavcodec-dev libavformat-dev libavutil-dev \
            libswscale-dev libavdevice-dev libavfilter-dev libswresample-dev
    else
        echo "Warning: apt-get did not work. Please install dependencies manually."
        return 1
    fi
}

# Check for pkg-config
if ! command -v pkg-config &> /dev/null; then
    echo "pkg-config not found. Installing system dependencies..."
    install_system_deps
fi

# Check for required FFmpeg libraries
echo "Checking for FFmpeg libraries..."
if ! pkg-config --exists libavcodec libavformat libavutil libswscale libavdevice libavfilter libswresample 2>/dev/null; then
    echo "FFmpeg development libraries not found. Installing..."
    install_system_deps
fi

# Install Python build dependencies
echo "Installing Python build dependencies..."
pip install pybind11 setuptools wheel

# Set pybind11 cmake path so CMake can find it
export pybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
export CMAKE_PREFIX_PATH="${pybind11_DIR}:${CMAKE_PREFIX_PATH}"
echo "pybind11_DIR set to: $pybind11_DIR"

# Limit GPU architectures to only what this image targets.
# The default builds for all supported archs which is very slow.
if [ -n "$PYTORCH_ROCM_ARCH" ]; then
    echo "Building for PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH"
fi

# Create temp directory for build
BUILD_DIR=$(mktemp -d -t torchcodec-XXXXXX)
echo "Building in temporary directory: $BUILD_DIR"

cleanup() {
    echo "Cleaning up $BUILD_DIR"
    rm -rf "$BUILD_DIR"
}
trap cleanup EXIT

# Clone and build
cd "$BUILD_DIR"
echo "Cloning TorchCodec from $TORCHCODEC_REPO (branch: $TORCHCODEC_BRANCH)..."
git clone --depth 1 --branch "$TORCHCODEC_BRANCH" "$TORCHCODEC_REPO" torchcodec

cd torchcodec

# Set build environment for ROCm compatibility
export TORCHCODEC_CMAKE_BUILD_DIR="${PWD}/build"
export TORCHCODEC_DISABLE_COMPILE_WARNING_AS_ERROR=1
export I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1
# Use ninja for faster builds and parallelize compilation
export CMAKE_GENERATOR=Ninja
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
# Use ccache if available to speed up recompilation
if command -v ccache &> /dev/null; then
    export CMAKE_C_COMPILER_LAUNCHER=ccache
    export CMAKE_CXX_COMPILER_LAUNCHER=ccache
fi

echo "Building TorchCodec (MAX_JOBS=$MAX_JOBS)..."
pip wheel . --no-build-isolation --no-deps -w "$BUILD_DIR/dist"

# Install the built wheel
BUILT_WHEEL=$(ls "$BUILD_DIR/dist"/torchcodec-*.whl 2>/dev/null | head -1)
if [ -z "$BUILT_WHEEL" ]; then
    echo "Error: No wheel produced"
    exit 1
fi

pip install "$BUILT_WHEEL"

# Cache the wheel for future runs
mkdir -p "$TORCHCODEC_WHEEL_CACHE"
cp "$BUILT_WHEEL" "$CACHED_WHEEL"
echo "Cached wheel to: $CACHED_WHEEL"

# Verify installation
echo "Verifying installation..."
if python3 -c "from torchcodec.decoders import VideoDecoder; print('TorchCodec installed successfully!')"; then
    echo "=== TorchCodec installation complete ==="
else
    echo "Error: TorchCodec installation failed verification"
    exit 1
fi
