#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Script to install TorchCodec from source (required for ROCm compatibility)

set -e

TORCHCODEC_REPO="${TORCHCODEC_REPO:-https://github.com/pytorch/torchcodec.git}"
TORCHCODEC_BRANCH="${TORCHCODEC_BRANCH:-main}"

echo "=== TorchCodec Installation Script ==="

# Check if torchcodec is already installed and working
if python3 -c "from torchcodec.decoders import VideoDecoder" 2>/dev/null; then
    echo "TorchCodec is already installed and working. Skipping."
    exit 0
fi

echo "TorchCodec not found. Installing from source..."

# Install system dependencies (FFmpeg + pkg-config)
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

echo "Building TorchCodec..."
pip install . --no-build-isolation

# Verify installation
echo "Verifying installation..."
if python3 -c "from torchcodec.decoders import VideoDecoder; print('TorchCodec installed successfully!')"; then
    echo "=== TorchCodec installation complete ==="
else
    echo "Error: TorchCodec installation failed verification"
    exit 1
fi