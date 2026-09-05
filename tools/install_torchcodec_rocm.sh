#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Script to install TorchCodec from source (required for ROCm compatibility)
# The PyPI wheel is built against upstream PyTorch and has ABI mismatches with
# ROCm's custom torch build, so we must compile from source.

set -e

TORCHCODEC_REPO="${TORCHCODEC_REPO:-https://github.com/pytorch/torchcodec.git}"
# v0.10.0, pinned to the immutable commit for reproducibility.
TORCHCODEC_COMMIT="${TORCHCODEC_COMMIT:-0b261b98080925f2b709712a5491a1e8dd817065}"
TORCHCODEC_CONSTRAINTS="${TORCHCODEC_CONSTRAINTS:-}"
TORCHCODEC_FORCE_REBUILD="${TORCHCODEC_FORCE_REBUILD:-0}"

echo "=== TorchCodec Installation Script ==="

case "$TORCHCODEC_FORCE_REBUILD" in
    0 | 1) ;;
    *)
        echo "Error: TORCHCODEC_FORCE_REBUILD must be 0 or 1"
        exit 2
        ;;
esac

if [ "$TORCHCODEC_FORCE_REBUILD" = "0" ] \
    && python3 -c "from torchcodec.decoders import VideoDecoder" 2>/dev/null; then
    echo "TorchCodec is already installed and working. Skipping."
    exit 0
fi

echo "Building TorchCodec from source..."

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
constraint_args=()
build_requirements=(packaging pybind11 setuptools wheel)
if [ -n "$TORCHCODEC_CONSTRAINTS" ]; then
    if [ ! -f "$TORCHCODEC_CONSTRAINTS" ]; then
        echo "Error: TorchCodec constraints file not found: $TORCHCODEC_CONSTRAINTS"
        exit 1
    fi
    constraint_args=(--constraint "$TORCHCODEC_CONSTRAINTS")
else
    build_requirements=(
        packaging==26.2 pybind11==3.0.4 setuptools==79.0.1 wheel==0.48.0
    )
fi
python3 -m pip install --no-deps \
    "${constraint_args[@]}" "${build_requirements[@]}"

# Set pybind11 cmake path so CMake can find it
pybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
export pybind11_DIR
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
echo "Cloning TorchCodec from $TORCHCODEC_REPO (commit: $TORCHCODEC_COMMIT)..."
printf '%s\n' "$TORCHCODEC_COMMIT" | grep -Eq '^[0-9a-f]{40}$'
git init -q torchcodec
git -C torchcodec remote add origin "$TORCHCODEC_REPO"
git -C torchcodec fetch --depth 1 origin "$TORCHCODEC_COMMIT"
git -C torchcodec checkout -q --detach FETCH_HEAD
test "$(git -C torchcodec rev-parse HEAD)" = "$TORCHCODEC_COMMIT"

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
# Never reuse a locally built wheel across changes to the parent ROCm/Torch ABI.
python3 -m pip wheel . --no-cache-dir --no-build-isolation --no-deps \
    -w "$BUILD_DIR/dist"

# Install the built wheel
BUILT_WHEEL=$(find "$BUILD_DIR/dist" -maxdepth 1 -type f \
    -name 'torchcodec-*.whl' -print -quit)
if [ -z "$BUILT_WHEEL" ]; then
    echo "Error: No wheel produced"
    exit 1
fi

python3 -m pip install --force-reinstall --no-deps "$BUILT_WHEEL"

# Verify installation
echo "Verifying installation..."
if python3 -c "from torchcodec.decoders import VideoDecoder; print('TorchCodec installed successfully!')"; then
    echo "=== TorchCodec installation complete ==="
else
    echo "Error: TorchCodec installation failed verification"
    exit 1
fi
