#!/bin/bash
# vLLM Incremental Build Configuration Script for SM120 Blackwell
# Usage: ./configure_build.sh [SM_ARCH] (default: auto-detect)

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/cmake-release-build"
CONDA_ENV="${VLLM_CONDA_ENV:-/workspace/aimo/miniconda/envs/vllm}"
PYTHON="${CONDA_ENV}/bin/python"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
LOG_DIR="${SCRIPT_DIR}/../logs"

# Auto-detect SM architecture if not provided
if [ -n "$1" ]; then
    SM_ARCH="$1"
else
    SM_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
    echo "Auto-detected SM architecture: ${SM_ARCH}"
fi

# Validate SM architecture
if [[ ! "$SM_ARCH" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Invalid SM architecture: ${SM_ARCH}"
    echo "Usage: ./configure_build.sh [SM_ARCH]"
    echo "Example: ./configure_build.sh 120"
    exit 1
fi

# Create log directory
mkdir -p "${LOG_DIR}"

# Get PyTorch paths
TORCH_DIR=$("${PYTHON}" -c "import torch; print(torch.__path__[0])")
TORCH_CMAKE_DIR="${TORCH_DIR}/share/cmake/Torch"
TORCH_INCLUDE_DIR="${TORCH_DIR}/include"

TORCH_API_INCLUDE="${TORCH_INCLUDE_DIR}/torch/csrc/api/include"

echo "========================================"
echo "vLLM Incremental Build Configuration"
echo "========================================"
echo "Build directory: ${BUILD_DIR}"
echo "Python: ${PYTHON}"
echo "PyTorch: ${TORCH_DIR}"
echo "PyTorch Include: ${TORCH_INCLUDE_DIR}"
echo "Torch API Include: ${TORCH_API_INCLUDE}"
echo "CUDA: ${CUDA_HOME}"
echo "SM Architecture: sm_${SM_ARCH}"
echo "========================================"

# Clean and recreate build directory
if [ -d "${BUILD_DIR}" ]; then
    echo "Cleaning existing build directory..."
    rm -rf "${BUILD_DIR}"
fi
mkdir -p "${BUILD_DIR}"

cd "${BUILD_DIR}"

# Configure CMake
echo "Configuring CMake for SM${SM_ARCH}..."
CMAKE_PREFIX_PATH="${TORCH_DIR}" \
CPATH="${TORCH_INCLUDE_DIR}:${TORCH_API_INCLUDE}" \
cmake .. \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DVLLM_TARGET_DEVICE=cuda \
  -DVLLM_PYTHON_EXECUTABLE="${PYTHON}" \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES="${SM_ARCH}" \
  -DTorch_DIR="${TORCH_CMAKE_DIR}" \
  -DCMAKE_CUDA_FLAGS="-I${TORCH_INCLUDE_DIR} -I${TORCH_API_INCLUDE}" \
  -DCMAKE_CXX_FLAGS="-I${TORCH_INCLUDE_DIR} -I${TORCH_API_INCLUDE}" \
  -G Ninja \
  2>&1 | tee "${LOG_DIR}/vllm_cmake_configure_sm${SM_ARCH}.log"


echo ""
echo "========================================"
echo "CMake configuration complete!"
echo "Log: ${LOG_DIR}/vllm_cmake_configure_sm${SM_ARCH}.log"
echo ""
echo "Next steps:"
echo "  1. cd ${BUILD_DIR}"
echo "  2. ninja -j\$(nproc)  # or ninja -j12 for memory-constrained builds"
echo "  3. ../install_vllm.sh"
echo "========================================"
