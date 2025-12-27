#!/bin/bash
# vLLM Ninja Build Script
# Usage: ./build_vllm.sh [PARALLEL_JOBS] (default: 12)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/cmake-release-build"
LOG_DIR="${SCRIPT_DIR}/logs"
PARALLEL_JOBS="${1:-12}"

# Create log directory
mkdir -p "${LOG_DIR}"

# Check if build directory exists
if [ ! -d "${BUILD_DIR}" ]; then
    echo "ERROR: Build directory not found: ${BUILD_DIR}"
    echo "Run ./configure_build.sh first"
    exit 1
fi

# Check if Ninja build file exists
if [ ! -f "${BUILD_DIR}/build.ninja" ]; then
    echo "ERROR: build.ninja not found in ${BUILD_DIR}"
    echo "Run ./configure_build.sh first"
    exit 1
fi

echo "========================================"
echo "vLLM Ninja Build"
echo "========================================"
echo "Build directory: ${BUILD_DIR}"
echo "Parallel jobs: ${PARALLEL_JOBS}"
echo "========================================"

cd "${BUILD_DIR}"

# Run ninja build
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/vllm_ninja_build_${TIMESTAMP}.log"

echo "Building vLLM..."
echo "Log: ${LOG_FILE}"
echo ""

# Use time to measure build duration
time ninja -j${PARALLEL_JOBS} 2>&1 | tee "${LOG_FILE}"

BUILD_STATUS=${PIPESTATUS[0]}

if [ ${BUILD_STATUS} -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Build successful!"
    echo "========================================"
    echo ""
    echo "Next step: ./install_vllm.sh"
else
    echo ""
    echo "========================================"
    echo "Build FAILED! Check log: ${LOG_FILE}"
    echo "========================================"
    exit 1
fi
