#!/bin/bash
# vLLM Installation Script (uses precompiled extensions from cmake-release-build)
# Usage: ./install_vllm.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/cmake-release-build"
CONDA_ENV="${VLLM_CONDA_ENV:-/workspace/aimo/miniconda/envs/vllm}"
PYTHON="${CONDA_ENV}/bin/python"
PIP="${CONDA_ENV}/bin/pip"
LOG_DIR="${SCRIPT_DIR}/../logs"

# Create log directory
mkdir -p "${LOG_DIR}"

echo "========================================"
echo "vLLM Installation (Precompiled Extensions)"
echo "========================================"
echo "Build directory: ${BUILD_DIR}"
echo "Python: ${PYTHON}"
echo "========================================"

# Verify build directory exists
if [ ! -d "${BUILD_DIR}" ]; then
    echo "ERROR: Build directory not found: ${BUILD_DIR}"
    echo "Run ./configure_build.sh and ./build_vllm.sh first"
    exit 1
fi

# Find .so files in build directory
echo "Finding compiled extensions..."
SO_FILES=$(find "${BUILD_DIR}" -name "*.so" -type f 2>/dev/null)
SO_COUNT=$(echo "${SO_FILES}" | grep -c ".so" || echo 0)

if [ "${SO_COUNT}" -eq 0 ]; then
    echo "ERROR: No .so files found in ${BUILD_DIR}"
    echo "Run ./build_vllm.sh first"
    exit 1
fi

echo "Found ${SO_COUNT} compiled extension(s)"

# Create symlinks for the compiled .so files into vllm/ directory
echo ""
echo "Creating symlinks for precompiled extensions..."

# Main _C.*.so extension
C_SO=$(find "${BUILD_DIR}" -name "_C.*.so" -type f | head -1)
if [ -n "${C_SO}" ]; then
    BASENAME=$(basename "${C_SO}")
    ln -sf "${C_SO}" "${SCRIPT_DIR}/vllm/${BASENAME}"
    echo "  Linked: vllm/${BASENAME}"
fi

# Main _core_C.*.so extension
CORE_C_SO=$(find "${BUILD_DIR}" -name "_core_C.*.so" -type f | head -1)
if [ -n "${CORE_C_SO}" ]; then
    BASENAME=$(basename "${CORE_C_SO}")
    ln -sf "${CORE_C_SO}" "${SCRIPT_DIR}/vllm/${BASENAME}"
    echo "  Linked: vllm/${BASENAME}"
fi

# MOE extension
MOE_SO=$(find "${BUILD_DIR}" -name "_moe_C.*.so" -type f | head -1)
if [ -n "${MOE_SO}" ]; then
    BASENAME=$(basename "${MOE_SO}")
    ln -sf "${MOE_SO}" "${SCRIPT_DIR}/vllm/${BASENAME}"
    echo "  Linked: vllm/${BASENAME}"
fi

# Flash attention extensions
mkdir -p "${SCRIPT_DIR}/vllm/vllm_flash_attn"

FA2_SO=$(find "${BUILD_DIR}" -name "_vllm_fa2_C.*.so" -type f | head -1)
if [ -n "${FA2_SO}" ]; then
    BASENAME=$(basename "${FA2_SO}")
    ln -sf "${FA2_SO}" "${SCRIPT_DIR}/vllm/vllm_flash_attn/${BASENAME}"
    echo "  Linked: vllm/vllm_flash_attn/${BASENAME}"
fi

FA3_SO=$(find "${BUILD_DIR}" -name "_vllm_fa3_C.*.so" -type f | head -1)
if [ -n "${FA3_SO}" ]; then
    BASENAME=$(basename "${FA3_SO}")
    ln -sf "${FA3_SO}" "${SCRIPT_DIR}/vllm/vllm_flash_attn/${BASENAME}"
    echo "  Linked: vllm/vllm_flash_attn/${BASENAME}"
fi

# cumem_allocator
CUMEM_SO=$(find "${BUILD_DIR}" -name "cumem_allocator.*.so" -type f | head -1)
if [ -n "${CUMEM_SO}" ]; then
    BASENAME=$(basename "${CUMEM_SO}")
    ln -sf "${CUMEM_SO}" "${SCRIPT_DIR}/vllm/${BASENAME}"
    echo "  Linked: vllm/${BASENAME}"
fi

# Install in editable mode with precompiled flag
echo ""
echo "Installing vLLM in editable mode with precompiled extensions..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/vllm_pip_install_${TIMESTAMP}.log"

cd "${SCRIPT_DIR}"
VLLM_USE_PRECOMPILED=1 "${PIP}" install -e . --no-build-isolation -v 2>&1 | tee "${LOG_FILE}"

INSTALL_STATUS=${PIPESTATUS[0]}

if [ ${INSTALL_STATUS} -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Installation successful!"
    echo "========================================"
    echo ""
    
    # Verify installation
    echo "Verifying installation..."
    "${PYTHON}" -c "import vllm; print(f'vLLM version: {vllm.__version__}')" 2>&1 || true
    
    echo ""
    echo "Log: ${LOG_FILE}"
else
    echo ""
    echo "========================================"
    echo "Installation FAILED! Check log: ${LOG_FILE}"
    echo "========================================"
    exit 1
fi
