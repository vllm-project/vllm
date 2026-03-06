#!/usr/bin/env bash
# Build script for vLLM with proper CUDA environment for RTX 3080 Ti (sm_86)
#
# Usage:
#   ./build_vllm.sh          # Normal build
#   ./build_vllm.sh clean    # Clean and rebuild
#   ./build_vllm.sh fast     # Skip flash attention (faster but limited features)
#
# This script sets all necessary environment variables to avoid:
# - cicc/ptxas crashes from memory pressure
# - Driver/library version mismatches
# - Building for unnecessary CUDA architectures

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
echo_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
echo_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# =============================================================================
# Environment Setup
# =============================================================================

# CRITICAL: Include NVIDIA driver libraries FIRST in LD_LIBRARY_PATH
# This fixes "Driver/library version mismatch" errors
if [[ -d "/run/opengl-driver/lib" ]]; then
    export LD_LIBRARY_PATH="/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo_info "Added /run/opengl-driver/lib to LD_LIBRARY_PATH"
else
    echo_warn "/run/opengl-driver/lib not found - GPU detection may fail"
fi

# Set CUDA architecture to RTX 3080 Ti (sm_86) ONLY
# This avoids building for sm_80/sm_90 which we don't need
export TORCH_CUDA_ARCH_LIST="8.6"
echo_info "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST (RTX 3080 Ti)"

# Calculate optimal MAX_JOBS based on available RAM
# Each CUDA compilation job uses ~2-3GB RAM
TOTAL_RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
MAX_JOBS_CALCULATED=$((TOTAL_RAM_GB / 3))
MAX_JOBS_CALCULATED=$((MAX_JOBS_CALCULATED > 1 ? MAX_JOBS_CALCULATED : 1))

# Use environment variable if set, otherwise use calculated value
export MAX_JOBS="${MAX_JOBS:-$MAX_JOBS_CALCULATED}"
echo_info "MAX_JOBS=$MAX_JOBS (based on ${TOTAL_RAM_GB}GB RAM)"

# NVCC thread count (within each job)
export NVCC_THREADS="${NVCC_THREADS:-1}"
echo_info "NVCC_THREADS=$NVCC_THREADS"

# Disable triton for simpler build
export VLLM_USE_TRITON_FLASH_ATTN="${VLLM_USE_TRITON_FLASH_ATTN:-0}"

# =============================================================================
# Pre-flight Checks
# =============================================================================

echo_info "Running pre-flight checks..."

# Check CUDA availability
if command -v nvcc &>/dev/null; then
    NVCC_VERSION=$(nvcc --version | grep release | awk '{print $5}' | tr -d ',')
    echo_info "CUDA compiler: nvcc $NVCC_VERSION"
else
    echo_error "nvcc not found! Please enter nix develop shell first."
    exit 1
fi

# Check NVIDIA driver
if [[ -f "/proc/driver/nvidia/version" ]]; then
    DRIVER_VERSION=$(grep -oP 'Kernel Module\s+\K[0-9.]+' /proc/driver/nvidia/version || echo "unknown")
    echo_info "NVIDIA driver: $DRIVER_VERSION"
else
    echo_warn "NVIDIA kernel module not loaded"
fi

# Test nvidia-smi
if nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null; then
    echo_info "GPU detected successfully"
else
    echo_warn "nvidia-smi failed - this may cause issues"
    echo_warn "Make sure /run/opengl-driver/lib is in LD_LIBRARY_PATH"
fi

# Check Python
PYTHON_VERSION=$(python --version 2>&1)
echo_info "Python: $PYTHON_VERSION"

# =============================================================================
# Build Options
# =============================================================================

BUILD_MODE="${1:-normal}"

case "$BUILD_MODE" in
clean)
    echo_info "Cleaning build artifacts..."
    rm -rf build/ dist/ *.egg-info/ .eggs/
    find . -name "*.so" -path "./vllm/*" -delete 2>/dev/null || true
    rm -rf ~/.cache/uv/builds-v0/.tmp* 2>/dev/null || true
    echo_info "Clean complete"
    ;;
fast)
    echo_info "Fast build mode: skipping flash attention"
    export VLLM_SKIP_FLASH_ATTN="${VLLM_SKIP_FLASH_ATTN:-1}"
    # Note: This won't actually skip flash-attn in current vLLM
    # You'd need to modify CMakeLists.txt for true skip
    echo_warn "Flash attention skip not fully supported - build may still include it"
    ;;
normal)
    echo_info "Normal build mode"
    ;;
*)
    echo_error "Unknown build mode: $BUILD_MODE"
    echo "Usage: $0 [clean|fast|normal]"
    exit 1
    ;;
esac

# =============================================================================
# Build
# =============================================================================

echo ""
echo "============================================================"
echo "Starting vLLM build"
echo "============================================================"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "MAX_JOBS=$MAX_JOBS"
echo "NVCC_THREADS=$NVCC_THREADS"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "============================================================"
echo ""

# Install PyTorch if needed (CUDA 12.8)
if ! python -c "import torch; print(torch.__version__)" 2>/dev/null; then
    echo_info "Installing PyTorch with CUDA 12.8..."
    uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
fi

# Build vLLM
echo_info "Building vLLM (this may take 30-60 minutes with MAX_JOBS=$MAX_JOBS)..."
BUILD_LOG="build_$(date +%Y%m%d_%H%M%S).log"

if uv pip install -e . 2>&1 | tee "$BUILD_LOG"; then
    echo ""
    echo_info "============================================================"
    echo_info "BUILD SUCCESSFUL!"
    echo_info "============================================================"
    echo_info "Build log saved to: $BUILD_LOG"
    echo ""

    # Quick test
    echo_info "Running quick verification..."
    if python -c "from vllm import LLM; print('vLLM import OK')"; then
        echo_info "vLLM is ready to use!"
    else
        echo_warn "vLLM import test failed - check build log"
    fi
else
    echo ""
    echo_error "============================================================"
    echo_error "BUILD FAILED!"
    echo_error "============================================================"
    echo_error "Build log saved to: $BUILD_LOG"
    echo ""
    echo "Common fixes:"
    echo "  1. Reduce MAX_JOBS: MAX_JOBS=1 ./build_vllm.sh"
    echo "  2. Enter nix develop shell first: nix develop"
    echo "  3. Check driver: nvidia-smi"
    exit 1
fi
