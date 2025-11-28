#!/bin/bash
# Build script for vLLM with Metal backend on Apple Silicon

set -e  # Exit on error

echo "================================================"
echo "vLLM Metal Backend Build Script"
echo "================================================"
echo ""

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "ERROR: This script must be run on macOS"
    exit 1
fi

# Check for Apple Silicon
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    echo "ERROR: This script requires Apple Silicon (arm64), found: $ARCH"
    exit 1
fi

echo "✓ Running on Apple Silicon ($ARCH)"

# Check for Xcode Command Line Tools
if ! command -v xcrun &> /dev/null; then
    echo "ERROR: Xcode Command Line Tools not found"
    echo "Please install with: xcode-select --install"
    exit 1
fi

echo "✓ Xcode Command Line Tools found"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "✓ Python $PYTHON_VERSION found"

# Check for PyTorch with MPS support
echo ""
echo "Checking PyTorch installation..."
python3 -c "import torch; assert torch.backends.mps.is_available(), 'MPS not available'" 2>/dev/null
if [ $? -eq 0 ]; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    echo "✓ PyTorch $TORCH_VERSION with MPS support found"
else
    echo "ERROR: PyTorch with MPS support not found"
    echo "Please install PyTorch with MPS support:"
    echo "  pip3 install torch torchvision torchaudio"
    exit 1
fi

# Set build configuration
export VLLM_TARGET_DEVICE=metal
export CMAKE_BUILD_TYPE=Release

echo ""
echo "================================================"
echo "Build Configuration:"
echo "  Target Device: $VLLM_TARGET_DEVICE"
echo "  Build Type: $CMAKE_BUILD_TYPE"
echo "  Python: $(which python3)"
echo "================================================"
echo ""

# Clean previous build (optional)
if [ "$1" == "--clean" ]; then
    echo "Cleaning previous build..."
    rm -rf build
    rm -rf vllm/_metal_C*.so
    rm -rf vllm/*.metallib
    echo "✓ Clean complete"
    echo ""
fi

# Build Metal kernels manually first (for verification)
echo "Compiling Metal kernels..."
mkdir -p build

xcrun -sdk macosx metal \
    -std=metal3.0 \
    -O3 \
    -ffast-math \
    csrc/metal/paged_attention_v1.metal \
    csrc/metal/paged_attention_v2.metal \
    csrc/metal/cache_ops.metal \
    -o build/metal_kernels.air

if [ $? -ne 0 ]; then
    echo "ERROR: Metal kernel compilation failed"
    exit 1
fi

xcrun -sdk macosx metallib \
    build/metal_kernels.air \
    -o build/vllm_metal_kernels.metallib

if [ $? -ne 0 ]; then
    echo "ERROR: Metal library creation failed"
    exit 1
fi

echo "✓ Metal kernels compiled successfully"
echo ""

# Build vLLM with Metal support
echo "Building vLLM with Metal backend..."
echo ""

pip3 install -e . -v

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: vLLM build failed"
    exit 1
fi

echo ""
echo "================================================"
echo "Build Complete!"
echo "================================================"
echo ""

# Verify installation
echo "Verifying Metal extension..."
python3 -c "
import sys
try:
    import vllm._metal_C as metal_ops
    print('✓ Metal C extension loaded successfully')
    print('  Available operations:')
    ops = [attr for attr in dir(metal_ops) if not attr.startswith('_')]
    for op in ops:
        print(f'    - {op}')
except ImportError as e:
    print('✗ Failed to import Metal extension:', e)
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "Installation successful!"
    echo ""
    echo "To use vLLM with Metal backend:"
    echo ""
    echo "  from vllm import LLM"
    echo "  llm = LLM(model='your-model', device='mps')"
    echo ""
    echo "The Metal backend will be automatically selected."
    echo "================================================"
else
    echo ""
    echo "Warning: Metal extension verification failed"
    echo "The build may have completed but the extension cannot be imported"
fi
