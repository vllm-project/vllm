#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# GPU Testing Setup Guide for ILP Optimization

# This script helps set up vLLM for GPU testing of the ILP optimization
# Run this from the vLLM project root: bash scripts/setup_gpu_testing.sh

set -e

echo "=========================================="
echo "vLLM GPU Testing Setup"
echo "=========================================="
echo ""

# Check prerequisites
echo "Step 1: Checking GPU and CUDA availability..."
echo "-------------------------------------------"

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found. NVIDIA GPU drivers not installed."
    echo "   Install NVIDIA drivers first: https://www.nvidia.com/Download/driverDetails.aspx"
    exit 1
fi

echo "✓ NVIDIA drivers found"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader

# Check CUDA toolkit
if ! command -v nvcc &> /dev/null; then
    echo "⚠ WARNING: nvcc (CUDA compiler) not found. You can still run tests but may have issues."
    echo "   Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
else
    echo "✓ CUDA Toolkit found"
    nvcc --version | head -1
fi

# Check PyTorch CUDA support
echo ""
echo "Step 2: Checking PyTorch CUDA support..."
echo "-------------------------------------------"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU 0: {torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else \"N/A\"}')" || echo "❌ Error checking PyTorch"

echo ""
echo "Step 3: Recommended vLLM Installation Commands"
echo "-------------------------------------------"
echo ""
echo "Option A: Use pip with precompiled wheels (fastest, requires compatible GPU)"
echo "  VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=cuda"
echo ""
echo "Option B: Compile from source (slower but ensures optimization)"
echo "  uv pip install -e . --torch-backend=cuda"
echo ""
echo "Option C: Use specific CUDA version (if needed)"
echo "  uv pip install -e . --torch-backend=cuda --no-build-isolation"
echo ""

echo "Step 4: After Installation, Verify with:"
echo "-------------------------------------------"
echo "  python3 -c \"import torch; print(torch.ops._C.gelu_and_mul)\" # Should work"
echo "  python3 benchmarks/benchmark_ilp_kernels.py --num-tokens 32 128 --d 512 4096"
echo ""

echo "Step 5: Run Full ILP Test Suite"
echo "-------------------------------------------"
echo "  # Quick validation (5 min)"
echo "  python3 benchmarks/benchmark_ilp_kernels.py --iterations 50"
echo ""
echo "  # Comprehensive benchmark (20-30 min)"
echo "  python3 benchmarks/benchmark_ilp_kernels.py --iterations 200 \\"
echo "    --num-tokens 32 64 128 256 512 1024 2048 \\"
echo "    --d 512 1024 2048 4096 8192"
echo ""

echo "=========================================="
echo "Setup guide complete!"
echo "=========================================="
