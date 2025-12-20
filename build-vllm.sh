#!/bin/bash
set -e

# vLLM Build Script for ROCm
# Run this INSIDE the vllm-rocm-dev Docker container

echo "=========================================="
echo "vLLM ROCm Build Script"
echo "=========================================="
echo ""

# Check we're in the right directory
if [ ! -f "setup.py" ]; then
    echo "ERROR: setup.py not found. Are you in the vLLM directory?"
    exit 1
fi

echo "Setting up ROCm environment variables..."
export ROCM_PATH=/opt/rocm
export USE_ROCM=1

echo "Installing build dependencies..."
pip install --upgrade pip setuptools wheel

echo ""
echo "=========================================="
echo "Starting vLLM Build"
echo "=========================================="
echo "This will take approximately 10-15 minutes."
echo "Progress will be shown below..."
echo ""
echo "Build started at: $(date)"
echo ""

# Build vLLM
pip install -e . --verbose

echo ""
echo "Build completed at: $(date)"
echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="
echo ""

# Verify installation
echo "Testing vLLM import..."
python -c "import vllm; print('✓ vLLM imported successfully')"
python -c "import vllm; print('vLLM version:', vllm.__version__)"

echo ""
echo "Checking ROCm availability..."
python -c "import torch; print('ROCm available:', torch.cuda.is_available())"

echo ""
echo "Getting device information..."
python -c "import torch; print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected (expected without AMD hardware)')"

echo ""
echo "=========================================="
echo "Installing Pre-commit Hooks"
echo "=========================================="
echo ""

pip install pre-commit
pre-commit install

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run a basic test:"
echo "   pytest tests/models/test_qwen.py -v"
echo ""
echo "2. Try reproducing Bug #6 (gfx1100 flash attention):"
echo "   pytest tests/kernels/test_flash_attn.py -v"
echo ""
echo "3. Document this setup in progress/environment-notes.md"
echo ""
echo "Build time and configuration have been logged."
echo "You can now start working on bug fixes!"
echo ""
