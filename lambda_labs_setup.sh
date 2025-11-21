#!/bin/bash
# Lambda Labs Setup Script for vLLM INT4 + LoRA Testing
# Fixes common issues encountered during setup

set -e  # Exit on error

echo "================================"
echo "Lambda Labs vLLM Setup Script"
echo "================================"

# 1. Fix NumPy compatibility issues with system packages
echo "[1/6] Fixing NumPy compatibility..."
sudo mv /usr/lib/python3/dist-packages/tensorflow /usr/lib/python3/dist-packages/tensorflow.bak 2>/dev/null || true
sudo mv /usr/lib/python3/dist-packages/scipy /usr/lib/python3/dist-packages/scipy.bak 2>/dev/null || true
python3 -m pip install --user 'numpy<2' --force-reinstall

# 2. Clone vLLM fork
echo "[2/6] Cloning vLLM fork..."
if [ ! -d ~/vllm ]; then
    cd ~
    git clone https://github.com/sheikheddy/vllm.git
fi
cd ~/vllm
git fetch origin
git checkout feat/int4-compressed-tensors-lora-support

# 3. Install vLLM
echo "[3/6] Installing vLLM (this takes 15-20 minutes)..."
python3 -m pip install --upgrade pip
python3 -m pip install -e .

# 4. Clone and install compressed-tensors fork
echo "[4/6] Installing compressed-tensors fork..."
if [ ! -d ~/compressed-tensors ]; then
    cd ~
    git clone https://github.com/sheikheddy/compressed-tensors.git
fi
cd ~/compressed-tensors
python3 -m pip install -e .

# 5. Install test dependencies
echo "[5/6] Installing test dependencies..."
python3 -m pip install --user pytest

# 6. Verify installation
echo "[6/6] Verifying installation..."
python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
python3 -c "import compressed_tensors; print(f'compressed-tensors version: {compressed_tensors.__version__}')"
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "================================"
echo "Setup Complete!"
echo "================================"
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""
echo "Next steps:"
echo "  - Run tests: cd ~/vllm && python3 tests/test_vllm_int4_lora_e2e.py"
echo "  - Or use the test scripts in /tmp/"
echo "================================"
