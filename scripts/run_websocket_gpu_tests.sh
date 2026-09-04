#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Script to setup environment and run WebSocket GPU integration tests

set -e

echo "=== WebSocket GPU Integration Test Setup ==="

# Check CUDA availability
echo "Checking CUDA availability..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Check GPU
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Check model cache
echo ""
echo "Checking model cache..."
if [ -d ~/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct ]; then
    echo "Model Qwen/Qwen2-0.5B-Instruct found in cache"
else
    echo "Model Qwen/Qwen2-0.5B-Instruct not in cache, downloading..."
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2-0.5B-Instruct')"
fi

# Install test dependencies if needed
echo ""
echo "Checking test dependencies..."
pip3 install pytest pytest-asyncio websockets openai httpx --quiet 2>/dev/null || true

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run the tests:"
echo "  pytest tests/entrypoints/openai/responses/test_websocket_gpu.py -v -s"
echo ""
echo "Or with specific GPU:"
echo "  CUDA_VISIBLE_DEVICES=0 pytest tests/entrypoints/openai/responses/test_websocket_gpu.py -v -s"
