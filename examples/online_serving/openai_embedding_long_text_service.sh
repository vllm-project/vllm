#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# vLLM Embedding Server with Chunked Processing
# This script starts a vLLM server with chunked processing enabled for long text embedding.

set -euo pipefail

# Configuration
MODEL_NAME=${MODEL_NAME:-"intfloat/multilingual-e5-large"}
MODEL_CODE=${MODEL_CODE:-"multilingual-e5-large"}
PORT=${PORT:-31090}
GPU_COUNT=${GPU_COUNT:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-10240}
API_KEY=${API_KEY:-"your-api-key"}

echo "🚀 Starting vLLM Embedding Server with Chunked Processing"
echo "================================================================"

# Environment variables for optimization
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Display configuration
echo "📋 Configuration:"
echo "   - Model: $MODEL_NAME"
echo "   - Port: $PORT"
echo "   - GPU Count: $GPU_COUNT"
echo "   - Max Model Length: $MAX_MODEL_LEN tokens"
echo "   - Chunked Processing: ENABLED"
echo "   - Pooling Type: CLS + Normalization"
echo ""

# Validate GPU availability
if command -v nvidia-smi &> /dev/null; then
    gpu_count=$(nvidia-smi --list-gpus | wc -l)
    echo "🖥️  Available GPUs: $gpu_count"
    if [ "$GPU_COUNT" -gt "$gpu_count" ]; then
        echo "⚠️  Warning: Requested $GPU_COUNT GPUs but only $gpu_count available"
        echo "   Adjusting to use $gpu_count GPUs"
        GPU_COUNT=$gpu_count
    fi
else
    echo "⚠️  Warning: nvidia-smi not found. GPU detection skipped."
fi

echo ""
echo "🔧 Starting server with chunked processing configuration..."

# Start vLLM server with chunked processing enabled
vllm serve "$MODEL_NAME" \
  --tensor-parallel-size "$GPU_COUNT" \
  --enforce-eager \
  --max-model-len "$MAX_MODEL_LEN" \
  --override-pooler-config '{"pooling_type": "CLS", "normalize": true, "enable_chunked_processing": true}' \
  --served-model-name ${MODEL_CODE} \
  --task embed \
  --use-v2-block-manager \
  --api-key "$API_KEY" \
  --trust-remote-code \
  --port "$PORT" \
  --host 0.0.0.0

echo ""
echo "✅ vLLM Embedding Server started successfully!"
echo ""
echo "📡 Server Information:"
echo "   - Base URL: http://localhost:$PORT"
echo "   - Model Code: ${MODEL_CODE}"
echo "   - API Key: $API_KEY"
echo ""
echo "🧪 Test the server with:"
echo "   python examples/online_serving/openai_embedding_long_text_client.py"
echo ""
echo "📚 Features enabled:"
echo "   ✅ Long text chunked processing"
echo "   ✅ Automatic chunk aggregation"
echo "   ✅ OpenAI-compatible API"
echo "   ✅ GPU acceleration" 