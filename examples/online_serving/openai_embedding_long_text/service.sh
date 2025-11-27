#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# vLLM Embedding Server with Enhanced Chunked Processing
# This script starts a vLLM server with chunked processing enabled for long text embedding.
# Now supports proper pooling type validation and model-specific configurations.

set -euo pipefail

# Configuration
MODEL_NAME=${MODEL_NAME:-"intfloat/multilingual-e5-large"}
MODEL_CODE=${MODEL_CODE:-"multilingual-e5-large"}

PORT=${PORT:-31090}
GPU_COUNT=${GPU_COUNT:-1}
MAX_EMBED_LEN=${MAX_EMBED_LEN:-3072000}
API_KEY=${API_KEY:-"your-api-key"}

# Enhanced pooling configuration with model-specific defaults
POOLING_TYPE=${POOLING_TYPE:-"auto"}  # auto, MEAN, CLS, LAST
export VLLM_ENABLE_CHUNKED_PROCESSING=true
export CUDA_VISIBLE_DEVICES=2,3,4,5

echo "🚀 Starting vLLM Embedding Server with Enhanced Chunked Processing"
echo "=================================================================="

# Environment variables for optimization
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Function to determine optimal pooling type for known models
get_optimal_pooling_type() {
    local model="$1"
    case "$model" in
        *"e5-"* | *"multilingual-e5"*)
            echo "MEAN"  # E5 series native pooling
            ;;
        *"bge-"*)
            echo "CLS"   # BGE series native pooling
            ;;
        *"gte-"*)
            echo "LAST"  # GTE series native pooling
            ;;
        *"sentence-t5"* | *"st5"*)
            echo "MEAN"  # Sentence-T5 native pooling
            ;;
        *"jina-embeddings"*)
            echo "MEAN"  # Jina embeddings native pooling
            ;;
        *"Qwen"*"Embedding"*)
            echo "LAST"  # Qwen embeddings native pooling
            ;;
        *)
            echo "MEAN"  # Default native pooling for unknown models
            ;;
    esac
}

# Auto-detect pooling type if not explicitly set
if [ "$POOLING_TYPE" = "auto" ]; then
    POOLING_TYPE=$(get_optimal_pooling_type "$MODEL_NAME")
    echo "🔍 Auto-detected pooling type: $POOLING_TYPE for model $MODEL_NAME"
fi

# Display configuration
echo "📋 Configuration:"
echo "   - Model: $MODEL_NAME"
echo "   - Port: $PORT"
echo "   - GPU Count: $GPU_COUNT"
echo "   - Enhanced Chunked Processing: ${VLLM_ENABLE_CHUNKED_PROCESSING}"
echo "   - Max Embed Length: ${MAX_EMBED_LEN} tokens"
echo "   - Native Pooling Type: $POOLING_TYPE + Normalization"
echo "   - Cross-chunk Aggregation: MEAN (automatic)"
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

# Chunked processing uses unified MEAN aggregation
echo "ℹ️  Chunked Processing: Using $POOLING_TYPE pooling within chunks, MEAN aggregation across chunks"
echo "   - All chunks processed for complete semantic coverage"
echo "   - Weighted averaging based on chunk token counts"

echo ""
echo "🔧 Starting server with enhanced chunked processing configuration..."

# Build pooler config JSON
POOLER_CONFIG="{\"pooling_type\": \"$POOLING_TYPE\", \"normalize\": true, \"enable_chunked_processing\": ${VLLM_ENABLE_CHUNKED_PROCESSING}, \"max_embed_len\": ${MAX_EMBED_LEN}}"

# Start vLLM server with enhanced chunked processing
vllm serve "$MODEL_NAME" \
  --tensor-parallel-size "$GPU_COUNT" \
  --enforce-eager \
  --pooler-config "$POOLER_CONFIG" \
  --served-model-name ${MODEL_CODE} \
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
echo "   - Native Pooling: $POOLING_TYPE | Cross-chunk: MEAN"
echo ""
echo "🧪 Test the server with:"
echo "   python examples/online_serving/openai_embedding_long_text/client.py"
echo ""
echo "📚 Enhanced features enabled:"
echo "   ✅ Intelligent native pooling type detection"
echo "   ✅ Unified MEAN aggregation for chunked processing"
echo "   ✅ Model-specific native pooling optimization"
echo "   ✅ Enhanced max embedding length (${MAX_EMBED_LEN} tokens)"
echo "   ✅ Complete semantic coverage for all pooling types"
echo "   ✅ OpenAI-compatible API"
echo "   ✅ GPU acceleration"
echo ""
echo "🔧 Advanced usage:"
echo "   - Set POOLING_TYPE=MEAN|CLS|LAST to override auto-detection"
echo "   - Set MAX_EMBED_LEN to adjust maximum input length"
echo "   - All pooling types use MEAN aggregation across chunks" 
