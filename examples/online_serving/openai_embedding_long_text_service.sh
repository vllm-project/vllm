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
ALLOW_NON_MEAN_CHUNKING=${ALLOW_NON_MEAN_CHUNKING:-"false"}
# export CUDA_VISIBLE_DEVICES=2,3,4,5

echo "🚀 Starting vLLM Embedding Server with Enhanced Chunked Processing"
echo "=================================================================="

# Environment variables for optimization
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Function to determine optimal pooling type for known models
get_optimal_pooling_type() {
    local model="$1"
    case "$model" in
        *"e5-"* | *"multilingual-e5"*)
            echo "MEAN"  # E5 series uses mean pooling
            ;;
        *"bge-"*)
            echo "CLS"   # BGE series uses CLS pooling
            ;;
        *"gte-"*)
            echo "MEAN"  # GTE series uses mean pooling
            ;;
        *"sentence-t5"* | *"st5"*)
            echo "MEAN"  # Sentence-T5 uses mean pooling
            ;;
        *)
            echo "MEAN"  # Default to MEAN for unknown models
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
echo "   - Enhanced Chunked Processing: ENABLED"
echo "   - Max Embed Length: ${MAX_EMBED_LEN} tokens"
echo "   - Pooling Type: $POOLING_TYPE + Normalization"
echo "   - Allow Non-MEAN Chunking: $ALLOW_NON_MEAN_CHUNKING"
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

# Warning for non-MEAN pooling types
if [ "$POOLING_TYPE" != "MEAN" ] && [ "$ALLOW_NON_MEAN_CHUNKING" != "true" ]; then
    echo ""
    echo "⚠️  IMPORTANT: Using $POOLING_TYPE pooling with chunked processing"
    echo "   This may produce different results than non-chunked processing."
    echo "   For BERT-type models with bidirectional attention, consider:"
    echo "   - Using MEAN pooling for mathematically equivalent results"
    echo "   - Setting ALLOW_NON_MEAN_CHUNKING=true to suppress this warning"
    echo ""
fi

echo ""
echo "🔧 Starting server with enhanced chunked processing configuration..."

# Build pooler config JSON
POOLER_CONFIG="{\"pooling_type\": \"$POOLING_TYPE\", \"normalize\": true, \"enable_chunked_processing\": true, \"max_embed_len\": ${MAX_EMBED_LEN}"

# Add allow_non_mean_chunking if needed
if [ "$ALLOW_NON_MEAN_CHUNKING" = "true" ]; then
    POOLER_CONFIG="${POOLER_CONFIG}, \"allow_non_mean_chunking\": true"
fi

POOLER_CONFIG="${POOLER_CONFIG}}"

# Start vLLM server with enhanced chunked processing
vllm serve "$MODEL_NAME" \
  --tensor-parallel-size "$GPU_COUNT" \
  --enforce-eager \
  --override-pooler-config "$POOLER_CONFIG" \
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
echo "   - Pooling Strategy: $POOLING_TYPE"
echo ""
echo "🧪 Test the server with:"
echo "   python examples/online_serving/openai_embedding_long_text_client.py"
echo ""
echo "📚 Enhanced features enabled:"
echo "   ✅ Intelligent pooling type detection and validation"
echo "   ✅ Long text chunked processing with proper aggregation"
echo "   ✅ Model-specific pooling strategy optimization"
echo "   ✅ Enhanced max embedding length (${MAX_EMBED_LEN} tokens)"
echo "   ✅ Automatic chunk aggregation (MEAN/CLS/LAST support)"
echo "   ✅ OpenAI-compatible API"
echo "   ✅ GPU acceleration"
echo ""
echo "🔧 Advanced usage:"
echo "   - Set POOLING_TYPE=MEAN|CLS|LAST to override auto-detection"
echo "   - Set ALLOW_NON_MEAN_CHUNKING=true for non-MEAN pooling without warnings"
echo "   - Set MAX_EMBED_LEN to adjust maximum input length" 
