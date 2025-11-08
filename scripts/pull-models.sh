#!/bin/bash

# ============================================================
# OLLAMA MODEL PULL SCRIPT
# ============================================================

set -e

echo "🤖 Pulling Ollama models for Local AI Stack..."

# Models to pull
LLM_MODEL="granite4:latest"
EMBEDDING_MODEL="qwen3-embedding:4b"
RERANKER_MODEL="dengcao/Qwen3-Reranker-4B"

# Function to pull model with retry
pull_model() {
    local model=$1
    local max_retries=3
    local retry=0

    while [ $retry -lt $max_retries ]; do
        echo "📥 Pulling $model (attempt $((retry + 1))/$max_retries)..."

        if docker compose exec ollama ollama pull "$model"; then
            echo "✅ Successfully pulled $model"
            return 0
        else
            retry=$((retry + 1))
            if [ $retry -lt $max_retries ]; then
                echo "⚠️  Failed to pull $model. Retrying in 5 seconds..."
                sleep 5
            fi
        fi
    done

    echo "❌ Failed to pull $model after $max_retries attempts"
    return 1
}

# Check if Ollama is running
if ! docker compose ps ollama | grep -q "Up"; then
    echo "Starting Ollama service..."
    docker compose up -d ollama
    echo "Waiting for Ollama to be ready..."
    sleep 10
fi

# Pull models
echo ""
echo "1/3 Pulling LLM model..."
pull_model "$LLM_MODEL"

echo ""
echo "2/3 Pulling embedding model..."
pull_model "$EMBEDDING_MODEL"

echo ""
echo "3/3 Pulling reranker model..."
pull_model "$RERANKER_MODEL"

# Verify models
echo ""
echo "📋 Installed models:"
docker compose exec ollama ollama list

echo ""
echo "✅ All models pulled successfully!"
echo ""
echo "Models:"
echo "  - LLM: $LLM_MODEL (128K context)"
echo "  - Embedder: $EMBEDDING_MODEL (4096 dims)"
echo "  - Reranker: $RERANKER_MODEL (4B params)"
