#!/bin/bash

# ============================================================
# QDRANT COLLECTION INITIALIZATION SCRIPT
# ============================================================

set -e

QDRANT_URL="http://localhost:6333"
EMBEDDING_SIZE=4096  # qwen3-embedding:4b

echo "🚀 Initializing Qdrant collections..."

# ============================================================
# Create mem0_memories collection
# ============================================================

echo "📝 Creating mem0_memories collection..."

curl -X PUT "${QDRANT_URL}/collections/mem0_memories" \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "size": '"${EMBEDDING_SIZE}"',
      "distance": "Cosine",
      "on_disk": true
    },
    "optimizers_config": {
      "indexing_threshold": 10000
    },
    "hnsw_config": {
      "m": 16,
      "ef_construct": 100,
      "full_scan_threshold": 10000
    },
    "quantization_config": {
      "scalar": {
        "type": "int8",
        "quantile": 0.99,
        "always_ram": false
      }
    }
  }'

echo ""
echo "✅ mem0_memories collection created"

# Create payload indexes for mem0
curl -X PUT "${QDRANT_URL}/collections/mem0_memories/index" \
  -H "Content-Type: application/json" \
  -d '{
    "field_name": "user_id",
    "field_schema": "keyword"
  }'

curl -X PUT "${QDRANT_URL}/collections/mem0_memories/index" \
  -H "Content-Type: application/json" \
  -d '{
    "field_name": "agent_id",
    "field_schema": "keyword"
  }'

curl -X PUT "${QDRANT_URL}/collections/mem0_memories/index" \
  -H "Content-Type: application/json" \
  -d '{
    "field_name": "category",
    "field_schema": "keyword"
  }'

curl -X PUT "${QDRANT_URL}/collections/mem0_memories/index" \
  -H "Content-Type: application/json" \
  -d '{
    "field_name": "importance",
    "field_schema": "float"
  }'

echo "✅ mem0_memories indexes created"

# ============================================================
# Create librechat_rag collection
# ============================================================

echo "📚 Creating librechat_rag collection..."

curl -X PUT "${QDRANT_URL}/collections/librechat_rag" \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "size": '"${EMBEDDING_SIZE}"',
      "distance": "Cosine",
      "on_disk": true
    },
    "optimizers_config": {
      "indexing_threshold": 10000
    },
    "hnsw_config": {
      "m": 16,
      "ef_construct": 100,
      "full_scan_threshold": 10000
    },
    "quantization_config": {
      "scalar": {
        "type": "int8",
        "quantile": 0.99,
        "always_ram": false
      }
    }
  }'

echo ""
echo "✅ librechat_rag collection created"

# Create payload indexes for RAG
curl -X PUT "${QDRANT_URL}/collections/librechat_rag/index" \
  -H "Content-Type: application/json" \
  -d '{
    "field_name": "user_id",
    "field_schema": "keyword"
  }'

curl -X PUT "${QDRANT_URL}/collections/librechat_rag/index" \
  -H "Content-Type: application/json" \
  -d '{
    "field_name": "file_id",
    "field_schema": "keyword"
  }'

curl -X PUT "${QDRANT_URL}/collections/librechat_rag/index" \
  -H "Content-Type: application/json" \
  -d '{
    "field_name": "chunk_index",
    "field_schema": "integer"
  }'

echo "✅ librechat_rag indexes created"

# ============================================================
# Create conversation_cache collection (optional)
# ============================================================

echo "💬 Creating conversation_cache collection..."

curl -X PUT "${QDRANT_URL}/collections/conversation_cache" \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "size": '"${EMBEDDING_SIZE}"',
      "distance": "Cosine",
      "on_disk": false
    },
    "optimizers_config": {
      "indexing_threshold": 5000
    },
    "hnsw_config": {
      "m": 8,
      "ef_construct": 50,
      "full_scan_threshold": 5000
    }
  }'

echo ""
echo "✅ conversation_cache collection created"

# Create payload index
curl -X PUT "${QDRANT_URL}/collections/conversation_cache/index" \
  -H "Content-Type: application/json" \
  -d '{
    "field_name": "conversation_id",
    "field_schema": "keyword"
  }'

echo "✅ conversation_cache indexes created"

# ============================================================
# Verify collections
# ============================================================

echo ""
echo "🔍 Verifying collections..."
curl -s "${QDRANT_URL}/collections" | python3 -m json.tool

echo ""
echo "✅ All Qdrant collections initialized successfully!"
echo ""
echo "Collections created:"
echo "  - mem0_memories (for long-term memory)"
echo "  - librechat_rag (for document embeddings)"
echo "  - conversation_cache (for short-term context)"
echo ""
echo "You can view the Qdrant dashboard at: http://localhost:6333/dashboard"
