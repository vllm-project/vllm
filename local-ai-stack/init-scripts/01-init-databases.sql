-- ============================================================
-- POSTGRESQL INITIALIZATION SCRIPT
-- Create databases and enable extensions
-- ============================================================

-- Enable PGVector extension for RAG
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- RAG API SCHEMA
-- ============================================================

-- Create schema for RAG
CREATE SCHEMA IF NOT EXISTS rag;

-- File embeddings table
CREATE TABLE IF NOT EXISTS rag.file_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    file_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(4096),  -- qwen3-embedding:4b dimension
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_file_embeddings_file_id ON rag.file_embeddings(file_id);
CREATE INDEX IF NOT EXISTS idx_file_embeddings_user_id ON rag.file_embeddings(user_id);
CREATE INDEX IF NOT EXISTS idx_file_embeddings_embedding ON rag.file_embeddings USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_file_embeddings_metadata ON rag.file_embeddings USING gin(metadata);

-- Files metadata table
CREATE TABLE IF NOT EXISTS rag.files (
    id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    filename VARCHAR(500) NOT NULL,
    filepath TEXT NOT NULL,
    mimetype VARCHAR(100),
    size_bytes BIGINT,
    chunk_count INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'pending',  -- pending, processing, completed, failed
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_files_user_id ON rag.files(user_id);
CREATE INDEX IF NOT EXISTS idx_files_status ON rag.files(status);

-- ============================================================
-- MEM0 / OPENMEMORY SCHEMA
-- ============================================================

-- Create schema for mem0
CREATE SCHEMA IF NOT EXISTS mem0;

-- Memories metadata table (vector data in Qdrant)
CREATE TABLE IF NOT EXISTS mem0.memories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    agent_id VARCHAR(255),
    category VARCHAR(100),
    memory_text TEXT NOT NULL,
    importance FLOAT DEFAULT 0.5,
    access_count INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMP,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_memories_user_id ON mem0.memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_agent_id ON mem0.memories(agent_id);
CREATE INDEX IF NOT EXISTS idx_memories_category ON mem0.memories(category);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON mem0.memories(importance);
CREATE INDEX IF NOT EXISTS idx_memories_metadata ON mem0.memories USING gin(metadata);

-- Memory relationships (for graph memory)
CREATE TABLE IF NOT EXISTS mem0.memory_relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_memory_id UUID REFERENCES mem0.memories(id) ON DELETE CASCADE,
    target_memory_id UUID REFERENCES mem0.memories(id) ON DELETE CASCADE,
    relationship_type VARCHAR(100),
    strength FLOAT DEFAULT 0.5,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_memory_rel_source ON mem0.memory_relationships(source_memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_rel_target ON mem0.memory_relationships(target_memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_rel_type ON mem0.memory_relationships(relationship_type);

-- ============================================================
-- ANALYTICS & MONITORING
-- ============================================================

-- Query performance tracking
CREATE TABLE IF NOT EXISTS analytics.query_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255),
    query_type VARCHAR(50),  -- rag, memory, embedding
    query_text TEXT,
    latency_ms INTEGER,
    result_count INTEGER,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_query_logs_user_id ON analytics.query_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_query_logs_type ON analytics.query_logs(query_type);
CREATE INDEX IF NOT EXISTS idx_query_logs_created_at ON analytics.query_logs(created_at);

-- ============================================================
-- FUNCTIONS & TRIGGERS
-- ============================================================

-- Update updated_at timestamp automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply to tables
CREATE TRIGGER update_rag_file_embeddings_updated_at BEFORE UPDATE ON rag.file_embeddings FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_rag_files_updated_at BEFORE UPDATE ON rag.files FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_mem0_memories_updated_at BEFORE UPDATE ON mem0.memories FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA rag TO librechat;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA rag TO librechat;
GRANT ALL PRIVILEGES ON SCHEMA mem0 TO librechat;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA mem0 TO librechat;
GRANT ALL PRIVILEGES ON SCHEMA analytics TO librechat;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO librechat;

-- ============================================================
-- INITIAL DATA / SEED (Optional)
-- ============================================================

-- Example: Pre-populate categories
INSERT INTO mem0.memories (user_id, category, memory_text, importance)
VALUES
    ('system', 'user_preferences', 'System initialization complete', 0.1),
    ('system', 'technical_knowledge', 'Local AI stack powered by LibreChat + mem0 + Qdrant + Ollama', 0.5)
ON CONFLICT DO NOTHING;
