# Powerful Local AI Stack

Complete local AI setup with LibreChat, mem0/OpenMemory, Qdrant, and Ollama for research, collaboration, brainstorming, and coding.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LibreChat (Frontend)                      │
│  Multi-Agent • Artifacts • Memory • RAG • Code Interpreter  │
└─────────────────┬───────────────────────────────────────────┘
                  │
     ┌────────────┼────────────┬────────────┐
     │            │            │            │
┌────▼────┐  ┌───▼────┐  ┌───▼──────┐ ┌──▼─────┐
│ Ollama  │  │  mem0  │  │ RAG API  │ │Reranker│
│ Models  │  │ Memory │  │ PGVector │ │Service │
└─────────┘  └───┬────┘  └────┬─────┘ └────────┘
                 │            │
            ┌────▼────────────▼────┐
            │   Qdrant (Vectors)   │
            └──────────────────────┘
```

## 🎯 Features

### LibreChat
- **Multi-Agent Orchestration**: Chain up to 10 specialized agents
- **Native Memory System**: Persistent context across conversations
- **Advanced RAG**: Semantic search with reranking
- **Code Interpreter**: Execute code in 10+ languages
- **Artifacts**: Generate React components, diagrams
- **Web Search**: Real-time information retrieval

### mem0/OpenMemory
- **Multi-Level Memory**: User, Session, Agent state
- **Semantic Search**: Context-aware memory retrieval
- **Self-Improving**: Continuous refinement
- **Privacy-First**: Fully local, no external data

### Qdrant
- **Hybrid Search**: Dense + sparse + filtering
- **Quantization**: 4-16x memory reduction
- **High Performance**: HNSW indexing

### Ollama Models
- **LLM**: granite4:latest (128K context)
- **Embedder**: qwen3-embedding:4b
- **Reranker**: Qwen3-Reranker-4B

## 📋 Prerequisites

- Docker 24.0+ with Docker Compose
- 16GB+ RAM (32GB recommended)
- 100GB+ free disk space
- (Optional) NVIDIA GPU with drivers for acceleration

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Create directory
mkdir -p local-ai-stack
cd local-ai-stack

# Copy all configuration files to this directory
# - docker-compose.yml
# - librechat.yaml
# - librechat.env
# - mem0-config.yaml
# - nginx.conf
# - init-scripts/01-init-databases.sql
# - reranker-service/
```

### 2. Configure Environment

```bash
# Edit librechat.env and change these values:
# - JWT_SECRET (random string, 32+ chars)
# - JWT_REFRESH_SECRET (random string, 32+ chars)
# - CREDS_KEY (exactly 32 characters)
# - CREDS_IV (exactly 16 characters)
# - Database passwords

# Generate random secrets:
openssl rand -hex 32  # For JWT secrets
openssl rand -hex 16  # For CREDS_KEY
openssl rand -hex 8   # For CREDS_IV
```

### 3. Pull Ollama Models

```bash
# Start only Ollama first
docker compose up -d ollama

# Wait for Ollama to be ready
docker compose exec ollama ollama pull granite4:latest
docker compose exec ollama ollama pull qwen3-embedding:4b
docker compose exec ollama ollama pull dengcao/Qwen3-Reranker-4B

# Verify models
docker compose exec ollama ollama list
```

### 4. Initialize Qdrant Collections

```bash
# Start Qdrant
docker compose up -d qdrant

# Wait 10 seconds for initialization
sleep 10

# Create collections (run the setup script)
./scripts/init-qdrant.sh
```

### 5. Start All Services

```bash
# Start everything
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f librechat
```

### 6. Access the Stack

- **LibreChat**: http://localhost:3080
- **Qdrant Dashboard**: http://localhost:6333/dashboard
- **Ollama API**: http://localhost:11434
- **mem0 API**: http://localhost:8080
- **RAG API**: http://localhost:8000

## 📚 Usage Guide

### 1. Register an Account

1. Navigate to http://localhost:3080
2. Click "Sign Up"
3. Create your account

### 2. Enable Memory

1. Open Settings
2. Enable "Personalization & Memory"
3. Start chatting - memories will be automatically saved

### 3. Use RAG (Chat with Files)

1. Click the paperclip icon
2. Upload documents (PDF, DOCX, TXT, etc.)
3. Files are automatically indexed
4. Ask questions about your documents

### 4. Create Agents

Use the built-in agent presets:

- **Research Assistant**: Web search + RAG + memory recall
- **Code Expert**: Code execution + documentation search
- **Team Collaborator**: Multi-agent workflow orchestration
- **Creative Brainstormer**: Artifacts + visual generation

Or create custom agents in Settings → Agents

### 5. Multi-Agent Chaining

1. Create a conversation with the "Collaborator" agent
2. It will automatically delegate to specialized sub-agents
3. Monitor the agent chain in the conversation

## 🔧 Advanced Configuration

### GPU Acceleration

Edit `docker-compose.yml`:

```yaml
ollama:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
  environment:
    OLLAMA_GPU_LAYERS: 35
```

### Memory Categories

Edit `mem0-config.yaml` to add custom categories:

```yaml
memory:
  categories:
    - user_preferences
    - technical_knowledge
    - project_context
    - custom_category_here
```

### RAG Chunk Size

Edit `librechat.env`:

```bash
CHUNK_SIZE=512  # Adjust based on your use case
CHUNK_OVERLAP=50
```

### Qdrant Quantization

For memory optimization, edit Qdrant collection configs in `scripts/init-qdrant.sh`:

```python
quantization_config = {
    "scalar": {
        "type": "int8",
        "quantile": 0.99
    }
}
```

## 🛠️ Troubleshooting

### Services Not Starting

```bash
# Check logs
docker compose logs [service_name]

# Common issues:
docker compose down
docker volume prune  # WARNING: Deletes data
docker compose up -d
```

### Ollama Models Not Loading

```bash
# Re-pull models
docker compose exec ollama ollama pull granite4:latest

# Check available models
docker compose exec ollama ollama list

# Restart Ollama
docker compose restart ollama
```

### RAG Not Working

```bash
# Check RAG API logs
docker compose logs rag_api

# Verify PostgreSQL connection
docker compose exec postgres psql -U librechat -c "SELECT version();"

# Re-initialize database
docker compose down postgres
docker volume rm local-ai-stack_postgres_data
docker compose up -d postgres
```

### Memory Not Persisting

```bash
# Check mem0 logs
docker compose logs openmemory

# Verify Qdrant connection
curl http://localhost:6333/collections/mem0_memories

# Reset memory store
docker compose restart openmemory
```

### Out of Memory

```bash
# Reduce concurrent models in docker-compose.yml
OLLAMA_MAX_LOADED_MODELS: 1  # Instead of 3

# Enable Qdrant quantization (see Advanced Configuration)

# Reduce chunk size for RAG
CHUNK_SIZE=256  # Instead of 512
```

## 📊 Monitoring

### View Resource Usage

```bash
docker stats
```

### Check Collection Status

```bash
# Qdrant collections
curl http://localhost:6333/collections

# PostgreSQL tables
docker compose exec postgres psql -U librechat -c "\dt rag.*"
```

### Performance Metrics

Check `analytics.query_logs` table:

```sql
docker compose exec postgres psql -U librechat -c "
  SELECT query_type, AVG(latency_ms), COUNT(*)
  FROM analytics.query_logs
  GROUP BY query_type;
"
```

## 🔒 Security Notes

### For Production Use

1. **Change all default passwords** in `librechat.env`
2. **Enable HTTPS** in nginx.conf
3. **Set up firewall rules**
4. **Enable authentication** on all services
5. **Use environment-specific secrets**

### Network Isolation

Services are isolated in backend/frontend networks. Only LibreChat is exposed via Nginx.

## 🗂️ Backup & Recovery

### Backup Data

```bash
# Backup volumes
docker run --rm -v local-ai-stack_postgres_data:/data -v $(pwd)/backups:/backup alpine tar czf /backup/postgres_backup.tar.gz /data
docker run --rm -v local-ai-stack_qdrant_storage:/data -v $(pwd)/backups:/backup alpine tar czf /backup/qdrant_backup.tar.gz /data
docker run --rm -v local-ai-stack_ollama_models:/data -v $(pwd)/backups:/backup alpine tar czf /backup/ollama_backup.tar.gz /data

# Backup configurations
tar czf backups/configs_backup.tar.gz *.yaml *.env *.conf
```

### Restore Data

```bash
# Stop services
docker compose down

# Restore volumes
docker run --rm -v local-ai-stack_postgres_data:/data -v $(pwd)/backups:/backup alpine tar xzf /backup/postgres_backup.tar.gz -C /
docker run --rm -v local-ai-stack_qdrant_storage:/data -v $(pwd)/backups:/backup alpine tar xzf /backup/qdrant_backup.tar.gz -C /
docker run --rm -v local-ai-stack_ollama_models:/data -v $(pwd)/backups:/backup alpine tar xzf /backup/ollama_backup.tar.gz -C /

# Start services
docker compose up -d
```

## 📈 Scaling

### Horizontal Scaling

For high-traffic scenarios:

1. **Qdrant**: Use cluster mode
2. **PostgreSQL**: Set up replication
3. **Ollama**: Run multiple instances behind load balancer
4. **LibreChat**: Scale with Docker Swarm or Kubernetes

### Vertical Scaling

- Increase container resource limits in `docker-compose.yml`
- Adjust PostgreSQL shared_buffers and effective_cache_size
- Tune Qdrant HNSW parameters (m, ef_construct)

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This setup is provided as-is. Individual components have their own licenses:

- LibreChat: MIT
- mem0: Apache 2.0
- Qdrant: Apache 2.0
- Ollama: MIT

## 🙏 Acknowledgments

- [LibreChat](https://github.com/danny-avila/LibreChat)
- [mem0](https://github.com/mem0ai/mem0)
- [Qdrant](https://github.com/qdrant/qdrant)
- [Ollama](https://github.com/ollama/ollama)
