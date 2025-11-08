# ⚡ Quick Start Guide

Get your powerful local AI stack running in **under 15 minutes**.

## 🎯 What You're Building

A complete local AI system with:
- **LibreChat**: ChatGPT-like interface with multi-agent support
- **mem0**: Persistent memory across conversations
- **Qdrant**: High-performance vector database
- **Ollama**: Local LLM inference (Granite4, 128K context)
- **RAG**: Chat with your documents
- **Reranker**: Precision result ranking

## 📋 Prerequisites Checklist

- [ ] Docker Desktop installed and running
- [ ] 16GB+ RAM (32GB recommended)
- [ ] 100GB+ free disk space
- [ ] Internet connection (for initial model downloads)

## 🚀 Installation (3 Commands)

### Step 1: Navigate to the Stack

```bash
cd /home/user/vllm/local-ai-stack
```

### Step 2: Run Automated Setup

```bash
./setup.sh
```

**What this does:**
1. ✅ Generates secure secrets
2. ✅ Starts infrastructure (Postgres, MongoDB, Qdrant, Redis)
3. ✅ Downloads AI models (granite4, qwen3-embedding, reranker)
4. ✅ Initializes Qdrant vector collections
5. ✅ Starts application services
6. ✅ Runs health check

**Duration:** ~10-15 minutes (model downloads are the slowest part)

### Step 3: Access LibreChat

```bash
# Open in your browser:
http://localhost:3080
```

## 🎨 First Use Tutorial

### 1. Create Your Account

1. Click **"Sign Up"**
2. Enter email and password
3. First account = admin user

### 2. Enable Memory

1. Click Settings (gear icon)
2. Navigate to **"Personalization"**
3. Toggle **"Memory"** ON
4. Test: Tell it "I prefer Python over JavaScript"
5. In a new chat, ask "What programming language do I prefer?"
   - It should remember!

### 3. Upload a Document (RAG)

1. Click the **paperclip icon** (📎)
2. Upload a PDF, DOCX, or TXT file
3. Wait for "✅ File processed"
4. Ask questions about the document
5. Notice the citations in responses

Example:
```
You: "Summarize the key points from this document"
AI: Based on the uploaded document... [with citations]
```

### 4. Create a Custom Agent

1. Settings → **Agents**
2. Click **"New Agent"**
3. Configure:
   ```yaml
   Name: Personal Assistant
   Model: granite4:latest
   Capabilities:
     ✅ File Search (RAG)
     ✅ Memory
     ✅ Web Search
     ✅ Artifacts
   System Prompt:
     "You are a helpful personal assistant.
      Remember user preferences and provide
      personalized responses."
   ```
4. Save and select in a new conversation

### 5. Test Multi-Agent Workflow

1. Select the **"Team Collaborator"** preset agent
2. Ask: "Create a marketing plan for a new AI product"
3. Watch it:
   - Break down the task
   - Delegate to specialized sub-agents
   - Compile results

### 6. Generate an Artifact

1. Ask: "Create a React component for a task list with checkboxes"
2. Watch the artifact render in the side panel
3. Edit the code live
4. Export when satisfied

## 🛠️ Useful Commands

### Check System Health

```bash
./scripts/health-check.sh
```

### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f librechat
docker compose logs -f ollama
docker compose logs -f rag_api
```

### Restart a Service

```bash
docker compose restart librechat
```

### Stop Everything

```bash
docker compose down
```

### Start Everything

```bash
docker compose up -d
```

### Pull New Models

```bash
# Add a new model
docker compose exec ollama ollama pull llama3:latest

# List installed models
docker compose exec ollama ollama list
```

### Check Qdrant Collections

```bash
# Via API
curl http://localhost:6333/collections

# Via Dashboard
open http://localhost:6333/dashboard
```

### Backup Your Data

```bash
# Quick backup script
mkdir -p backups/$(date +%Y%m%d)
docker run --rm \
  -v local-ai-stack_postgres_data:/data \
  -v $(pwd)/backups/$(date +%Y%m%d):/backup \
  alpine tar czf /backup/postgres.tar.gz /data

# Backup Qdrant
docker run --rm \
  -v local-ai-stack_qdrant_storage:/data \
  -v $(pwd)/backups/$(date +%Y%m%d):/backup \
  alpine tar czf /backup/qdrant.tar.gz /data
```

## 🧪 Example Use Cases

### 1. Research Assistant

**Scenario:** Analyzing multiple research papers

```
1. Upload 5-10 PDF research papers
2. Enable "Research Assistant" agent
3. Ask: "What are the common themes across these papers?"
4. Follow up: "Create a comparison table"
5. Agent uses RAG to extract info and generates artifact table
```

### 2. Code Project Helper

**Scenario:** Working on a Python project

```
1. Upload project documentation
2. Enable "Code Expert" agent
3. Tell it: "I'm building a REST API with FastAPI"
   → Saved to memory
4. Ask: "Show me the authentication pattern from the docs"
   → Uses RAG
5. Ask: "Write a middleware for logging"
   → Uses memory of your tech stack
6. Agent generates code artifact with live preview
```

### 3. Meeting Notes Processor

**Scenario:** Process meeting transcripts

```
1. Upload meeting transcript (TXT)
2. Ask: "Extract action items and assign them"
3. Ask: "What were the key decisions?"
4. All extracted info is saved to memory
5. Later: "What did we decide about the marketing campaign?"
   → Recalls from memory
```

### 4. Learning Companion

**Scenario:** Learning a new topic

```
1. Tell it: "I'm learning machine learning, focus on practical examples"
   → Saved to memory
2. Upload ML textbook chapters
3. Ask: "Explain gradient descent with an example"
   → Uses RAG for accuracy
4. Ask: "Generate a Python code example"
   → Executes in code interpreter
5. Memory persists across sessions
```

## 🔧 Troubleshooting Quick Fixes

### "Service is unhealthy"

```bash
# Restart the problematic service
docker compose restart [service_name]

# Check logs for errors
docker compose logs [service_name]
```

### "Model not found"

```bash
# Re-pull models
./scripts/pull-models.sh
```

### "Out of memory"

```bash
# Reduce loaded models
# Edit docker-compose.yml:
OLLAMA_MAX_LOADED_MODELS: 1  # Instead of 3

# Restart
docker compose restart ollama
```

### "RAG not finding documents"

```bash
# Re-index
docker compose restart rag_api

# Check Qdrant collection
curl http://localhost:6333/collections/librechat_rag
```

### "Memory not working"

```bash
# Restart mem0
docker compose restart openmemory

# Check collection
curl http://localhost:6333/collections/mem0_memories
```

### "Can't access LibreChat"

```bash
# Check Nginx
docker compose logs nginx

# Direct access (bypass Nginx)
open http://localhost:3080
```

## 🎓 Next Steps

1. **Read** `FEATURES.md` for advanced configurations
2. **Explore** agent presets and customization
3. **Set up** web search (add Tavily API key to librechat.env)
4. **Enable** code interpreter (subscribe at code.librechat.ai)
5. **Configure** MCP servers for GitHub, filesystem access
6. **Tune** performance based on your hardware
7. **Backup** regularly (see README.md)

## 📊 Performance Expectations

### Without GPU (CPU-only)
- First response: 3-5 seconds
- Streaming: 15-25 tokens/second
- RAG search: 200-400ms
- Memory recall: 50-100ms

### With GPU (RTX 4090)
- First response: 0.5-1 second
- Streaming: 80-120 tokens/second
- RAG search: 100-200ms
- Memory recall: 30-50ms

## 🆘 Getting Help

1. **Check logs:** `docker compose logs -f`
2. **Run health check:** `./scripts/health-check.sh`
3. **Review** README.md for detailed troubleshooting
4. **Check** service status: `docker compose ps`

## 🎉 You're All Set!

You now have a **production-grade, privacy-first, cost-free** AI stack running locally. Enjoy! 🚀

**Key URLs to Bookmark:**
- LibreChat: http://localhost:3080
- Qdrant Dashboard: http://localhost:6333/dashboard
- Ollama API: http://localhost:11434

---

**Pro Tip:** Start with simple queries and progressively test more complex features. The system learns and improves as you use it thanks to the memory system!
