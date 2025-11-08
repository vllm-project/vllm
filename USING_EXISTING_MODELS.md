# 🔧 Using Your Existing Ollama Models

## Where Are Your Ollama Models?

Find your models directory on your PC:

**Windows:**
```
C:\Users\YourUsername\.ollama\models
```

**Mac:**
```
~/.ollama/models
```

**Linux:**
```
~/.ollama/models
```

## How to Configure docker-compose.yml

### Before (Downloads new models):
```yaml
ollama:
  volumes:
    - ollama_models:/root/.ollama  # ← Creates new empty volume
```

### After (Uses your existing models):
```yaml
ollama:
  volumes:
    - C:/Users/YourUsername/.ollama:/root/.ollama  # Windows
    # OR
    - ~/.ollama:/root/.ollama  # Mac/Linux
```

## Step-by-Step:

1. **Find where your Ollama models are stored:**
   - Run in terminal: `ollama list`
   - Note the path (usually shown in Ollama settings)

2. **Clone this repo to your PC**

3. **Edit docker-compose.yml:**
   - Find line ~107 (the `ollama:` service)
   - Change the volume path to YOUR models directory
   - Example:
     ```yaml
     volumes:
       - /home/youruser/.ollama:/root/.ollama
     ```

4. **Run `./setup.sh`**
   - It will SKIP downloading models (already there!)
   - Saves ~15GB download and time

## Your Existing Models:

You said you have:
- ✅ granite4:latest
- ✅ qwen3-embedding:4b
- ✅ dengcao/Qwen3-Reranker-4B (or need to pull this one?)

The stack will use these immediately!

## Docker Volume vs Your Directory:

**Docker Volume (default):**
- Isolated storage managed by Docker
- Have to re-download models

**Your Directory (recommended for you):**
- Uses your existing models
- Faster startup
- No re-download
- Can update models outside Docker too

---

**Ready to create the GitHub repo?** Tell me:
1. What's the path to your Ollama models?
2. What do you want to name the repo? (e.g., "local-ai-stack")
