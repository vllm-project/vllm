# Search LMCache KV Entry in Redis

This example shows how to search the LMCache KV entry in Redis.

## Installing Redis

### Ubuntu Installation

```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl status redis-server
```

### RHEL/CentOS Installation

```bash
sudo yum install redis
sudo systemctl start redis
sudo systemctl enable redis
sudo systemctl status redis
```

## Configuration Steps

### Create a LMCache Configuration File

Create a file `/tmp/lmcache-config.yaml` with Redis configuration:

```yaml
# Basic LMCache settings
chunk_size: 256
local_cpu: True
max_local_cpu_size: 5

# Redis connection
remote_url: "redis://your-redis-host:6379"
```

### Run the Container with Redis Support

```bash
docker run --runtime nvidia --gpus all \
    -v /tmp/lmcache-config.yaml:/config/lmcache-config.yaml \
    --env "LMCACHE_CONFIG_FILE=/config/lmcache-config.yaml" \
    --env "HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>" \
    --env "LMCACHE_CHUNK_SIZE=256" \
    --env "LMCACHE_LOCAL_CPU=True" \
    --env "LMCACHE_MAX_LOCAL_CPU_SIZE=5" \
    -v ~/.cache/huggingface:/home/ubuntu/.cache/huggingface \
    --network host \
    lmcache/vllm-openai:latest \
    mistralai/Mistral-7B-Instruct-v0.2 --port 8001 --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

  Then run the following command to query vLLM to populate the LMCache:
  ```bash
  curl -X 'POST' \
   'http://127.0.0.1:8001/v1/chat/completions' \
   -H 'accept: application/json' \
   -H 'Content-Type: application/json' \
   -d '{
      "model": "mistralai/Mistral-7B-Instruct-v0.2",
      "messages": [
      {"role": "system", "content": "You are a helpful AI coding assistant."},
      {"role": "user", "content": "Write a segment tree implementation in python"}
      ],
      "max_tokens": 150
   }'
  ```

## Viewing and Managing LMCache Entries in Redis

### LMCache Redis Keys

LMCache stores data in Redis using a structured key format. Each key contains the following information in a delimited format:

```
model_name@world_size@worker_id@chunk_hash
```

Where:
- `model_name`: Name of the language model
- `world_size`: Total number of workers in distributed deployment
- `worker_id`: ID of the worker that created this cache entry
- `chunk_hash`: Hash of the token chunk (SHA-256 based)

For example, a typical key might look like:
```
vllm@mistralai/Mistral-7B-Instruct-v0.2@1@0@a1b2c3d4e5f6...
```

### Using redis-cli to View LMCache Data

To inspect and manage LMCache entries in Redis:

#### Connect to Redis
   ```bash
   redis-cli -h localhost -p 6379
   ```

#### List all LMCache keys
   ```bash
   # Show all keys
   KEYS *

   # Show keys for a specific model
   KEYS *Mistral-7B*
   ```

  For example, to check if a key exists:
  ```console
  localhost:6379> KEYS *
  1) "vllm@mistralai/Mistral-7B-Instruct-v0.2@1@0@2aea46f4fa38170e8425a6e6ee3c5173a1fa97917bc1a583888c87ad4f9a9a20metadata"
  2) "vllm@mistralai/Mistral-7B-Instruct-v0.2@1@0@2aea46f4fa38170e8425a6e6ee3c5173a1fa97917bc1a583888c87ad4f9a9a20kv_bytes"
  ```

#### Check if a key exists
   ```bash
   EXISTS "vllm@model_name@1@0@hash_value"
   ```

#### View memory usage for a key
   ```bash
   MEMORY USAGE "vllm@model_name@1@0@hash_value"
   ```

#### Delete specific keys
   ```bash
   # Delete a single key
   DEL "vllm@model_name@1@0@hash_value"
   
   # Delete all keys matching a pattern
   redis-cli -h <host> -p <port> --scan --pattern "vllm@model_name*" | xargs redis-cli -h <host> -p <port> DEL
   ```

#### Monitor Redis in real-time
   ```bash
   MONITOR
   ```

#### Get Redis stats for LMCache
   ```bash
   # Get memory stats
   INFO memory
   
   # Get statistics about operations
   INFO stats
   ```

