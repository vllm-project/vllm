# User Controllable Caching
This is an example to demonstrate user controllable caching (e.g., specify whether to cache a request or not).

## Prerequisites
Your server should have at least 1 GPU.  

This will use the port 8000 for 1 vllm.

## Steps
1. Start the vllm engine at port 8000:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --kv-transfer-config '{"kv_connector": "LMCacheConnectorV1","kv_role": "kv_both"}'
```

2. Send a request to vllm engine with caching enabled (default behavior):  

**Using `/v1/completions`:**
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Explain the significance of KV cache in language models.",
    "max_tokens": 10
  }'
```

**Using `/v1/chat/completions`:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "user", "content": "Explain the significance of KV cache in language models."}
    ],
    "max_tokens": 10
  }'
```

You should be able to see logs indicating the KV cache is stored:

```plaintext
INFO: Storing KV cache for 13 out of 13 tokens (skip_leading_tokens=0)
```

3. Send a request with caching disabled:

**Using `/v1/completions`:**
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "What is the weather today in Chicago?",
    "max_tokens": 10,
    "kv_transfer_params": {
      "lmcache.skip_save": true
    }
  }'
```

**Using `/v1/chat/completions`:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "user", "content": "What'"'"'s the weather today in Chicago?"}
    ],
    "max_tokens": 10,
    "kv_transfer_params": {
      "lmcache.skip_save": true
    }
  }'
```

You should be able to see logs indicating the KV cache is NOT stored:

```plaintext
INFO: User has specified not to store the cache (store_cache: false)
```

Note that cache is stored by default. 
To disable caching, pass `"kv_transfer_params": {"lmcache.skip_save": true}` in your request. 
This works for both `/v1/completions` and `/v1/chat/completions` endpoints.