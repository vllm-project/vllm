# Cache with configs
This is an example to cache with configs, includes tags and the other configs.
- tags will be used to generate the key
- configs will be used to interact with the backends, such as set the ttl

## Prerequisites
Your server should have at least 1 GPU.

This will use the port 8000 for 1 vllm.

## Steps
1. Start the vllm engine at port 8000:

```bash
VLLM_USE_V1=1 \
LMCACHE_TRACK_USAGE=false \
LMCACHE_CONFIG_FILE=example.yaml \
vllm serve /disc/f/models/opt-125m/ \
           --served-model-name "facebook/opt-125m" \
           --enforce-eager  \
           --port 8000 \
           --gpu-memory-utilization 0.8 \
           --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
           --trust-remote-code
```

2. Send a request to vllm engine with tags and configs by `kv_transfer_params: {}`:
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "prompt": "Explain the significance of KV cache in language models." * 100,
    "max_tokens": 10,
	"kv_transfer_params": {
	  "lmcache.tag.user": "example_user_1",
	  "lmcache.ttl": 60
	}
  }'
```
- set tags: use `lmcache.tag.xxx`
- set configs: use `lmcache.xxx`
