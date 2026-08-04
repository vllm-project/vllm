# LMCache Clear
This is an example to demonstrate how to clear KV cache in an LMCacheEngine externally.

## Prerequisites
Your server should have at least 1 GPU.  

This will use port 8000 for 1 vllm and port 8001 for LMCache. The controller occupies ports 9000 and 9001.

## Steps
1. Start the vllm engine at port 8000:

```bash
CUDA_VISIBLE_DEVICES=0 LMCACHE_CONFIG_FILE=example.yaml vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 4096  --gpu-memory-utilization 0.8 --port 8000 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```

2. Start the lmcache controller at port 9000 and the monitor at port 9001:

```bash
lmcache_controller --host localhost --port 9000 --monitor-port 9001
```

3. Send a request to vllm engine:  
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Explain the significance of KV cache in language models.",
    "max_tokens": 10
  }'
```

4. Clear the KV cache in the system:
```bash
curl -X POST http://localhost:9000/clear \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "lmcache_default_instance",
    "location": "LocalCPUBackend"
  }'
```
You should be able to see a return message indicating the number of tokens' KV cache that has been successfully cleared in the system:

```plaintext
{"event_id": "xxx", "num_tokens": 12}
```
