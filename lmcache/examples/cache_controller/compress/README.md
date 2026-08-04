# LMCache Compress
This is an example to demonstrate how to compress or decompress a request's KV cache externally.

## Prerequisites
Your server should have at least 1 GPU.

This will use port 8000 for vllm and port 8001 for the LMCache worker. The controller itself occupies port 9000 and 9001.

## Steps
1. Start vllm engine at port 8000

```bash
CUDA_VISIBLE_DEVICES=0 LMCACHE_CONFIG_FILE=example.yaml vllm serve meta-llama/Llama-3.1-8B-Instruct --gpu-memory-utilization 0.8 --port 8000 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
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

LMCache will automatically offloads the KV cache to CPU.

4. Tokenize the prompt:  
```bash
curl -X POST http://localhost:8000/tokenize \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Explain the significance of KV cache in language models."
  }'
```

You should be able to see the returned token ids as:
```plaintext
{"count":12,"max_model_len":4096,"tokens":[128000,849,21435,279,26431,315,85748,6636,304,4221,4211,13],"token_strs":null}
```

5. Using Cachegen to compress request's KV cache:
```bash
curl -X POST http://localhost:9000/compress \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "lmcache_default_instance",
    "method": "cachegen",
    "location": "LocalCPUBackend",
    "tokens": [128000, 849, 21435, 279, 26431, 315, 85748, 6636, 304, 4221, 4211, 13]
  }'
```
You should be able to see a return message indicating the KV cache has started to be compressed

```plaintext
{"num_tokens": 12, "event_id": "xxx"}
```

`num_tokens: 12` means that there are 12 tokens's KV cache are being compressed in the system. The returned `event_id` can be used to check the status of the compress operation (this functionality is coming soon).

6. Using Cachegen to decompress request's KV cache:
```bash
curl -X POST http://localhost:9000/decompress \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "lmcache_default_instance",
    "method": "cachegen",
    "location": "LocalCPUBackend",
    "tokens": [128000, 849, 21435, 279, 26431, 315, 85748, 6636, 304, 4221, 4211, 13]
  }'
```
You should be able to see a return message indicating the KV cache has started to be decompressed

```plaintext
{"num_tokens": 12, "event_id": "xxx"}
```

`num_tokens: 12` means that there are 12 tokens's KV cache are being decompressed in the system. The returned `event_id` can be used to check the status of the decompress operation .
