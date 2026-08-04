# LMCache Move/Migrate
This is an example to demonstrate how to move/migrate a request's KV cache across LMCacheEngines externally.

## Prerequisites
Your server should have at least 2 GPUs. [NIXL](https://github.com/ai-dynamo/nixl) is required to be installed.

This will use port 8000 and 8001 for 2 vllms and port 8500 and 8501 for the corresponding LMCache workers. Also, ports 8200, 8201, 8202 and 8203 are used for p2p KV cache transfer. The controller itself occupies port 9000, 8300 and 9400.

## Steps
1. Start two vllm engines at port 8000 and port 8001:

```bash
PYTHONHASHSEED=123 UCX_TLS=rc CUDA_VISIBLE_DEVICES=0 LMCACHE_CONFIG_FILE=instance1.yaml vllm serve meta-llama/Llama-3.1-8B-Instruct --gpu-memory-utilization 0.8 --port 8000 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```

```bash
PYTHONHASHSEED=123 UCX_TLS=rc CUDA_VISIBLE_DEVICES=1 LMCACHE_CONFIG_FILE=instance2.yaml vllm serve meta-llama/Llama-3.1-8B-Instruct --gpu-memory-utilization 0.8 --port 8001 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```

2. Start the lmcache controller at port 9000 and the monitor at port 9001:

```bash
PYTHONHASHSEED=123 lmcache_controller --host localhost --port 9000 --monitor-ports '{"pull": 8300, "reply": 8400}'
```

3. Send a request to vllm engine 1:  
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Explain the significance of KV cache in language models.",
    "max_tokens": 10
  }'
```

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
{"count":12,"tokens":[128000,849,21435,279,26431,315,85748,6636,304,4221,4211,13],"token_strs":null}
```

5. Move the request's KV cache from vllm engine 1's CPU to vllm engine 2's CPU using request's token ids:
```bash
curl -X POST http://localhost:9000/move \
  -H "Content-Type: application/json" \
  -d '{
    "old_position": ["lmcache_instance_1", "LocalCPUBackend"],
    "new_position": ["lmcache_instance_2", "LocalCPUBackend"],
    "tokens": [128000, 849, 21435, 279, 26431, 315, 85748, 6636, 304, 4221, 4211, 13]
  }'
```
You should be able to see a return message indicating the KV cache has started to be moved in the system:

```plaintext
{"num_tokens": 12, "event_id": "xxx"}
```

`num_tokens: 12` means that there are 12 tokens's KV cache are stored in the system. The returned `event_id` can be used to check the status of the move operation (this functionality is coming soon).
