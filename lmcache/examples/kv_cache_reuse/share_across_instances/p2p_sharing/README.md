# P2P KV Cache Sharing
This is an example to demonstrate P2P KV cache sharing.
## Prerequisites
Your server should have at least 2 GPUs.
[NIXL](https://github.com/ai-dynamo/nixl) should be installed as well. 

The LMCache controller will use port 8300 to pull messages from LMCache workers and port 8400 to reply requests to LMCache workers.

The two LMCache workers will use the port 8010 and 8011 for 2 vllms, port 8200 and 8202 for p2p initializations, and port 8201 and 8203 for p2p lookups.

## Steps
1. Start the LMCache controller:
```bash
PYTHONHASHSEED=123 lmcache_controller --host localhost --port 9000 --monitor-ports '{"pull": 8300, "reply": 8400}'
``` 

2. Start two vllm engines (each with an LMCache worker):

Start vllm engine 1 at port 8010:
```bash
PYTHONHASHSEED=123 UCX_TLS=rc CUDA_VISIBLE_DEVICES=0 LMCACHE_CONFIG_FILE=example1.yaml vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --gpu-memory-utilization 0.8 --port 8010 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```
Start vllm engine 2 at port 8011:
```bash
PYTHONHASHSEED=123 UCX_TLS=rc CUDA_VISIBLE_DEVICES=1 LMCACHE_CONFIG_FILE=example2.yaml vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct  --gpu-memory-utilization 0.8 --port 8011 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'  
```


3. Send request to vllm engine 1:  
```bash
curl -X POST http://localhost:8010/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"meta-llama/Meta-Llama-3.1-8B-Instruct\",
    \"prompt\": \"$(printf 'Explain the significance of KV cache in language models.%.0s' {1..100})\",
    \"max_tokens\": 10
  }"
```

4. Send request to vllm engine 2:  
```bash
curl -X POST http://localhost:8011/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"meta-llama/Meta-Llama-3.1-8B-Instruct\",
    \"prompt\": \"$(printf 'Explain the significance of KV cache in language models.%.0s' {1..100})\",
    \"max_tokens\": 10
  }"
```
The cache will be automatically retrieved from vllm engine 1.
You should be able to see logs (from vllm engine 2) like the following:
```bash
(EngineCore_DP0 pid=2577584)[2025-09-21 00:00:11,706] LMCache INFO:[0m Established connection to peer_init_url localhost:8200. The peer_lookup_url: localhost:8201 (p2p_backend.py:278:lmcache.v1.storage_backend.p2p_backend)
(EngineCore_DP0 pid=2577584)[2025-09-21 00:00:11,792] LMCache INFO: Retrieved 1002 out of total 1002 out of total 1002 tokens. size: 0.1223 gb, cost 60.3595 ms, throughput: 2.0264 GB/s; (cache_engine.py:496:lmcache.v1.cache_engine)
```
