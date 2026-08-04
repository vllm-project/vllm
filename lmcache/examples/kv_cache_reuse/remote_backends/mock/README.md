# Mock Remote Connector

LMCache provides a mock remote connector that allows you to manually set the peeking latency, read throughput, and write throughput inside of the remote url. It will create copies of your KV cache in unmanaged local RAM.


Deploy a serving engine with the mock remote backend: 
```bash
LMCACHE_CONFIG_FILE=example.yaml vllm serve meta-llama/Llama-3.1-8B-Instruct --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' --disable-log-requests --no-enable-prefix-caching
```

Check the retrieval (storing is async so the throughput there is meaningless) logs on the second query to confirm that the throughput is slightly lower than 2 GB / s (the CPU <-> GPU allocation / transfer also has overhead).

```bash
curl -X POST http://localhost:8000/v1/completions   -H "Content-Type: application/json"   -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "'"$(printf 'Elaborate the significance of KV cache in language models. %.0s' {1..1000})"'",
    "max_tokens": 10
  }'
```

Logs: 
```text
(EngineCore_0 pid=586318) [2025-09-03 05:06:41,751] LMCache INFO: Reqid: cmpl-b34e7c5b2f3e46a592722db2c27f6fc0-0, Total tokens 12002, LMCache hit tokens: 12002, need to load: 12001 (vllm_v1_adapter.py:1049:lmcache.integration.vllm.vllm_v1_adapter)
(EngineCore_0 pid=586318) [2025-09-03 05:06:42,736] LMCache INFO: Retrieved 12002 out of total 12002 out of total 12002 tokens. size: 1.4651 gb, cost 980.6983 ms, throughput: 1.4939 GB/s; (cache_engine.py:503:lmcache.v1.cache_engine)
```