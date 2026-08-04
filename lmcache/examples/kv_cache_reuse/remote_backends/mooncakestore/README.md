lmcache could use [mooncakestore](https://github.com/kvcache-ai/Mooncake) as a backend storage.

Mooncakestore is a memory storage which support RDMA and TCP. lmcache's mooncakestore connector can use TCP/RDMA transport for now.

This is a simple instruction how to use lmcache with mooncakestore

# install mooncakestore

```
pip install mooncake-transfer-engine
```

# start mooncake store and lmcache

1. start mooncake store

```
mooncake_master -v=1
mooncake_http_metadata_server --port 8005
```

2. start vllm with lmcache connector
```
LMCACHE_CONFIG_FILE="mooncake.yaml" \
vllm serve meta-llama/Llama-3.1-8B-Instruct
           --port 8000 \
           --gpu-memory-utilization 0.8 \
           --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
```