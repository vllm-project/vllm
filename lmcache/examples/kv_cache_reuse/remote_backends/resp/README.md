# LMCacheRedis KV Connector

The cpp binding is in `csrc/redis/*`.

The key optimization is multi-threading and batching on the C layer and the python side being awoken through the eventfd API where a callback consumes the completion belonging to a non-blocking submission. 

An GET / SET benchmark is in `examples/kv_cache_reuse/remote_backends/resp/benchmark_resp_client.py`. The python client lives inside of `lmcache/v1/storage_backend/resp_client.py`.

Install lmcache from source, then run a sanity check:

```bash
# Run with defaults: host=127.0.0.1, port=6379, chunk-mb=4.0, num-workers=8, num-keys=500
python benchmark_resp_client.py

# Or customize parameters:
python benchmark_resp_client.py \
    --host localhost \
    --port 6379 \
    --chunk-mb 1.0 \
    --num-workers 8 \
    --num-keys 1280 \
    --username default \
    --password YOUR_PASSWORD
```

## Quickstart

Start up redis with multiple io threads: 
```bash
git clone https://github.com/redis/redis.git
cd redis
git checkout 8.2
make -j
./src/redis-server --protected-mode no --save '' --appendonly no --io-threads 4
```

Clear the state between queries
```bash
sudo apt install redis-cli
redis-cli -p 6379 FLUSHALL
redis-cli -p 6379 DBSIZE
```

Deploy LMCache with the custom LMCacheRedis KV Connector

`save_unfull_chunk` must be off (default is off) and also we must not save the chunk metadata. 

The "golden spot" for high throughput transfers for redis is ~4 MB (any higher or lower will cause performance degradation, for a model like meta-llama/Llama-3.1-8B-Instruct, this is around 16 tokens
```bash
LMCACHE_CONFIG_FILE=resp.yaml \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
    --no-enable-prefix-caching \
    --load-format dummy
```

## MP Mode (Multiprocess)

MP mode runs LMCache as a separate server process, communicating with vLLM over
ZMQ. The RESP connector serves as an L2 adapter, supporting variable-size
chunks. See [`csrc/storage_backends/README.md`](../../../../csrc/storage_backends/README.md)
for the full native backend architecture.

### Launch Redis

```bash
# Build Redis 8.2 with IO threads
git clone https://github.com/redis/redis.git && cd redis
git checkout 8.2 && make -j
./src/redis-server --protected-mode no --save '' --appendonly no --io-threads 4 --port 6379
```

### Launch LMCache MP Server

```bash
lmcache server \
    --l1-size-gb 10 \
    --eviction-policy LRU \
    --chunk-size 16 \
    --l2-adapter '{"type": "resp", "host": "localhost", "port": 6379, "num_workers": 8}' \
    --port 6555
```

The `--l2-adapter` JSON accepts these fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | str | (required) | Must be `"resp"` |
| `host` | str | (required) | Redis hostname |
| `port` | int | (required) | Redis port |
| `num_workers` | int | 8 | C++ worker threads for parallel I/O |
| `username` | str | `""` | Redis ACL username |
| `password` | str | `""` | Redis AUTH password |
| `max_capacity_gb` | float | 0 | Max L2 capacity in GB for usage tracking (required for L2 eviction) |

### Launch vLLM with LMCache MP Connector

```bash
PORT=8000
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --kv-transfer-config '{
        "kv_connector": "LMCacheMPConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "lmcache.mp.host": "tcp://localhost",
            "lmcache.mp.port": 6555
        }
    }' \
    --no-enable-prefix-caching \
    --port $PORT \
    --load-format dummy
```

### Test (MP and non-MP mode)

Send the same prompt twice. The first request stores KV cache to Redis via the
MP server; the second retrieves it.

```bash
PORT=8000
PROMPT="$(printf 'Elaborate the significance of KV cache in language models. %.0s' {1..1000})"

# First request: store
curl -s -X POST http://localhost:${PORT}/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","prompt":"'"$PROMPT"'","max_tokens":10}'

# Second request with same prefix: retrieve from Redis
curl -s -X POST http://localhost:${PORT}/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","prompt":"'"$PROMPT"'","max_tokens":10}'
```

Check Redis to verify data was stored:
```bash
redis-cli -p 6379 DBSIZE
```
