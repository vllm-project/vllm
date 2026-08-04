## LMCache can use [Valkey](https://valkey.io/) as an L2 backend (MP mode).

Offloads KV cache to a Valkey standalone or cluster via the `valkey` L2
adapter (zero-copy through the `valkey-glide-sync` client). In MP mode
vLLM talks to a separate LMCache server over ZMQ, which stores to Valkey.
For the non-MP `valkey://` connector instead, see
[`VALKEY_CONNECTOR_BENCHMARKING.md`](VALKEY_CONNECTOR_BENCHMARKING.md).

Requires a GPU and `pip install 'valkey-glide-sync>=2.3.0'`.

## Step 1: Start a local Valkey

For this example we run Valkey **locally on this machine** (alongside
vLLM). In production, point `startup_nodes` at your managed/remote Valkey
instead and skip this step.

```bash
# Local standalone
valkey-server --port 6390 --save "" --appendonly no --daemonize yes

# Or a local 6-node cluster (3 primaries + 3 replicas, all on localhost)
for p in 7000 7001 7002 7003 7004 7005; do
  valkey-server --port $p --cluster-enabled yes --cluster-config-file nodes-$p.conf \
    --save "" --appendonly no --daemonize yes
done
valkey-cli --cluster create 127.0.0.1:{7000,7001,7002,7003,7004,7005} \
  --cluster-replicas 1 --cluster-yes
```

## Step 2: Start the LMCache MP server

```bash
# Standalone (use the cluster line below for a cluster)
lmcache server --l1-size-gb 4 --eviction-policy LRU --chunk-size 16 --port 6555 \
  --l2-adapter '{"type":"valkey","startup_nodes":"localhost:6390","num_workers":8}'

# Cluster
#   --l2-adapter '{"type":"valkey","cluster_mode":true,"startup_nodes":"127.0.0.1:7000,127.0.0.1:7001","num_workers":8}'
```

`--l2-adapter` is a JSON object with `type:"valkey"`, a
`"host:port[,host:port]"` `startup_nodes` string, and optional
`cluster_mode`, `username`/`password`, `key_prefix`, `tls_enable`,
`num_workers`, `max_capacity_gb`. Run `lmcache server --help` for the full
field reference.

## Step 3: Start vLLM with the MP connector

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":6555}}' \
  --no-enable-prefix-caching --load-format dummy --port 8000
```

## Step 4: Send requests

Send the same prompt twice — the first stores KV cache to Valkey, the
second retrieves it:

```bash
curl -X POST http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "'"$(printf 'Elaborate the significance of KV cache in language models. %.0s' {1..1000})"'",
    "max_tokens": 10
  }'
```

## Confirming the offload worked

Watch vLLM's periodic engine logs for **`External prefix cache hit
rate`** — the fraction of prefix KV served from LMCache's L2 (Valkey). It
is `0.0%` on the first request and rises on repeats.

Confirm data physically landed in Valkey:

```bash
valkey-cli -p 6390 dbsize       # standalone
valkey-cli -c -p 7000 dbsize    # cluster
```
