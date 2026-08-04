# P2P KV Cache Sharing (multiprocess mode)

In a multi-node deployment, every node runs its own LMCache server, and each
one only caches the KV of the requests it served. **P2P KV cache sharing turns
those per-node caches into one logical cache:** when a node looks up a prefix
it does not have locally, it reads that prefix's KV directly from the memory of
the peer that holds it, over the datacenter network using RDMA — instead of
recomputing the prefix.

This example shows how to run it both on a single node (for testing/debugging)
and across multiple nodes, and what logs to expect.

> See also the full reference: [`docs/source/mp/p2p.rst`](../../docs/source/mp/p2p.rst).

## How it works

- **Coordinator** — one small HTTP service per deployment that tracks which
  LMCache servers are alive. It only manages membership; it never sees KV data.
- **LMCache server** — each server polls the coordinator for live peers and, for
  every peer, opens a connection used to look up and RDMA-read that peer's KV.
- **vLLM** talks only to its **local** LMCache server (via `LMCacheMPConnector`).
  The P2P fetch happens inside the LMCache server, transparently to vLLM.

## Requirements

- LMCache (full install, with CUDA) + vLLM, on every node.
- A **coordinator** URL reachable by all nodes (P2P refuses to start without
  `--coordinator-url`).
- An **RDMA-capable network** (InfiniBand / RoCE) for production performance.
- A **single, contiguous L1 region** — P2P is incompatible with the GDS
  (`--gds-l1-path`) and Device-DAX (`--l1-devdax-path`) L1 tiers; the server
  refuses to start in those configurations.
- Recommended: `--l1-align-bytes 65536` (64 KB) on P2P servers for larger,
  better-aligned RDMA reads.

---

## Single-node setup (testing & debugging)

Run two LMCache servers and two vLLM servers over `localhost`, plus one
coordinator, on a 2-GPU machine. This exercises the entire P2P path without a
real network.

Open five terminals:

```bash
# Terminal 1 — coordinator
lmcache coordinator --host 0.0.0.0 --port 9300

# Terminal 2 — node A: LMCache server
lmcache server \
    --host 127.0.0.1 --port 6555 --http-port 7555 \
    --l1-size-gb 50 --eviction-policy LRU \
    --l1-align-bytes 65536 \
    --instance-id node-a \
    --coordinator-url http://127.0.0.1:9300 \
    --coordinator-advertise-ip 127.0.0.1 \
    --p2p-advertise-url 127.0.0.1:8555

# Terminal 3 — node A: vLLM on GPU 0, connector -> local LMCache (port 6555)
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-14B --port 8000 \
    --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"lmcache.mp.port":6555}}'

# Terminal 4 — node B: LMCache server
lmcache server \
    --host 127.0.0.1 --port 6556 --http-port 7556 \
    --l1-size-gb 50 --eviction-policy LRU \
    --l1-align-bytes 65536 \
    --instance-id node-b \
    --coordinator-url http://127.0.0.1:9300 \
    --coordinator-advertise-ip 127.0.0.1 \
    --p2p-advertise-url 127.0.0.1:8556

# Terminal 5 — node B: vLLM on GPU 1, connector -> local LMCache (port 6556)
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-14B --port 8001 \
    --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"lmcache.mp.port":6556}}'
```

The two LMCache servers must differ in **every** port: ZMQ (`--port`), HTTP
(`--http-port`), and the P2P transfer endpoint (`--p2p-advertise-url`). Give
each a distinct `--instance-id`.

> **Note:** on a single host, `localhost` traffic uses the loopback/TCP path,
> not RDMA, so latencies are not representative of a real fabric. Single-node
> mode is for **functional** testing — benchmark on a real multi-node RDMA
> deployment.

### Test it

```bash
# 1. Populate node A's cache (cold — expect ~0 LMCache hits).
python send_request.py --port 8000

# 2. Send the SAME prompt to node B. B never served it and its cache is empty,
#    so any LMCache hit must have been read from A over P2P.
python send_request.py --port 8001
```

The second call should print a non-zero `num_lmcache_cached_tokens`.

---

## Multi-node setup

Start the coordinator on a host all nodes can reach (here `10.0.0.1`), then run
one LMCache server + one vLLM per node. Adding more nodes is just more copies of
the per-node block, all pointing at the same coordinator.

```bash
# Coordinator host (10.0.0.1)
lmcache coordinator --host 0.0.0.0 --port 9300
```

On **each** node, set `NODE_IP` to that node's routable address and run:

```bash
NODE_IP=10.0.0.2          # this node's address (change per node)
COORDINATOR=10.0.0.1

# RDMA tuning for UCX (adjust transports/rails to your fabric).
export UCX_TLS=rc,sm,self
export UCX_MAX_RMA_RAILS=8

# 1. LMCache server with P2P enabled. Bind 0.0.0.0, advertise NODE_IP.
lmcache server \
    --host 0.0.0.0 --port 6555 --http-port 7555 \
    --l1-size-gb 100 --eviction-policy LRU \
    --l1-align-bytes 65536 \
    --instance-id "lmcache-${NODE_IP}" \
    --coordinator-url "http://${COORDINATOR}:9300" \
    --coordinator-advertise-ip "${NODE_IP}" \
    --p2p-advertise-url "${NODE_IP}:8555" \
    --p2p-listen-url 0.0.0.0:8555

# 2. vLLM, connector pointed at the LOCAL LMCache server (port 6555).
vllm serve Qwen/Qwen3-14B --port 8000 \
    --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"lmcache.mp.port":6555}}'
```

`--coordinator-advertise-ip` is the address peers reach this node's control
plane at; `--p2p-advertise-url` is its RDMA transfer endpoint. Binding `0.0.0.0`
while advertising `NODE_IP` (via `--p2p-listen-url`) is the usual pattern when a
node has multiple interfaces.

Once two nodes are up, send a long prompt to one node and the same prompt to
another — the second serves it from the first over RDMA:

```bash
python send_request.py --host 10.0.0.2 --port 8000
python send_request.py --host 10.0.0.3 --port 8000
```

---

## What logs to expect

**Coordinator** logs each instance as it registers:

```
Registered instance node-a at 127.0.0.1:7555
Registered instance node-b at 127.0.0.1:7556
```

**Each LMCache server** at startup and as it discovers its peer (INFO level):

```
Started PeriodicThread: p2p-controller-thread (level=medium, interval=5.0s, init_wait=0.0s)
Registered with coordinator as node-a
Added L2 adapter 0 (p2p)          # created when the peer is discovered
```

With `LMCACHE_LOG_LEVEL=DEBUG`, the P2P controller also names the peer:

```
Added P2P adapter 0 for peer node-b (127.0.0.1:8556)
```

When a peer leaves (or after ~3 missed discovery polls), you'll see the adapter
torn down:

```
Deleted L2 adapter 0
Removed P2P adapter for peer node-b   # DEBUG
```

## Verifying P2P is working

Query a server's status endpoint for its P2P state and connected peers:

```bash
curl -s http://127.0.0.1:7555/status | python3 -m json.tool
# look for: "p2p_state": "registered", "p2p_peer_count": 1, "p2p_peers": ["node-b"]
```

List the fleet from the coordinator:

```bash
curl -s http://127.0.0.1:9300/instances | python3 -m json.tool
```

A successful P2P read shows up as a non-zero `num_lmcache_cached_tokens` from
`send_request.py` on a node that never served the prompt (see the test steps
above).

## Notes

- **Read-only:** a node reads KV from its peers; it never writes into a peer's
  memory.
- **One hop:** a node reads directly from the peer that holds the prefix; reads
  are not chained across multiple peers.
