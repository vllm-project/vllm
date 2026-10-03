# UMBP Distributed Mode

Distributed mode stores offloaded KV cache in a pool that spans hosts. A
`umbp_master` process indexes every object in the cluster and routes puts.
Every vLLM worker is a pool node: it owns a slice of host DRAM and serves it to
the other nodes over MORI-IO RDMA. A prefix stored by any engine attached to
the master can be restored by any other engine, on the same host or another
one.

```text
host 1                                   host 2
engine A worker ── DRAM slice ◄─ RDMA ─► DRAM slice ── engine B worker
      │  put/get routing                        │
      └────────────► umbp_master ◄──────────────┘
                         ▲  lookup (batch_exists)
         scheduler processes of every engine
```

A worker writes into its own pool or, when the master places the object
elsewhere, into a peer's pool; it reads local objects with a device copy and
remote ones over RDMA through a host staging arena. Scheduler lookups ask the
master through a client that owns no pool, runs no transfer engine, and serves
nothing, so the scheduler process allocates no DRAM and the master never
places an object there.

## Running the master

Run one master per cluster, independently of the engines:

```bash
MORI=$(python -c "import mori, os; print(os.path.dirname(mori.__file__))")
UMBP_ROUTE_PUT_NODE_AFFINITY=local "$MORI/umbp_master" 0.0.0.0:15558 9091
```

The arguments are the listen address and the Prometheus metrics port.
`UMBP_ROUTE_PUT_NODE_AFFINITY=local` makes the master place a put on the
writer's own node while it has room, so stores are local copies and only reads
cross the network. Without it the master places each object on the node with
the most free space. `UMBP_HEARTBEAT_TTL_SEC` (default 10) times
`UMBP_MAX_MISSED_HEARTBEATS` (default 3) is how long a crashed node's objects
stay indexed; see [Failure behavior](#failure-behavior).

## Host prerequisites

Every engine host needs:

- RDMA devices visible to the engine: in a container, `/dev/infiniband`, an
  unlimited `memlock` ulimit, and a verbs provider matching the host driver.
  An image whose provider does not match fails at worker startup with
  `no active RDMA device on this host`.
- **Hugepages for pools larger than about 1 GiB per worker**, on NICs with a
  small memory-translation cache such as AMD Pollara (ionic). There, one
  process could not register 2 GiB of 4 KiB-paged host memory for RDMA. The
  failure is not visible at startup: every transfer to or from another node
  fails instead, and requests recompute. Reserve `vm.nr_hugepages` for the sum
  of the pools on the host and set `dram_use_hugepages: true`. MORI falls back
  to 4 KiB pages with a warning when no hugepages are free.
- `peer_service_port + N` open between hosts, for every physical GPU `N` that
  runs a worker.

## Connector configuration

```bash
vllm serve <model> \
  --enable-prefix-caching \
  --kv-transfer-config '{
    "kv_connector": "UMBPStoreConnector",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "recompute",
    "kv_connector_extra_config": {
      "mode": "distributed",
      "master_address": "10.0.0.1:15558",
      "peer_service_port": 17100,
      "capacity_bytes": 68719476736,
      "dram_use_hugepages": true,
      "load_async": true
    }
  }'
```

The same configuration can be used on every host.

| Option | Default | Meaning |
| --- | --- | --- |
| `master_address` | required | Master `host:port`. |
| `peer_service_port` | required | Base port of the per-worker peer service. Each worker binds this plus its physical GPU index. |
| `node_address` | vLLM host IP | Address peers use to reach this host. Honors `VLLM_HOST_IP`. |
| `io_engine_host` | `node_address` | Listener host of the MORI-IO engine. |
| `io_engine_port` | any free port | Base port of the MORI-IO engine, offset like `peer_service_port`. |
| `node_id` | `vllm` | Prefix of the identities workers register with. |
| `capacity_bytes` | 64 GiB | Pool size per worker. |
| `total_capacity_bytes` | | Pool size per engine, split evenly across its ranks. Exclusive with `capacity_bytes`. |
| `dram_use_hugepages`, `dram_hugepage_size`, `dram_numa_node`, `dram_prefault` | | Allocation of each worker's pool. |
| `dram_page_size` | master default (2 MiB) | Allocation granularity. An object occupies whole pages, so pick a divisor of the KV object size; a warning reports padding above 25%. |
| `ranged_scratch_size` | 128 MiB | Size of each of the two host staging arenas for off-node transfers. Must hold one KV object (one block across all layers); startup fails otherwise. |
| `cache_remote_fetches`, `ranged_locality_prefetch` | `true` | Keep a copy of objects read from a peer in the reader's pool, making it another replica. |
| `local_first` | `true` | Answer from the local pool before asking the master. |
| `staging_buffer_size`, `backend_policy_path` | | Passed through to MORI. |
| `lookup_timeout_ms` | `2000` | Longest a scheduler lookup waits for the master before reporting misses. |
| `num_workers`, `timeout_ms` | `4`, `30000` | Transfer threads per worker, and how long a transfer may run before the worker treats the pool as stalled. |
| `load_failure_quarantine_ms` | `30000` | How long an object that failed to load is treated as a miss. |
| `load_async`, `lookup_async`, `lazy_offload`, `key_namespace` | | Shared connector options, as in embedded mode. |

`dram_use_shared_memory`, `dram_shm_name`, and the `dram_*_watermark` options
are rejected: the master decides eviction, and each pool is private to its
worker.

Workers register as `<node_id>-<hostname>-gpu<N>-<pid>-<seq>`. The process id
keeps a restarted engine from colliding with its predecessor's identity, which
the master keeps until that predecessor's heartbeats expire.

## Sharing scope

Objects are keyed exactly as in the other modes: model, revision, KV layout,
cache groups, TP/PCP/DCP topology, and the vLLM block hash. Engines with the
same model and layout share objects regardless of host or data-parallel rank;
each TP rank reads its own shard. `key_namespace` overrides the derived
namespace and is the way to separate weight versions that share a model path.

## Failure behavior

Set `kv_load_failure_policy` to `recompute`. The pool is best effort, and a
load can fail after a lookup hit when the node holding the object has died;
with the default `fail` policy that request fails instead of recomputing.

| Event | Behavior |
| --- | --- |
| Master not reachable at engine startup | Engine startup fails. |
| Master dies while engines run | Lookups report misses, loads and stores fail, and requests recompute. Engines keep serving. |
| Master hangs, or its host stops answering | MORI's lookup and routing calls have no deadline and block until the master answers. A lookup gives up after `lookup_timeout_ms` and reports misses; later lookups report misses at once until that call returns. A store that outlives `timeout_ms` stays pending with its source blocks pinned until MORI returns, and new loads and stores fail immediately meanwhile. Requests recompute. |
| Master restarts at the same address | Live nodes re-register on their next heartbeat and resend their objects; hits resume without restarting engines, after gRPC's reconnect backoff. |
| Engine stops cleanly | Its workers unregister; objects in their pools leave the index. Copies other nodes made remain. |
| Engine crashes | Its objects stay indexed until the master expires the node. The first load of each fails and recomputes; the object is then treated as a miss for `load_failure_quarantine_ms`, so requests do not retry it. |
| Engine restarts | It joins at once under new identities and reads the rest of the pool. |
| `POST /reset_prefix_cache?reset_external=true` | Reports failure and clears nothing: the pool is shared by every attached engine, and MORI has no cluster-wide clear. |

Keep `load_async` enabled (the default). An asynchronous load is polled
without a timeout, so it never gives up on a MORI call that may still write
into its blocks. A synchronous wait on a load that outlives `timeout_ms`
raises and fails the step rather than letting vLLM reuse those blocks.

## Limitations

- Pipeline parallelism is rejected, as in the other modes.
- Layer-wise store is disabled; layer-wise load is supported.
- The master is a single, non-replicated index. Its loss degrades to
  recomputation, not to errors.
- The first transfer between two nodes sets up their RDMA connection and took
  about 2 s in validation.
- Master-side evictions do not produce KV removal events.
- There is no per-engine quota; engines can evict each other's objects.
