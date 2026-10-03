# UMBP Standalone Mode

Standalone mode stores offloaded KV cache in one node-local
`umbp_standalone_server` process. Every vLLM engine, data-parallel rank, and
worker on the host attaches to that server, so a prefix stored by one of them
can be restored by any other, and the cache survives engine restarts.

```text
engine A (TP ranks)     engine B / DP rank 1      scheduler processes
      │  HIP IPC register      │                         │ lookup
      └──────────┬─────────────┴─────────────────────────┘
                 ▼  gRPC over a Unix-domain socket
        umbp_standalone_server  ── one DRAM pool per host
```

Workers register their GPU KV allocations with the server through HIP IPC.
Transfers then run in the server between those allocations and its pool; no KV
bytes cross the socket. Scheduler lookups query the server directly.

## Running the server

Launch the server before the engines and keep it running independently of
them. Its DRAM pool is sized from its own environment:

```bash
export UMBP_STANDALONE_ADDRESS=unix:///run/umbp/node.grpc.sock
export UMBP_DRAM_CAPACITY=$((512 * 1024**3))
python -c "import mori, os; print(os.path.dirname(mori.__file__))"  # binary location
umbp_standalone_server
```

The server needs the same GPU access as the engines (`/dev/kfd`, `/dev/dri`)
and must share the host IPC namespace with them. In containers, run it in the
engine container or with `--ipc=host` and the same device mounts.

`UMBP_DRAM_*` variables control huge pages, NUMA placement, prefaulting, and
eviction watermarks, as described in the MORI UMBP documentation.

## Connector configuration

```bash
vllm serve <model> \
  --enable-prefix-caching \
  --kv-transfer-config '{
    "kv_connector": "UMBPStoreConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "mode": "standalone",
      "endpoint": "/run/umbp/node.grpc.sock",
      "load_async": true
    }
  }'
```

| Option | Default | Meaning |
| --- | --- | --- |
| `endpoint` | required | Server socket path, or `unix://` address. TCP is rejected. |
| `startup_timeout_ms` | `30000` | How long a client waits for the server at startup. |
| `auto_start` | `false` | Spawn the server from the first worker if none is running. |
| `num_workers` | `4` | Transfer threads per worker. |
| `timeout_ms` | `30000` | How long a transfer may run before the worker treats the server as stalled. |
| `layerwise_load`, `layerwise_load_stages` | `true`, `4` | With `load_async=false`, load in this many stages of consecutive layers; each attention layer waits only for its own stage. Models with Mamba-style layers always load asynchronously. |
| `load_async`, `lookup_async`, `lazy_offload`, `key_namespace` | | Shared connector options, as in embedded mode. |

`capacity_bytes` and `dram_*` options are only accepted together with
`auto_start`, where they size the spawned server. Otherwise the server's
environment owns capacity, and the connector rejects them rather than silently
ignoring them. `total_capacity_bytes` is always rejected.

Prefer an externally launched server in production. `auto_start` forks the
server from a worker process, so the server's lifetime and environment are tied
to whichever engine started first.

## Sharing scope

Objects are keyed by the model, revision, KV layout, cache groups, TP/PCP/DCP
topology, and the vLLM block hash. The data-parallel rank is not part of the
key, so DP ranks and independent engines that serve the same model with the
same layout share objects. Engines with a different layout use different keys
and cannot read each other's objects. `key_namespace` overrides the derived
namespace; use it to separate weight versions that share a model path.

## Failure behavior

| Event | Behavior |
| --- | --- |
| Server not reachable at engine startup | Engine startup fails. |
| Server dies while engines run | Lookups report misses, loads and stores fail, and requests recompute. Engines keep serving. |
| Server stalls while engines run | A store that outlives `timeout_ms` stays pending with its source blocks pinned until the server answers; new loads and stores fail immediately meanwhile. A synchronous wait on a stalled transfer raises. |
| Server restarted at the same address | Engines attached before the restart lose their GPU registrations; restart them to resume offloading. |
| Engine restarts | Objects remain in the server and are restored by the new engine. |
| `POST /reset_prefix_cache?reset_external=true` | Clears the whole server pool, for every attached engine. |

## Limitations

- Pipeline parallelism is rejected, as in embedded mode.
- Layer-wise store is disabled; layer-wise load is supported.
- Server-side evictions do not produce KV removal events.
- The server has no per-engine quota: engines share one pool and can evict
  each other's objects.
