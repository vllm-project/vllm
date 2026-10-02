# UMBP Embedded DRAM Offloading

`UMBPStoreConnector` offloads KV cache blocks to host DRAM managed by UMBP,
the unified memory buffer pool of [MORI](https://github.com/ROCm/mori), and
restores them when a later request shares a prefix. In embedded mode each vLLM
worker owns a private pool in its own process: a stored prefix can be restored
by the engine that stored it, after it has been evicted from GPU memory.

```text
vLLM GPU KV cache ── store ──► worker's UMBP DRAM pool
                  ◄── load ───
```

vLLM schedules requests, owns the GPU KV blocks and their hashes, and plans
every transfer. MORI allocates the pool, indexes objects, and evicts them.

## Requirements

- MORI built with UMBP (`BUILD_UMBP=ON`), importable as `mori`.
- Prefix caching enabled.

## Usage

```bash
vllm serve <model> \
  --enable-prefix-caching \
  --kv-transfer-config '{
    "kv_connector": "UMBPStoreConnector",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "recompute",
    "kv_connector_extra_config": {
      "mode": "embedded",
      "total_capacity_bytes": 68719476736
    }
  }'
```

Use `kv_load_failure_policy: "recompute"`. Each pool evicts on its own, so an
object can disappear between a lookup and the load that follows it; with the
default `fail` policy that request fails instead of recomputing.

With `kv_role: "kv_consumer"` the connector only restores from the pool and
never stores.

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `capacity_bytes` | 64 GiB | Pool size per worker. |
| `total_capacity_bytes` | | Pool size per engine, split evenly across its workers. Exclusive with `capacity_bytes`. |
| `load_async` | `true` | Load in the background while the request waits. Models with Mamba-style or sparse-attention indexer layers always load asynchronously, since those layers read restored state before a synchronous load could have landed. |
| `lookup_async` | `false` | Look up the pool off the scheduler thread; the request waits until the lookup completes. |
| `enable_lookup` | `true` | Restore from the pool. With `false` the connector only stores. |
| `save_decode_cache` | `false` | Also store blocks produced by decoding. |
| `enable_partial_hash_hits` | `false` | Allow hits that end inside a block, at core's prefix-match unit (`cache_config.prefix_match_unit`). |
| `num_workers`, `timeout_ms` | `4`, `30000` | Transfer threads per worker, and how long a worker waits for a transfer it must finish before failing the step rather than reusing its blocks. |
| `lookup_instance` | | Distinguishes independent engines serving the same model on one host. Data-parallel ranks are already distinguished. |
| `lookup_dir` | `/tmp` | Directory of the per-worker lookup sockets. |
| `key_namespace` | derived | Overrides the namespace derived from the model, revision, KV layout, cache groups, parallel topology and draft model. |
| `dram_high_watermark`, `dram_low_watermark` | MORI defaults | Pool occupancy at which eviction starts and stops. |
| `dram_use_hugepages`, `dram_hugepage_size`, `dram_numa_node`, `dram_prefault` | MORI defaults | Allocation of each worker's pool. |

## Limitations

- Pipeline parallelism is rejected.
- Pools are private to a worker; engines do not share them.
- Stores are submitted once per step after the forward pass.
