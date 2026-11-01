# UMBPStoreConnector Usage Guide

`UMBPStoreConnector` offloads KV cache blocks to host DRAM managed by UMBP,
the unified memory buffer pool of [MORI](https://github.com/ROCm/mori). It
restores them when a later request shares a prefix that has left GPU memory.

## Requirements

- MORI built with UMBP (`BUILD_UMBP=ON`), importable as `mori`.
- Prefix caching enabled.

## Embedded mode

Embedded mode is the default. Each vLLM worker owns a private DRAM pool in its
own process. The engine that stored a prefix can restore it; independent
engines do not share these pools.

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

### Embedded options

Set these options in `kv_connector_extra_config`:

| Option | Default | Meaning |
| --- | --- | --- |
| `capacity_bytes` | 64 GiB | Pool size per worker. |
| `total_capacity_bytes` | | Pool size per engine, split evenly across its workers. Exclusive with `capacity_bytes`. |
| `lookup_instance` | | Distinguishes independent engines serving the same model on one host. Data-parallel ranks are already distinguished. |
| `lookup_dir` | `/tmp` | Directory of the per-worker lookup sockets. |
| `dram_high_watermark`, `dram_low_watermark` | MORI defaults | Pool occupancy at which eviction starts and stops. |
| `dram_use_hugepages`, `dram_hugepage_size`, `dram_numa_node`, `dram_prefault` | MORI defaults | Allocation of each worker's pool. |

## Shared options

Set these options in `kv_connector_extra_config`:

| Option | Default | Meaning |
| --- | --- | --- |
| `load_async` | `true` | Load in the background while the request waits. Models with Mamba-style or sparse-attention indexer layers always load asynchronously, since those layers read restored state before a synchronous load could have landed. |
| `lookup_async` | `false` | Look up the pool off the scheduler thread; the request waits until the lookup completes. |
| `kv_event_max_objects` | `65536` | Maximum objects tracked for KV events. When full, new stored-object notifications are suppressed until tracked objects leave the pool. Storage and lookup continue normally. |
| `enable_lookup` | `true` | Restore from the pool. With `false` the connector only stores. |
| `save_decode_cache` | `false` | Also store blocks produced by decoding. |
| `enable_partial_hash_hits` | `false` | Allow hits that end inside a block, at core's prefix-match unit (`cache_config.prefix_match_unit`). |
| `num_workers`, `timeout_ms` | `4`, `30000` | Transfer threads per worker, and the transfer timeout. A synchronous wait that times out fails the step rather than reusing its blocks. |
| `key_namespace` | derived | Overrides the namespace derived from the model, revision, KV layout, cache groups, parallel topology and draft model. |

Use `kv_load_failure_policy: "recompute"` in `kv_transfer_config`, as in the
example above. An object can be evicted between lookup and load; the default
`fail` policy fails the request instead of recomputing it.

With `kv_role: "kv_consumer"` the connector only restores and never stores.

## Limitations

- Pipeline parallelism is rejected.
- Stores are submitted once per step after the forward pass.

With KV events enabled, the scheduler verifies residency on every rank before
publishing a stored-object event. It periodically rechecks tracked objects
during engine steps to report removals. These checks do not delay the release
of GPU source blocks after a completed transfer. With events disabled, no
historical object identities are retained for event reporting.
