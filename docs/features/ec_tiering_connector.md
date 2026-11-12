# Encoder Cache Filesystem Tiering

`ECTieringConnector` extends CPU encoder-cache offloading with asynchronous
filesystem storage. Enable Model Runner V2 and use `ec_role="ec_both"`:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve <model> --ec-transfer-config '{
  "ec_connector": "ECTieringConnector",
  "ec_role": "ec_both",
  "ec_connector_extra_config": {
    "ec_cpu_bytes": 8589934592,
    "secondary_tiers": [{
      "type": "fs",
      "root_dir": "/data/ec-cache",
      "capacity_bytes": 137438953472
    }]
  }
}'
```

The connector currently requires `ec_both`, including when `secondary_tiers` is
empty. Separate `ec_producer` and `ec_consumer` roles are rejected during
initialization. CPU-only and NIXL disaggregated deployments can use
[`ECCPUConnector`](ec_cpu_connector.md).

Encoder outputs pass through CPU memory when stored or loaded. Compatible
filesystem entries can be reused after a process restart. The filesystem tier
uses LRU eviction within the current namespace. Before accepting a write, it
checks available capacity, including reservations for accepted writes and space
still occupied by pending deletions. If space is insufficient, it submits
asynchronous eviction deletes and skips the new store. A later store attempt can
proceed after deletion has succeeded and its completion has been collected.
Active reads protect their files from eviction.

A failed deletion restores the old entry and its capacity accounting; a
subsequent store attempt can retry deletion asynchronously. A failed write
releases its capacity reservation. The scheduler never waits for disk space.

`capacity_bytes` budgets the current instance's managed entry payloads, including
in-flight write reservations. It is not an operating-system filesystem quota.
Other namespaces and files owned by other processes are outside this accounting.
If startup recovery finds more data than the configured capacity, existing data
must be evicted before new writes can be accepted.
