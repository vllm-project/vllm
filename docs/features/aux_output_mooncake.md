# Sizing Mooncake storage for routed experts

With the `mooncake` auxiliary-output backend, configure storage through
`MOONCAKE_CONFIG_PATH`. `aux_output_config.max_bytes` applies only to SHM;
Mooncake capacity is not automatically derived from the GPU KV cache.

- `global_segment_size` is the memory each embedded Store client contributes
  to the shared pool. Count every client, including Worker clients and the
  EngineCore publisher; this is not a single limit for the whole deployment.
  In `standalone-store` mode, set it to zero and provision the external pool.
- `local_buffer_size` is each client's transfer buffer, not retained-output
  capacity. It must accommodate at least one R3 block. This backend caps its
  transfer batches at the smaller of this value and 64 MiB.

Estimate the payload before choosing a pool size:

```text
bytes per token = captured layers × experts per token × dtype itemsize
retained payload ≈ retained token rows × bytes per token
```

For example, 43 layers, 6 experts and `uint8` require 258 bytes per token.
One 256K-token output occupies 64.5 MiB; 32 such outputs occupy about
2.02 GiB before storage overhead and additional retained outputs.

Choose capacity for peak concurrency **and** the delay before consumers fetch
their results. Allow headroom for retained cache blocks, non-reusable tail
objects, storage overhead, and other users of the same pool. Deduplication
does not eliminate independent instances' or DP ranks' output copies.

Returned keys are references, not copies or permanent retention guarantees.
Finishing a request does not delete its stored output. The deployment must
retain objects until consumers have fetched them; an evicted or missing key
causes a read error. Validate capacity and retention under the intended load
rather than treating the default size as sufficient.
