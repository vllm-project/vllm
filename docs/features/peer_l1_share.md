# Peer L1 KV cache sharing

`PeerL1Connector` reuses prompt-prefix KV directly from another data-parallel
worker's GPU memory. It helps DP+EP MoE deployments when requests from one
session reach different ranks and would otherwise recompute the same prefix.
It is an optional cache: a miss falls back to local computation.

Workers on the same host use CUDA/HIP IPC. Independent instances can read
compatible peer caches using MoRI-IO RDMA. Local IPC sources take priority;
a remote hit must provide enough additional tokens to justify an async load.

## Requirements

- A MoE model with prefix caching enabled on every participating worker.
- TP=PP=DCP=PCP=1; scale with DP and optionally expert parallelism.
- Identical weights, tokenizer, position encoding, dtype and KV geometry.
- A C++17 compiler and pthread for the native control service. Set `CXX` if needed.
  Its build cache is under `$VLLM_CACHE_ROOT/peer_l1`.
- For RDMA: MoRI-IO, working RDMA devices/providers and network reachability
  between control endpoints and MoRI transport endpoints.

On ROCm, set these before launching either path so copies retain high priority:

```bash
export DEBUG_HIP_IGNORE_STREAM_PRIORITY=0
export ROCPROFILER_QUEUE_INTERPOSITION=1
```

Legacy profiler interception, including some profiling modes, can override
queue priority. Check the connector's startup warning if copies delay compute.

## Same-instance IPC

Enable the connector on all DP ranks of the instance:

```bash
vllm serve ibm-research/PowerMoE-3b \
  --data-parallel-size 2 --enable-expert-parallel --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"PeerL1Connector","kv_role":"kv_both"}'
```

IPC participants must share a process failure domain. A failed IPC load stops
execution rather than releasing source rows while a copy might still read them.

## Cross-instance RDMA

Give every instance a distinct ID and a reachable numeric control IP. Use the
same `kv_compatibility_id` only for instances with matching KV semantics.
For example, this instance discovers owners through `10.0.0.1:28500`:

```bash
export PEER_L1_SECRET='replace-with-a-deployment-secret'
vllm serve ibm-research/PowerMoE-3b \
  --data-parallel-size 2 --enable-expert-parallel --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"PeerL1Connector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"instance_id":"instance-b","kv_compatibility_id":"weights-and-config-v1","control_host":"10.0.0.2","control_port":28600,"peer_seeds":["tcp://10.0.0.1:28500"],"shared_secret_env":"PEER_L1_SECRET"}}'
```

The seed instance needs its own ID/address and the same compatibility ID and
secret; it may use an empty seed list. Owners use `control_port + 2 * rank` and
the next port for subscriptions. Permit the full range for the configured DP size.
Cross-instance `xxhash` also requires a common `PYTHONHASHSEED`.

RDMA loads are asynchronous. A failed receive recomputes and disables further
RDMA attempts for that request; IPC remains eligible. Owners also enter a
bounded exponential cooldown. An uncertain DMA retains target blocks until
proven drained, potentially until process exit.

## Options and limits

Options belong in `kv_connector_extra_config`:

| Option | Default | Meaning |
| --- | --- | --- |
| `pin_window` | `16384` | Upper bound on immediate pins per owner absorb interval. |
| `pin_cap` | `65536` | Limit on outstanding pin references and queued key positions. |
| `busy_timeout_ms` | `1000` | Reader's wait budget for an undecided busy/queued acquire. |
| `lease_ms` | `30000` for RDMA | Grant lifetime measured from its original acquire. |
| `read_timeout_s` | `1.0` | Transfer budget from first possible post, within half the grant lease. |
| `min_gain_pages` | `4` | Additional aligned prefix pages needed to prefer RDMA. |
| `rdma_quarantine_fraction` | `0.0625` | Pool fraction above which new RDMA plans stop. |

The immediate window adapts to free headroom. Oversized hits can wait for the
owner's next absorb, which bounds how many free rows a queued grant takes.
Committed IPC grants remain pinned until completion or cancellation.
Queueing, session setup and the step fence share the original half-lease bound;
the transfer timeout does not extend the lease.

Sleep mode and composition with other KV connectors are unsupported.
The RDMA mover has one I/O thread, so a slow source can delay other reads.
Quarantine is a safety mechanism, not an admission guarantee: already submitted
reads can exceed its threshold before feedback disables new reads.

## Metrics

Metrics use the `vllm:peer_l1_` prefix and vLLM's per-engine labels:

- `reader_lookups_total`, `reader_tokens_total{source}` and `reader_granted_rows_total`.
- `reader_busy_replies_total`, `reader_deadline_fallbacks_total`,
  `reader_truncations_total{reason}` and busy/deferral wait histograms.
- `owner_grants_total`, `owner_rows_pinned_total`, `owner_deferred_rejections_total{reason}`,
  `owner_effective_window`, `owner_reserve` and `owner_withheld_high_water`.
- `worker_copy_bytes_total`, copy GPU/main-stream wait histograms, and
  `scheduler_seconds_per_step` for connector CPU time.
- `worker_rdma_reads_total`, `worker_rdma_successes_total`,
  `worker_rdma_failures_total{reason}` and `reader_quarantine_refs`.

Failure reasons distinguish deadlines before/after post, failed status,
exceptions, full queue, changed owner and cancellation. Sum over `reason` for the
failure total. The first owner/reason failure logs at INFO, with repeats limited
to one per minute. No request ID is used as a metric label.

## Security

Deploy only on a trusted, isolated network. `shared_secret` is supplied in
`kv_connector_extra_config` and is plain text in the config, command line and
potential logs. Prefer `shared_secret_env` to name an inherited environment
variable containing the secret. Every participant must use the same secret.
Authentication does not encrypt traffic and does not protect MoRI's separate
bootstrap/RDMA ports; restrict those ports to the same trusted deployment.

See the [ownership notes](../design/peer_l1_share.md) for implementation invariants.
