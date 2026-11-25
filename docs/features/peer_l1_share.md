# Peer L1 KV cache sharing

`PeerL1Connector` reuses prompt-prefix KV directly from another data-parallel
worker's GPU memory. It helps DP+EP MoE deployments when requests from one
session reach different ranks and would otherwise recompute the same prefix.
It is an optional cache: a miss falls back to local computation.

Workers on the same host use CUDA/HIP IPC. Independent instances can read
compatible peer caches through a pluggable RDMA transport; MoRI-IO is provided.
Local IPC sources take priority;
a remote hit must provide enough additional tokens to justify an async load.

## Requirements

- Linux for the native service's eventfd and abstract Unix sockets.
- A MoE model with prefix caching enabled on every participating worker.
- TP=PP=DCP=PCP=1; scale with DP and optionally expert parallelism.
- Identical weights, tokenizer, position encoding, dtype and KV geometry.
- A C++17 compiler, pthread, and ninja for the native control service. Set `CXX`
  if needed. PyTorch builds the library at startup under
  `VLLM_CACHE_ROOT/peer_l1/vllm_peer_l1_<source hash>/`, outside the installed
  package. The hash covers all native headers and C++ sources. After an
  interrupted build, confirm no builder is still running and remove any stale
  `lock` file in that directory by hand; PyTorch does not time out waiting for it.
- For RDMA: an available transport, working RDMA devices/providers and reachable
  control and transport endpoints.

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

### MoRI-IO

Install MoRI-IO to use the provided `mori` transport. Peer L1 sets
`MORI_DISABLE_AUTO_XGMI=1`; this process-wide setting also affects a MoRI-IO PD
engine in the same process. An explicit conflicting setting is rejected.
Restrict MoRI's separate bootstrap and RDMA ports to the trusted deployment;
the control service's authentication does not protect those ports.

## Options and limits

Options belong in `kv_connector_extra_config`:

| Option | Default | Meaning |
| --- | --- | --- |
| `role` | `full` | `decode` only publishes and orders cache rows. |
| `instance_id` | Empty | Explicit cross-instance identity; omit for same-instance IPC. |
| `kv_compatibility_id` | Empty | Shared identifier for compatible model/KV semantics. |
| `control_host` | Empty | Advertised reachable numeric IP for cross-instance control. |
| `control_port` | `0` | Cross-instance base control port; must be in `1..64999`. |
| `peer_seeds` | `[]` | At most 128 `tcp://` discovery endpoints. |
| `shared_secret_env` | Empty | Inherited environment variable containing the shared secret; required with `control_host`. A named variable must be nonempty. |
| `rdma_transport` | Automatic | Registered name, currently `mori`; exactly one available implementation is required when omitted. |
| `protected_cap_blocks` | Derived for decode | Limit on unique externally protected decode rows plus admission reserve. Defaults to twice the longest model-context prefix row bound, capped at 65536. Zero disables new decode grants; ignored in `full` role. |
| `read_timeout_s` | `1.0` | Finite positive transfer budget from first possible post, within half the grant lease. |

Unknown keys are rejected, including plain-text `shared_secret`. Supply secrets
through `shared_secret_env`. Transport selection
accepts registered names only, not `module:Class` paths.

The owner window is capped at 16384 rows and pin references at 65536. RPC timeout
is 50 ms, the reader busy budget is 1000 ms, and leases are 30000 ms for RDMA and
1000 ms for IPC (committed IPC grants still require explicit completion).
RDMA must add at least four aligned prefix pages to displace an IPC source.
Quarantine disables new RDMA plans above 1/16 of the usable pool; failed owners
cool down exponentially from 1 to 30 seconds.

The immediate window adapts to free headroom. Oversized hits can wait for the
owner's next absorb, which bounds how many free rows a queued grant takes.
Committed IPC grants remain pinned until completion or cancellation.
Queueing, session setup and the step fence share the original half-lease bound;
the transfer timeout does not extend the lease.

Sleep mode is unsupported. For PD deployments, the default `full` role belongs
on the prefill instance using `MultiConnector [PD connector, PeerL1Connector]`.
MultiConnector selects the first positive hit; any child that requests waiting
makes the whole lookup wait. An unselected peer grant is cancelled.
The [decode owner role](peer_l1_decode_lending.md) also lends cached KV from a
decode instance without competing with PD for request loads. Other prefix
sources, such as offloading or LMCache, remain unsupported: each request loads
from only one child.
The RDMA mover has one I/O thread, so a slow source can delay other reads.
Quarantine is a safety mechanism, not an admission guarantee: already submitted
reads can exceed its threshold before feedback disables new reads.

## Metrics

Metrics use the `vllm:peer_l1_` prefix and vLLM's per-engine labels:

| Family | Type | Additional labels |
| --- | --- | --- |
| `reader_lookups_total` | Counter | None |
| `reader_tokens_total` | Counter | `transport`, `owner_role` |
| `reader_granted_rows_total` | Counter | None |
| `reader_truncations_total` | Counter | `reason`: `window`, `pin_cap`, `protected_cap` |
| `reader_deadline_fallbacks_total` | Counter | None |
| `worker_copy_bytes_total` | Counter | `transport`, `owner_role` |
| `worker_rdma_reads_total` | Counter | None |
| `worker_rdma_failures_total` | Counter | `reason`, `owner_role` |
| `reader_quarantine_refs` | Gauge | None |
| `reader_rdma_quarantine_disabled` | Gauge | None |
| `owner_protected_blocks` | Gauge | None |
| `owner_protected_high_water` | Gauge | None |
| `decode_gate_open` | Gauge | None |
| `owner_rejections_total` | Counter | `reason`, `stage` |
| `decode_order_rows_total` | Counter | `action`: `front`, `retired`, `scanned` |

`transport` is `ipc` or `rdma`; `owner_role` is `full` or `decode`.
Token counts describe selected peer prefixes; copy bytes describe submitted work,
including failed RDMA receives. Neither is a successful-restore count.
RDMA failure reasons are `deadline_before_post`, `deadline_after_post`,
`status_failed`, `exception`, `queue_full`, `owner_changed` and `cancelled`.
Each failed receive is counted once. The first owner/reason failure logs at
INFO, with repeats limited to one per minute.
Owner rejection stages are `immediate` and `deferred`. Immediate reasons are
`window`, `cap`, `protected_cap` and `gate`; deferred reasons are `frozen`,
`lease_expiry`, `missing_key`, `cap`, `budget` and `gate`.
No request ID or peer identity is used as a metric label.

## Security

Same-instance owner RPC listens on Linux abstract Unix sockets named
`vllm-peer-l1/<engine hash>/<rank>`; subscriptions use the same name plus `.sub`.
These sockets are reachable by other local processes in the network namespace.
By default, rank 0 generates a random 32-byte instance credential and distributes
it through the trusted DP memory-handle bootstrap, then to the schedulers.
It stays in process memory, is excluded from object representations, and is
never included in HELLO/MEMBERS or logs. Unauthenticated connections are closed
without a reply. All ranks must run the same build; mixed handshake versions
are unsupported.

Cross-instance owners listen at `control_host:control_port + 2 * rank`, with
subscriptions on the next port. Setting `control_host` requires
`shared_secret_env` naming a nonempty inherited environment variable. All peers
must use the same secret; its SHA-256 digest replaces the random credential.
Plain-text `shared_secret` configuration is rejected.

This is a static shared-secret check, sent in clear text and replayable. It
provides neither encryption nor server authentication or isolation between
authenticated peers. Isolate the DP bootstrap, control sockets and RDMA
endpoints on a trusted network. It cannot protect against processes that can
read a participant's memory or control its bootstrap.

Native libraries build under
`VLLM_CACHE_ROOT/peer_l1/vllm_peer_l1_<source hash>/`. Keep that cache writable
only by the serving account.

See the [ownership notes](../design/peer_l1_share.md) for implementation invariants.
