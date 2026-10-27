# HiSparse local KV offload architecture

Status: experimental

## The short version

There are three different jobs:

1. The normal KV cache system manages GPU block pools and tables.
2. `HiSparseCoordinator` manages logical host blocks, host-prefix identity, and
   host/GPU residency transitions.
3. `HiSparseConnector` carries residency work between scheduler and worker;
   `HiSparseWorker` coordinates transfers, while per-cache
   `HiSparseRuntime` objects own host/hot views and GPU replacement state.

Neither worker-side object allocates or frees logical blocks. HMA provides the
GPU allocation shared by the device and hot groups; it does not manage CPU
memory or KV identity.

```text
request ────► HiSparseCoordinator ── host blocks + residency policy
                    │
                    ├──► KV cache manager ── device/hot GPU leases (HMA)
                    │
                    └──► HiSparseConnector ── host bytes + copies + GPU LRU
```

This is a local KV connector. When P/D or another offload connector is also
configured, `MultiConnector` composes it with `HiSparseConnector`.

`host_pool_gib` is configured on `HiSparseConnector` and is the usable
host-cache capacity per data-parallel replica, not a node-wide memory budget.
Tensor-parallel ranks hold replicated views of that logical cache. Those views
may use private per-rank backing or one shared
physical allocation without changing the configured capacity. Physical host
memory consumption is therefore topology- and implementation-dependent. The
realized capacity may be slightly smaller because the budget is rounded down
to complete host blocks.

Startup logs report two concurrency bounds at `max_model_len`. The generic
`Maximum concurrency` line charges each request its full admission footprint,
including the in-flight window of every device group. The `HiSparse
steady-state maximum concurrency` line charges running requests that read from
host only their active tail pages, plus one request being admitted at its full
footprint. Host-pool metrics are listed in
[Metrics](../usage/metrics.md#hisparse-kv-connector-metrics).

## Nomenclature

| Term | Meaning |
| --- | --- |
| Host block / host page | A block of the separate CPU pool and the KV page it holds. The host pages are the authoritative copy of a request's sparse-MLA KV. |
| Device page | A request's KV page held in a GPU block of its device group. Hot rows are also in GPU memory but are not device pages. |
| Hot cache | The GPU cache group, and its tensor, that holds every request's hot buffer. |
| Hot buffer / hot row | A request's fixed set of GPU blocks that hold host pages fetched for attention, and one row within it. |
| Write-back | Copying a device page to its host page. Happens eagerly, not only under memory pressure. |
| Page transfer | One tracked write-back or restore of a page, identified by its transfer ID. |
| Mirror | Copying each step's newly written KV rows to their host pages, rather than waiting to write back whole pages. |
| Durable | A page whose write-back (or import) has completed, so its device block can be released. |
| Publish | Registering durable host pages in the prefix cache so later requests can hit them. |
| Resolve residency | Mapping attention's top-k positions to physical rows: device pages directly, hot rows on a hit, and host rows copied into the hot buffer on a miss (`resolve_residency`). |
| Restore | Copying a host page back into a device page. |
| Import | KV arriving from a P/D prefill node. |
| Pin / release | Holding or dropping a device page's allocation reference. Releasing (`BlockPool.unpin_blocks` in core) keeps the block readable until the pool evicts it. |
| Lease / free | An extra block reference a page transfer or pending publication holds (`BlockPool.touch`) so the block outlives the copy, and dropping it (`BlockPool.free_blocks`). |
| GPU copy | A durable device page whose reference was released but whose contents stay readable until the pool evicts its block. |
| Revive | A host-prefix hit taking back a released GPU copy the pool has not yet evicted, re-pinning it instead of reading the page from host. |
| Evict | The pool handing a released block to another allocation, which destroys any GPU copy it held. |
| Residency update | The scheduler telling the worker which GPU block, if any, each of a request's pages is read from (`take_residency_updates`). The block table stays append-only. |

## Ownership

| Thing | Owner | What “owner” means |
| --- | --- | --- |
| HiSparse host-prefix identity | `HiSparseCoordinator` | maps tokens to logical host blocks |
| Device block leases | normal KV cache manager | allocates and frees HMA blocks |
| Device block tables | normal KV cache manager | append-only record of a request's device allocations |
| Per-page residency | `HiSparseCoordinator` | decides which pages attention reads from GPU and sends residency updates |
| Residency transitions | `HiSparseCoordinator` | plans write-back-before-release transactions |
| Logical host block allocation | `HiSparseCoordinator` | owns the separate CPU block pool and its lifecycle |
| Pinned host-pool lifecycle | `HiSparseWorker` | worker-wide backing and teardown |
| Per-cache host view and hot contents | `HiSparseRuntime` | binds host/hot storage and fills cache-manager-provided hot leases |
| Hot row map and LRU | `HiSparseRuntime` | resolves hits and chooses victims on GPU |
| Device-cache route | `HiSparseCacheHandle` | exposes device or host/hot resolution to attention |
| Sparse attention | attention backend | consumes a device cache and physical row IDs |
| HMA | allocator | provides GPU capacity; owns no KV meaning |

The key distinction is logical allocation versus contents. `HiSparseCoordinator`
owns host block IDs and their request/prefix associations. `HiSparseWorker` and
its per-cache runtimes own the corresponding bytes. The normal cache manager
sees only device pools.
The host group has `block_pool_id=None`; device-pool consumers must narrow it
before indexing, so host ownership cannot masquerade as a numeric GPU pool.

For single-node MP tensor parallelism, every TP worker maps the same pinned host
pool and uses the same block and layer offsets. MLA KV is replicated
across TP ranks, so this stores one physical copy instead of one copy per rank.
TP rank 0 writes the shared host pool; peers wait on its IPC events before
reading it. Other executor and parallel layouts retain private per-rank pools.

The shared layout backs the per-replica logical capacity with one physical pool;
the private layout allocates one physical pool per rank. Physical pool size
includes block-stride alignment.

## Code boundary

```text
scheduler process                           worker process

HiSparseConnector                          HiSparseConnector
  └─ HiSparseCoordinator                         └─ HiSparseWorker
       │                                          │
       │ connector metadata                       ├─ host bytes
       │ - page transfers                         ├─ copy scheduling
       │ - residency updates                      └─ per-layer hot state
       └───────────────────────────────────────────────►│
       ◄──────── connector worker metadata ─────────────┘
                    enqueued and completed transfer IDs
```

The command travels in `kv_connector_metadata`; transfer updates return in
`KVConnectorOutput.kv_connector_worker_meta`. The model runner does not
interpret page transfers. Enqueue acknowledgements let the scheduler free
device leases in stream order; completion acknowledgements make the copied
pages durable and publish them.

## Device pages

Device pages are intentionally outside `HiSparseRuntime`.

KV-cache initialization binds cache-manager allocations to the attention-facing
`HiSparseCacheHandle` before constructing `HiSparseWorker`. That same
handle's runtime retains the device group index needed by a transfer plan.
There is no second device object or registration wrapper.

```text
KV cache setup
   │
   ├─ bind device allocation ──► HiSparseCacheHandle
   │                               cache + block table + slot mapping
   │
   ├─ bind host/hot allocation ──► HiSparseRuntime
   │                               host + hot + GPU LRU
   │
   └─ register cache handles ────► HiSparseWorker
                                   step-level transfers

HiSparseWorker registers the same HiSparseCacheHandle objects directly
```

Attention construction links each layer to the most recent layer that actually
owns an indexer. This releases a follower's duplicate LRU tensors before GPU
memory profiling. Cache binding only attaches storage; it does not infer
semantic groups from the physical packed-tensor order. The construction cursor
is discarded with the worker's pinned state.

Every HiSparse decode batch uses the same fused resolver. It checks device
pages first, then hot rows, then pinned host memory. A device hit exits inside
the kernel before hot-LRU lookup or host copying; there is no framework-level
residency route or separate CUDA graph. No CPU decision is added to the decode
path. The resolver consumes the existing graph-stable request mapping from
attention metadata; neither the worker nor individual cache handles keep a
duplicate mapping.

Speculative decoding resolves all verification rows of a request in one pass:
one block resolves the union of the rows' top-k against the request's hot-buffer
state, so rows that select the same host row share its hot row and no row
evicts a hot row another row of the step still reads. Draft layers write their
rows after the target forward, so their host mirror runs at the start of the
next step, after the drafter, and the step's page transfers are submitted
behind it.

## P/D import target

The decoder chooses the landing target once per request from the normal cache
admission calculation. If the complete imported prefix fits the device pools,
NIXL transfers it directly into device pages. Otherwise, if the fixed
host-backed GPU footprint and host blocks fit, the request imports into
the host tier. There is no context-length threshold or other heuristic, and a
request waiting for capacity retains its choice across admission retries.

A host import reads through a bounded decoder-GPU staging pool before copying
into registered host memory. Pages needed immediately are mirrored into their
device destinations during that copy. Both landing targets then use the same
fused decode resolver described above.

## Indexer KV offloading

HiSparse does not keep a private CPU copy of indexer KV. The indexer remains a
normal prefix-cacheable GPU cache group. If `OffloadingConnector` is configured
with HiSparse, it stores and restores that group through the generic KV
offloading path; HiSparse continues to own only the sparse MLA host tier.

The two prefix sources can have different hit lengths. When the HiSparse host
prefix extends beyond the device indexer prefix, the scheduler asks
`OffloadingConnector` to restore only the missing indexer suffix, capped at the
host prefix boundary. If that suffix is unavailable, all groups fall back to
the shorter prefix they share. NIXL P/D transfers continue to place indexer KV
directly in its GPU group.

## Write-back transaction

A device page cannot be released until its write-back has completed.

```text
HiSparseCoordinator                            HiSparseWorker
          │                                     │
          │ lease device and host blocks      │
          │── HiSparsePageTransfer ─────────────►│
          │                                     │ enqueue GPU-to-host copy
          │◄── enqueued transfer ID ────────────│
          │ free device leases                │
          │                                     │ copy reaches its event
          │◄── completed transfer ID ───────────│
          │ mark page durable                   │
          │ release it if the request can       │
          │   read from host                    │
          │ free host lease; publish            │
```

“Enqueued” means the copy has entered the worker stream. Stream ordering makes
it safe to free the device leases, since later GPU work cannot overtake the
copy, but the page is not yet durable. “Completed” means the worker has
observed the copy's event; only then does the coordinator mark the page
durable, release its device block once the request can read from host,
publish the host page for prefix reuse and free its host lease. A host-write event separately protects direct
CPU readers from writes already queued on the accelerator.

The worker transfer contains only its transfer ID and physical copy
coordinates. Request identity and logical page state remain in the scheduler.

## Hot lookup and LRU

The NVIDIA CUDA path keeps replacement entirely on the accelerator:

```text
top-K logical positions
        │
        ▼
device page? ── yes ──► device physical row
        │ no
        ▼
hot row? ──────── yes ──► existing hot physical row + update GPU LRU
        │ no
        ▼
choose GPU LRU victim ──► copy pinned host row ──► hot physical row
```

ROCm is not currently supported because the fused HiSparse cache operations are
implemented only by CUDA kernels. A future platform-specific worker may provide
the same command, output, and cache-resolution boundaries.

## Main classes

| Class | Inherits / implements | Responsibility |
| --- | --- | --- |
| `HiSparseCoordinator` | plain scheduler component | host allocation, host prefixes, device leases, and page-transfer state machine |
| `HiSparseConnector` | `KVConnectorBase_V1`, `SupportsHMA` | scheduler/worker metadata and lifecycle boundary |
| `HiSparseDeviceManager` | `SingleTypeKVCacheManager` | normal block-pool bookkeeping with host-backed holes |
| `PagedCacheView` | immutable data object | shared device/hot HMA tensor binding |
| `HiSparseWorker` | connector-owned worker component | worker-wide transfer scheduling and host-pool lifecycle |
| `HiSparseRuntime` | plain worker-owned component | per-cache host/hot tensors, GPU LRU, and fused resolution |
| `HiSparseCacheHandle` | plain attention component | device view and fused cache resolution |
| `HiSparseTransferCommand` | dataclass | opaque scheduler-to-worker work |

## Performance invariants

- Device hits bypass hot-LRU lookup and host copies inside the fused resolver.
- Hot lookup, victim selection, and LRU updates stay on the GPU.
- A hot miss still copies directly from registered pinned host memory.
- Top-K resolution stays inside the attention invocation and remains graph
  capturable.
- Compatible layers still share one miss plan.
- Index-sharing followers release their private LRU state before memory sizing.
- Indexer KV is untouched by HiSparse unless a generic KV offloader is configured.
- Device and hot leases can still share one packed HMA allocation.
- No device scalar readback or CPU/device metadata round trip is added.
- The abstraction wraps the fused kernel; it does not add another kernel
  launch.
- When HiSparse is disabled, the scheduler does not construct an offload
  command or empty update table.

## What remains platform-specific

The command/result and attention-layer boundaries can be shared. The host
allocator, copy implementation, hot layout, and replacement policy should stay
platform-specific. NVIDIA uses the current accelerator LRU and fused host/hot
kernel. ROCm is not currently supported; AMD or other accelerator backends can
implement their own worker without forcing NVIDIA's policy into the shared
boundary.
