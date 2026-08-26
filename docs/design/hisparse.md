# HiSparse local KV offload architecture

Status: experimental

## The short version

There are three different jobs:

1. The normal KV cache system manages GPU block pools and tables.
2. `HiSparseCoordinator` manages logical host blocks, source-prefix identity, and
   host/GPU residency transitions.
3. `HiSparseConnector` carries residency work between scheduler and worker;
   `HiSparseWorker` coordinates transfers, while per-cache
   `HiSparseRuntime` objects own host/hot views and GPU replacement state.

Neither worker-side object allocates or frees logical blocks. HMA provides the
GPU allocation shared by the resident and hot groups; it does not manage CPU
memory or KV identity.

```text
request ────► HiSparseCoordinator ── source blocks + residency policy
                    │
                    ├──► KV cache manager ── resident/hot GPU leases (HMA)
                    │
                    └──► HiSparseConnector ── host bytes + copies + GPU LRU
```

This is a local KV connector. When P/D or another offload connector is also
configured, `MultiConnector` composes it with `HiSparseConnector`.

`host_pool_gib` is the usable host-cache capacity per data-parallel replica,
not a node-wide memory budget. Tensor-parallel ranks hold replicated views of
that logical cache. Those views may use private per-rank backing or one shared
physical allocation without changing the configured capacity. Physical host
memory consumption is therefore topology- and implementation-dependent. The
realized capacity may be slightly smaller because the budget is rounded down
to complete host blocks.

## Ownership

| Thing | Owner | What “owner” means |
| --- | --- | --- |
| HiSparse source and prefix identity | `HiSparseCoordinator` | maps tokens to logical host blocks |
| Resident GPU block leases | normal KV cache manager | allocates and frees HMA blocks |
| Resident block tables | normal KV cache manager | tells attention where resident pages are |
| Residency transitions | `HiSparseCoordinator` | plans spill-before-free transactions |
| Logical host block allocation | `HiSparseCoordinator` | owns the separate CPU block pool and its lifecycle |
| Pinned host-pool lifecycle | `HiSparseWorker` | worker-wide backing and teardown |
| Per-cache host view and hot contents | `HiSparseRuntime` | binds host/hot storage and fills cache-manager-provided hot leases |
| Hot row map and LRU | `HiSparseRuntime` | resolves hits and chooses victims on GPU |
| Resident-cache route | `HiSparseCacheHandle` | exposes resident or host/hot resolution to attention |
| Sparse attention | attention backend | consumes a device cache and physical row IDs |
| HMA | allocator | provides GPU capacity; owns no KV meaning |

The key distinction is logical allocation versus contents. `HiSparseCoordinator`
owns host block IDs and their request/prefix associations. `HiSparseWorker` and
its per-cache runtimes own the corresponding bytes. The normal cache manager
sees only device pools.
The source group has `block_pool_id=None`; device-pool consumers must narrow it
before indexing, so host ownership cannot masquerade as a numeric GPU pool.

## Code boundary

```text
scheduler process                           worker process

HiSparseConnector                          HiSparseConnector
  └─ HiSparseCoordinator                         └─ HiSparseWorker
       │                                          │
       │ connector metadata                       ├─ host bytes
       │ - page transfers                         ├─ copy scheduling
       │ - block-table replacements               └─ per-layer hot state
       └───────────────────────────────────────────────►│
       ◄──────── connector worker metadata ─────────────┘
                    enqueued and completed transfer IDs
```

The command travels in `kv_connector_metadata`; transfer updates return in
`KVConnectorOutput.kv_connector_worker_meta`. The model runner does not
interpret page transfers. Enqueue acknowledgements let the scheduler release
source leases in stream order; completion acknowledgements publish the copied
host pages.

## Resident device pages

Resident pages are intentionally outside `HiSparseRuntime`.

KV-cache initialization binds cache-manager allocations to the attention-facing
`HiSparseCacheHandle` before constructing `HiSparseWorker`. That same
handle's runtime retains the resident source index needed by a transfer plan.
There is no second resident object or registration wrapper.

```text
KV cache setup
   │
   ├─ bind resident allocation ──► HiSparseCacheHandle
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

Every HiSparse decode batch uses the same fused resolver. It checks resident
pages first, then hot rows, then pinned host memory. A resident hit exits inside
the kernel before hot-LRU lookup or host copying; there is no framework-level
residency route or separate CUDA graph. No CPU decision is added to the decode
path. The resolver consumes the existing graph-stable request mapping from
attention metadata; neither the worker nor individual cache handles keep a
duplicate mapping.

Speculative decoding resolves and consumes each verification step in order.
Each step receives distinct replayable plan rows while sharing the request's
hot-cache state, so a later step cannot reuse a hot row before an earlier step
has consumed it.

## P/D import target

The decoder chooses the landing target once per request from the normal cache
admission calculation. If the complete imported prefix fits the device pools,
NIXL transfers it directly into resident GPU pages. Otherwise, if the fixed
host-backed GPU footprint and host source blocks fit, the request imports into
the host tier. There is no context-length threshold or other heuristic, and a
request waiting for capacity retains its choice across admission retries.

A host import reads through a bounded decoder-GPU staging pool before copying
into registered host memory. Pages needed immediately are mirrored into their
resident destinations during that copy. Both landing targets then use the same
fused decode resolver described above.

## Indexer KV offloading

HiSparse does not keep a private CPU copy of indexer KV. The indexer remains a
normal prefix-cacheable GPU cache group. If `OffloadingConnector` is configured
with HiSparse, it stores and restores that group through the generic KV
offloading path; HiSparse continues to own only the sparse MLA host tier.

The two prefix sources can have different hit lengths. When the HiSparse host
prefix extends beyond the GPU-resident indexer prefix, the scheduler asks
`OffloadingConnector` to restore only the missing indexer suffix, capped at the
host prefix boundary. If that suffix is unavailable, all groups fall back to
the shorter prefix they share. NIXL P/D transfers continue to place indexer KV
directly in its GPU group.

## Spill transaction

A resident block cannot be reused until its contents have been handed to the
worker.

```text
HiSparseCoordinator                            HiSparseWorker
          │                                     │
          │ pin source and destination leases   │
          │── SparseKVPageTransfer ─────────────►│
          │                                     │ enqueue GPU-to-host copy
          │◄── enqueued transfer ID ────────────│
          │ replace resident table entry        │
          │ release resident lease to HMA       │
          │                                     │ copy reaches its event
          │◄── completed transfer ID ───────────│
          │ mark host page valid                │
          │ release destination host lease      │
```

“Enqueued” means the copy has entered the worker stream. Stream ordering makes
it safe to reuse the resident GPU block for later work, but the host page is
not yet published. “Completed” means the worker has observed the copy's event;
only then does the coordinator publish the host page for prefix reuse and
release its destination lease. A host-write event separately protects direct
CPU readers from writes already queued on the accelerator.

The worker transfer contains only its transfer ID and physical copy
coordinates. Request identity and logical page state remain in the scheduler.

## Hot lookup and LRU

The NVIDIA CUDA path keeps replacement entirely on the accelerator:

```text
top-K logical positions
        │
        ▼
resident page? ── yes ──► resident physical row
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
| `HiSparseCoordinator` | plain scheduler component | host allocation, source prefixes, resident leases, and spill state machine |
| `HiSparseConnector` | `KVConnectorBase_V1`, `SupportsHMA` | scheduler/worker metadata and lifecycle boundary |
| `HiSparseResidentManager` | `SingleTypeKVCacheManager` | normal block-pool bookkeeping with host-backed holes |
| `PagedCacheView` | immutable data object | shared resident/hot HMA tensor binding |
| `HiSparseWorker` | connector-owned worker component | worker-wide transfer scheduling and host-pool lifecycle |
| `HiSparseRuntime` | plain worker-owned component | per-cache host/hot tensors, GPU LRU, and fused resolution |
| `HiSparseCacheHandle` | plain attention component | resident view and fused cache resolution |
| `SparseKVOffloadCommand` | dataclass | opaque scheduler-to-worker work |

## Performance invariants

- Resident hits bypass hot-LRU lookup and host copies inside the fused resolver.
- Hot lookup, victim selection, and LRU updates stay on the GPU.
- A hot miss still copies directly from registered pinned host memory.
- Top-K resolution stays inside the attention invocation and remains graph
  capturable.
- Compatible layers still share one miss plan.
- Index-sharing followers release their private LRU state before memory sizing.
- Indexer KV is untouched by HiSparse unless a generic KV offloader is configured.
- Resident and hot leases can still share one packed HMA allocation.
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
