# HiSparse local KV offload architecture

Status: worktree design draft

## The short version

There are three different jobs:

1. The normal KV cache system manages GPU block pools and tables.
2. `HiSparseManager` manages logical host blocks, source-prefix identity, and
   host/GPU residency transitions.
3. The sparse offload store owns the worker-side host bytes, the small GPU hot
   buffer, and movement between them.

The offload store does **not** allocate or free logical blocks. HMA provides the
GPU allocation shared by the resident and hot groups; it does not manage CPU
memory or KV identity.

```text
request ────► HiSparseManager ── source blocks + residency policy
                    │
                    ├──► KV cache manager ── resident/hot GPU leases (HMA)
                    │
                    └──► sparse offload store ── host bytes + copies + GPU LRU
```

This is local storage tiering. A P/D connector is a separate feature and is
not part of this diagram.

## Ownership

| Thing | Owner | What “owner” means |
| --- | --- | --- |
| HiSparse source and prefix identity | `HiSparseManager` | maps tokens to logical host blocks |
| Resident GPU block leases | normal KV cache manager | allocates and frees HMA blocks |
| Resident block tables | normal KV cache manager | tells attention where resident pages are |
| Residency transitions | `HiSparseManager` | plans spill-before-free transactions |
| Logical host block allocation | `HiSparseManager` | owns the separate CPU block pool and its lifecycle |
| Pinned host tensor and bytes | `HiSparseOffloadStore` | worker-side backing for spilled pages |
| Hot GPU block contents | `HiSparseOffloadStore` | fills cache-manager-provided hot leases |
| Hot row map and LRU | `HiSparseOffloadLayer` | resolves hits and chooses victims on GPU |
| Sparse attention | attention backend | consumes a device cache and physical row IDs |
| HMA | allocator | provides GPU capacity; owns no KV meaning |

The key distinction is logical allocation versus contents. `HiSparseManager`
owns host block IDs and their request/prefix associations. The offload store
owns the corresponding bytes. The normal cache manager sees only device pools.
The source group has `block_pool_id=None`; device-pool consumers must narrow it
before indexing, so host ownership cannot masquerade as a numeric GPU pool.

## Code boundary

```text
scheduler process                         worker process

HiSparseManager
  │
  │ SparseKVOffloadCommand
  │   - page transfers
  │   - block-table replacements
  │   - fully-resident route bit
  ▼
GPUModelRunner ───────────────────────► HiSparseOffloadStore
  ▲                                        │
  │ completed transfer IDs                 ├─ host allocations
  │                                        ├─ copy scheduling
  └────────────────────────────────────────└─ per-layer hot stores
```

`SchedulerOutput` carries one opaque `sparse_kv_offload` command instead of
separate HiSparse fields. The model runner passes it to the store and does not
interpret page transfers. The scheduler receives completed transfer IDs and
only then lets the cache manager release the source leases.

## Resident device pages

Resident pages are intentionally outside the store.

KV-cache initialization binds cache-manager allocations to the attention-facing
`HiSparseLayer` before constructing `HiSparseOffloadStore`. That same layer
retains the resident group ID needed by a transfer plan. There is no second
resident object or store registration wrapper.

```text
KV cache setup
   │
   ├─ bind resident allocation ──► HiSparseLayer
   │                               cache + block table + slot mapping
   │
   └─ construct offload store ───► HiSparseOffloadLayer
                                   host + hot + GPU LRU

HiSparseOffloadStore registers the same HiSparseLayer objects directly
```

Attention construction links each layer to the most recent layer that actually
owns an indexer. This releases a follower's duplicate LRU tensors before GPU
memory profiling. Cache binding only attaches storage; it does not infer
semantic groups from the physical packed-tensor order. The construction cursor
is discarded with the store's pinned state.

For a fully resident batch, `HiSparseLayer` uses the normal paged resident
cache directly. For a hybrid batch, the fused HiSparse kernel checks resident
pages first, then hot rows, then pinned host memory. No CPU decision is added
to the decode path. The resolver consumes the existing graph-stable request
mapping from attention metadata; neither the store nor individual layers keep
a duplicate mapping.

## Spill transaction

A resident block cannot be reused until its contents have been handed to the
store.

```text
HiSparseManager                            offload store
          │                                     │
          │ pin source and destination leases   │
          │── SparseKVPageTransfer ─────────────►│
          │                                     │ enqueue GPU-to-host copy
          │◄── completed transfer ID ───────────│
          │ mark host copy valid                │
          │ replace resident table entry        │
          │ release resident lease to HMA       │
```

“Completed” here means the copy was enqueued in stream order. A host-write
event protects later CPU readers.

The worker transfer contains only its completion ID and physical copy
coordinates. Request identity and logical page state remain in the scheduler.

## Hot lookup and LRU

The NVIDIA and AMD path keeps replacement entirely on the accelerator:

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

The abstraction does not require another platform to use this GPU LRU. A
platform-specific store may implement its own host/hot policy behind the same
command, output, and layer-resolution boundaries.

## Main classes

| Class | Inherits / implements | Responsibility |
| --- | --- | --- |
| `HiSparseManager` | plain scheduler component | host allocation, source prefixes, resident leases, and spill state machine |
| `HiSparseResidentManager` | `SingleTypeKVCacheManager` | normal block-pool bookkeeping with host-backed holes |
| `PagedCacheView` | immutable data object | shared resident/hot HMA tensor binding |
| `HiSparseOffloadStore` | plain worker component | host/hot ownership and step-level transfers |
| `HiSparseOffloadLayer` | plain store-owned component | per-layer host/hot tensors, GPU LRU, fused resolution |
| `HiSparseLayer` | plain attention component | resident view, route, and direct store registration |
| `SparseKVOffloadCommand` | dataclass | opaque scheduler-to-store work |

## Performance invariants

- Fully resident attention still reads ordinary paged GPU KV directly.
- Hot lookup, victim selection, and LRU updates stay on the GPU.
- A hot miss still copies directly from registered pinned host memory.
- Top-K resolution stays inside the attention invocation and remains graph
  capturable.
- Compatible layers still share one miss plan.
- Index-sharing followers release their private LRU state before memory sizing.
- Resident and hot leases can still share one packed HMA allocation.
- No device scalar readback or CPU/device metadata round trip is added.
- The abstraction wraps the fused kernel; it does not add another kernel
  launch.
- When HiSparse is disabled, the scheduler does not construct an offload
  command or empty update table.

## What remains platform-specific

The command/result and attention-layer boundaries can be shared. The host
allocator, copy implementation, hot layout, and replacement policy should stay
platform-specific. NVIDIA and AMD keep the current accelerator LRU and fused
host/hot kernel. Ascend can implement its own store without forcing that policy
into this path.
