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

`host_pool_gib` is configured on `HiSparseConnector` and is the usable
host-cache capacity per data-parallel replica, not a node-wide memory budget.
MLA tensor-parallel ranks hold replicated views of that logical cache, using
private per-rank backing or one shared physical allocation. QSA stores full K/V
heads in private per-rank pools, with each rank writing its own shard. Its
logical capacity counts each unique K/V head once; heads replicated when TP
exceeds the K/V head count still consume physical memory in each rank's pool.
Padding also contributes to physical allocation, not logical capacity. Physical
host memory consumption is therefore topology- and implementation-dependent. The
realized capacity may be slightly smaller because the budget is rounded down
to complete host blocks.

Startup logs report two concurrency bounds at `max_model_len`. The generic
`Maximum concurrency` line charges each request its full admission footprint,
including the in-flight window of every resident group. The `HiSparse
steady-state maximum concurrency` line charges running requests that read from
host only their active tail pages, plus one request being admitted at its full
footprint. Host-pool metrics are listed in
[Metrics](../usage/metrics.md#hisparse-kv-connector-metrics).

## QSA configuration and validation scope

For a model using the Qwen4Exp QSA implementation, configure `HiSparseConnector`
to enable main K/V offloading. This also selects Model Runner V2 and supplies
default `HiSparseConfig` values. For example, replace `QSA_MODEL` with the model
identifier or checkpoint path:

```bash
vllm serve QSA_MODEL \
  --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --attention-config '{"indexer_kv_dtype":"bf16"}' \
  --enforce-eager --no-async-scheduling \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --max-model-len 8192 --max-num-batched-tokens 8192 \
  --kv-transfer-config '{"kv_connector":"HiSparseConnector","kv_role":"kv_both","kv_connector_extra_config":{"host_pool_gib":8}}'
```

This example selects eager execution, synchronous scheduling, and BF16 main
K/V and indexer keys, with MTP, prefix caching, and chunked prefill disabled.
Choose the context length and host capacity for the workload. The hybrid KV
cache manager must remain enabled. Add `--tensor-parallel-size` as appropriate
for the model and device count; `host_pool_gib` remains a logical capacity per
DP replica.

HiSparse QSA automatically resolves the device layout to `BLNHC`, so one copy row
contains a token's K and V for every local K/V head. The host pool uses compact
token rows within each layer. HiSparse preserves the indexer's sparse selection;
compressed indexer keys, raw-key rings, and recurrent state remain GPU-resident.

The NVIDIA CUDA single-engine implementation has been checked in the following
configurations. The rows describe complementary checks; they do not imply that
every combination of dtype, execution mode, and topology has been tested.

| Path | Validated configuration and behavior |
| --- | --- |
| Eager decode | Synchronous TP2 execution with MTP disabled; full K/V offload, physical page reuse, and host refill with private per-rank K/V shards |
| Cache dtypes | BF16 model/query dtype with independently selected BF16 or FP8 main K/V and indexer formats; all four combinations cover original scales, complete K/V rows, nonresident prefill, and selected-K/V replay, with matching eager on/off model evaluations |
| MTP, FULL replay, and async scheduling | TP2 with three draft tokens, shared logical selection, native `FULL_DECODE_ONLY` replay, and async scheduling; real page reuse and host refill cover target verification and draft decoding, including acceptance/rejection, padding, and stable replay storage |
| Prefix and request lifecycle | BF16 at TP1 with MTP disabled and `FULL_DECODE_ONLY` plus async scheduling; chunked prefill, shared prefixes with private writable tails, cancellation of a prefix-sharing request, and natural preemption/recompute with state invalidation and resource return |

Use `FULL_DECODE_ONLY` or `FULL_AND_PIECEWISE` for full decode graphs.
HiSparse rejects `cudagraph_mode=FULL`, which also requires full prefill graphs.

The dtype representation checks and lifecycle checks exercise their respective
shared paths; they do not constitute a full dtype-by-lifecycle matrix. Existing
backend restrictions on recurrent-state formats and supported parallelism still
apply. Related MLA HiSparse and non-offload QSA paths have regression coverage.

HiSparse preserves the indexer's selection budget, logical positions, and valid
counts in these paths. It does not change the dtypes or ownership of raw indexer
rings and recurrent state. Model evaluations compare each offload configuration
against its matching non-offload configuration, with the same prompts, decoding
settings, and output budget. Reports retain per-response finish reasons, including
responses that reach the output limit. Results for one dtype or execution mode do
not establish the others. Cache-correctness checks and model-quality evaluations
do not establish throughput, latency, or capacity gains; those require separate
workload-specific measurements.

NIXL P/D integration for QSA is separate follow-up work, including transfer and
lifecycle handling for main K/V, indexer caches, and hybrid state. The P/D import
and generic indexer offloading sections below describe the connector
architecture; they do not establish validated QSA P/D combinations.

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

For MLA with single-node MP tensor parallelism, every TP worker maps the same
pinned host pool and uses the same block and layer offsets. Source KV is replicated
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

Speculative decoding resolves all verification rows of a request in one pass:
one block resolves the union of the rows' top-k against the request's hot-cache
state, so rows that select the same host row share its hot row and no row
evicts a hot row another row of the step still reads. Draft layers write their
rows after the target forward, so their host mirror runs at the start of the
next step, after the drafter, and the step's page transfers are submitted
behind it.

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
offloading path; HiSparse continues to own only the main sparse-attention host tier.

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
