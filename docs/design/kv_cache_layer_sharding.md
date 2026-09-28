# Layer-sharded KV cache (KVPP)

KVPP stores each replicated MLA cache bundle on one rank in the PCP x TP replica
group. Every rank still runs every layer. A separate NCCL process group broadcasts the
owner's bundle into reusable receiver storage immediately before attention
needs historical KV. Logical blocks, prefix hashes, request admission, and
scheduler reference counts retain their existing meanings.

## Enablement and limits

Use `--enable-kvpp` with PCP x TP >= 2 and Model Runner V2
(`VLLM_USE_V2_MODEL_RUNNER=1`). KVPP is disabled by default
(`CacheConfig.enable_kvpp=False`). KVPP on GPU rejects Model Runner V1. This
implementation supports NVIDIA CUDA and NCCL broadcast; it does not expose a
transport selector.

Choose eager execution with `--enforce-eager`, or compiled piecewise execution:

```text
-cc.mode=3 -cc.cudagraph_mode=PIECEWISE -cc.use_inductor_graph_partition=false
```

Keep the default `splitting_ops`: attention and KV cache updates must remain
outside captured graphs so cache acquisition, prefetch, and release run on every
replay. FULL, FULL_AND_PIECEWISE, breakable graphs, and Inductor graph partitioning
are not enabled for KVPP. The isolated GSM8K guard uses compiled PIECEWISE.

For example, combine `--tensor-parallel-size 2` with
`--prefill-context-parallel-size 2` to distribute bundles across four replica
ranks. PCP MoE currently requires `VLLM_MOE_SKIP_PADDING=0`: its token gather
does not gather the rank-local routing padding mask. This existing PCP limit
also applies with KVPP disabled; the GSM8K guard sets this environment variable.

The first version requires replicated full-attention MLA caches, a common block
size, a layer-compact layout, and at least one owned bundle per replica rank.
A cache layer opts in with `AttentionLayerBase.get_kv_cache_bundle()`. DeepSeek MLA
returns its main cache and, when present, its sparse-indexer cache as one bundle.
With DCP=1, PCP gathers new prefill KV before the ordinary cache update, so every
PCP rank retains a full replica. KVPP reuses that path unchanged. PCP retains its
Model Runner V2 capability limits, including no PCP + PP execution yet.
Draft caches remain local. DCP, DBO, HiSparse, and cross-layer KV sharing are
rejected until their storage and lifetime contracts are implemented. Ordinary
TP-sharded K/V heads cannot use owner-retains-updates.

`SimpleCPUOffloadConnector` can offload owned persistent cache; other KV
connectors fail closed. External or remote connectors need an ownership manifest
and a distributed completion contract before they can support this placement.

## Placement and capacity

Workers discover bundle order from the loaded model. Contiguous, count-balanced
partitions choose owners within the PCP x TP replica group. Before cache
allocation, the engine checks that all ranks agree on logical cache specs, bundle order, and
ownership. `KVCacheStoragePlan` attaches worker-local physical regions to the
otherwise ordinary `KVCacheConfig`.

The allocator places owned bundles in persistent ranges and nonowner bundles in
two alternating scratch slots. Bundle components are packed contiguously in one byte
arena; attention still sees stable tensor views. Capacity search uses the same
layout builder as final allocation. It converts each worker's physical memory
budget to a logical block capacity; the common KV cache config flow then handles
override, auto-fit, null-block reservation, admission checks, and the minimum
block count across workers. The ordinary config builder, including its layout
validation and tensor descriptors, runs unchanged. Once the common block count
is final, KVPP replaces only physical tensor placement and attaches the storage
plan; other config fields are preserved. Allocation uses the exact packed size,
and an override larger than physical capacity is rejected.

Placement does not add a scheduler or block manager. The allocator uses the same
backing allocation and view construction for both placements. KVPP adds no
alignment padding beyond the cache specs. Broadcast and offload consume the final
worker storage plan instead of deriving ownership or scratch placement again.

The offload path registers only persistent owner views. A common per-block byte
budget across ranks keeps distributed CPU/disk block IDs aligned even when owner
partitions have unequal sizes. Scratch does not consume offload pool
capacity.

## Attention-time broadcast

The first historical bundle broadcasts when its first cache component is
acquired, immediately before attention uses it. Acquiring bundle `i` waits for
its broadcast and launches bundle `i+1` on a separate CUDA stream. The first
acquisition can be the sparse indexer or MLA cache update. Each broadcast sends
the complete allocated bundle, including inactive blocks. Batches without
history skip broadcasts. History is read from the scheduled requests before PCP
partitioning: a rank-local query offset can include tokens from the current
prefill and must not be treated as historical KV.

CUDA events enforce three dependencies: the transfer waits for preceding forward
cache writes, attention waits for its bundle's transfer, and reusing a scratch
slot waits for its previous attention use. The owner's persistent source and
receiver scratch destination therefore remain valid until NCCL finishes. The
runtime checks ordered bundle access and requires all bundles from the previous
forward to be released before preparing the next one.

`KVPPRuntime` is selected through `Platform.get_kvpp_runtime_cls()` and scoped to
`ForwardContext.kvpp_runtime` by `set_forward_context`. After the connector's
pre-forward call, `maybe_prepare_kvpp` prepares the runtime and determines whether the
batch has history. There is no per-forward teardown; failed forwards retain their
state, and runner shutdown synchronizes the device before releasing cache storage.
The KVPP group is created alongside TP in `initialize_model_parallel` only when
KVPP is enabled and destroyed by `destroy_model_parallel`. It spans PCP x TP
within each DP replica and PP stage, with local rank `pcp_rank * tp_size + tp_rank`
and a separate communicator. The GPU worker warms up its broadcast through
`warmup_process_group` after distributed initialization and before the initial
memory snapshot. The runtime uses its device group for asynchronous broadcasts.
Device-specific runtime implementations provide communication operations and events while group
lifecycle remains in `parallel_state`.

## Validation

Module tests cover placement agreement, packed capacity, scratch aliases,
offload's persistent-only views, broadcast lifetimes across PCP x TP within each
PP stage, and communication-group teardown and reinitialization. The isolated
`DeepSeek-V2-Lite-Chat` GSM8K config under `tests/evals/gsm8k/configs/` is the
real-weight end-to-end guard. Evaluate prefill, decode, prefix reuse, and pool
reload separately when expanding supported execution modes.
