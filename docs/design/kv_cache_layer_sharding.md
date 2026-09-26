# Layer-sharded KV cache (KVPP)

KVPP stores each replicated MLA cache bundle on one tensor-parallel rank. Every
rank still runs every layer. A separate NCCL process group broadcasts the
owner's bundle into reusable receiver storage immediately before attention
needs historical KV. Logical blocks, prefix hashes, request admission, and
scheduler reference counts retain their existing meanings.

## Enablement and limits

Use `--kv-cache-placement layer_sharded --enforce-eager` with TP >= 2 and Model
Runner V2 (`VLLM_USE_V2_MODEL_RUNNER=1`). The default `replicated` placement is
unchanged. KVPP on GPU rejects Model Runner V1. This implementation supports
NVIDIA CUDA and NCCL broadcast; it does not expose a transport selector.

The first version requires replicated full-attention MLA caches, a common block
size, a layer-compact layout, and at least one owned bundle per TP rank. A cache
layer opts in with `AttentionLayerBase.get_kv_cache_bundle()`. DeepSeek MLA
returns its main cache and, when present, its sparse-indexer cache as one bundle.
Draft caches remain local. DCP, PCP, DBO, HiSparse, cross-layer KV sharing, and
graph execution are rejected until their storage and lifetime contracts are
implemented. Ordinary TP-sharded K/V heads cannot use owner-retains-updates.

`SimpleCPUOffloadConnector` can offload owned persistent cache; other KV
connectors fail closed. External or remote connectors need an ownership manifest
and a distributed completion contract before they can support this placement.

## Placement and capacity

Workers discover bundle order from the loaded model. Contiguous, count-balanced
partitions choose owners within the TP replica group. Before transport creation,
the engine checks that all ranks agree on logical cache specs, bundle order, and
ownership. `KVCacheStoragePlan` attaches worker-local physical regions to the
otherwise ordinary `KVCacheConfig`.

The allocator places owned bundles in persistent ranges and nonowner bundles in
two alternating scratch slots. Bundle components use aligned offsets in one byte
arena; attention still sees stable tensor views. Capacity search uses the same
layout builder as final allocation and chooses the minimum feasible logical block
count across ranks. Allocation includes alignment padding. The scheduler retains
the usual null block and rejects an override larger than physical capacity.

The offload path registers only persistent owner views. A common per-block byte
budget across ranks keeps distributed CPU/disk block IDs aligned even when owner
partitions have unequal sizes. Scratch and padding do not consume offload pool
capacity.

## Attention-time broadcast

The first historical bundle broadcasts when its first cache component is
acquired, immediately before attention uses it. Acquiring bundle `i` waits for
its broadcast and launches bundle `i+1` on a separate CUDA stream. The first
acquisition can be the sparse indexer or MLA cache update. Each broadcast sends
the complete allocated bundle, including inactive blocks. Batches without
history skip broadcasts.

CUDA events enforce three dependencies: the transfer waits for preceding forward
cache writes, attention waits for its bundle's transfer, and reusing a scratch
slot waits for its previous attention use. The owner's persistent source and
receiver scratch destination therefore remain valid until NCCL finishes. The
runtime checks ordered bundle access and rejects concurrent forwards sharing the
same scratch arena.

The CUDA runtime is selected through `Platform.get_kv_cache_runtime_cls()` and
scoped to `ForwardContext`. This lets a future device implementation supply its
own communicator and events while keeping logical scheduler state and model
bundle declarations independent of the transport.

## Validation

Module tests cover placement agreement, capacity and alignment, scratch aliases,
offload's persistent-only views, and multi-rank broadcast lifetimes. The isolated
`DeepSeek-V2-Lite-Chat` GSM8K config under `tests/evals/gsm8k/configs/` is the
real-weight end-to-end guard. Evaluate prefill, decode, prefix reuse, and pool
reload separately when expanding supported execution modes.
