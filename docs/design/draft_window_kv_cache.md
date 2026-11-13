# Draft-Window KV Cache

Some target-conditioned draft models use only a bounded trailing context even
when the target model attends to the full prefix. Giving both models one KV
block pool ties their lifetimes together: the target needs stable full-prefix
blocks, while the drafter can recycle blocks outside its window.

vLLM automatically separates these lifetimes for compatible DFlash and DSpark
models. There is no user-facing switch. In unsupported configurations, vLLM
uses the regular shared KV-cache layout.

## Compatibility

The independent draft-window pool is enabled when all of the following hold:

- the DFlash or DSpark speculator consumes target hidden states;
- target groups use full-attention KV;
- every draft KV group uses one common sliding-window geometry and the same
  block size as the target groups;
- the hybrid KV-cache manager is enabled;
- pipeline, prefill-context, and decode-context parallel sizes are all one;
- HiSparse and model-defined custom KV grouping are not enabled;
- KV-cache residency metrics are disabled;
- KV-cache event publishing is disabled;
- `num_gpu_blocks_override` is not set;
- KV transfer is disabled or uses only Mooncake P2P/Store connectors.

Tensor and data parallel deployments are supported. The automatic extensible
KV cache is disabled for eligible configurations because its current resize
operation assumes one capacity shared by every KV group. If the extensible
cache is explicitly enabled, it takes precedence and vLLM retains the shared
layout. Other KV-transfer connectors retain the shared layout until
allocator-local block identities and sparse recovery have been audited for
those connectors.

## Storage

Target and draft groups have independent block allocators and backing tensors.
The target pool is sized from the memory left after reserving the draft pool.
The draft pool has a fixed upper bound derived from:

- each draft group's sliding-window size and retained tokens;
- speculative lookahead and in-flight tokens;
- the configured maximum number of sequences; and
- block-boundary and null-block space.

The bound is capped by the storage needed for full-length requests. It does not
add an arbitrary retention margin.

Block identifiers are local to their owning pool. Scheduler metadata therefore
keeps the pool identity for copy-on-write and block-zeroing operations.

## Prefix recovery

A target prefix hit can extend beyond the pages still resident in a draft
window. vLLM preserves the target hit and plans recovery independently for each
draft group:

1. Compute the historical token range needed by each draft window at the next
   draft boundary.
2. Reuse any cached draft pages in that range.
3. Find the first missing page across the draft groups.
4. Replay one continuous suffix from that page to the target hit boundary.

Continuous replay is intentional. It restores a valid attention history while
avoiding the target model's full common-endpoint replay. During replay, target
slot mappings below the preserved hit boundary are masked, so only draft KV is
rewritten.

This protection is based on the known target hit boundary. It does not maintain
an endpoint cache or add endpoint-specific eviction state.

## Prefill/decode transfer

With Mooncake P/D disaggregation, the consumer reports the logical positions
of the pages it needs after applying its local Draft window. The producer maps
those positions to the retained pages instead of assuming both endpoints own
the same dense suffix. Positioned metadata that is missing, inconsistent, or
outside the producer's retained range is rejected rather than suffix-aligned.

Main and Draft region registration follows each cache group's physical pool.
Blocks remain pinned to their owning pool until the relevant workers report
completion. Mooncake Store uses the same pool identity when registering,
pinning, and releasing blocks.

Heterogeneous-TP transfer follows cache-spec layout metadata. Head-sharded
regions are split or gathered according to the producer and consumer TP sizes,
while TP-replicated regions are sent by one producer rank. NVFP4 head
re-sharding is enabled only when the cache spec declares byte-sliceable
head-major slots and all layout checks pass.

## Correctness fallbacks

The optimization is not used when its storage or execution assumptions cannot
be proved from the resolved configuration. Unsupported configurations continue
to use the existing shared-pool path.
