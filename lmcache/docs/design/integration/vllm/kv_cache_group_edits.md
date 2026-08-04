# KV Cache Group Edits

## Summary

`lmcache/integration/vllm/kv_cache_group_edits.py` is the single place where
vLLM KV cache groups are re-presented ("edited") before LMCache registration.
The connector calls `apply_kv_cache_group_edits(kv_cache_config, kv_caches)`
once in `register_kv_caches`, and the edited dict feeds both engine-group-info
creation (`kv_cache_groups.py`) and transfer registration.

LMCache derives each group's transfer metadata (block size, page layout,
dtype) from the registered tensors, and interprets store/retrieve block IDs in
vLLM's scheduler-side block-id space (`kv_cache_spec.block_size` units). The
edits exist to restore one invariant the raw tensors can violate:

> **The registered tensor's paging granularity must equal the block-id
> granularity.**

## Structure

Each case is one `KVCacheGroupEdit` rule in the module's `_EDITS` registry:
`matches(spec, kv_cache)` decides **structurally** — from the vLLM spec kind
(`get_kv_cache_spec_kind`, which also unwraps `UniformTypeKVCacheSpecs`) and
the registered tensor, never from model name or architecture — and
`apply(spec, kv_cache)` produces a view over the same storage. First matching
rule wins; unmatched layers pass through. Covering a new group kind means
adding one rule.

Model name/arch is deliberately not an input: the same architecture yields
different group structures depending on runtime decisions
(`mamba_cache_mode`, attention backend's kernel block size, TP), all of which
the config + tensors already resolve.

## Edits

Both rules apply only to Mamba-hybrid models (the registry is only consulted
when `kv_cache_config.has_mamba_layers`).

### 1. Mamba state pages

A Mamba / linear-attention layer (e.g. Qwen3.5 GDN) registers `[conv_state,
ssm_state]` — two tensors with different shapes and dtypes, laid out
contiguously in one padded page (`conv | ssm | pad`). The raw pair trips
format discovery (the SSM view starts mid-page). The edit reinterprets each
page as one bf16 tensor shaped `(num_blocks, 2, block_size, 1, head_size)`
over the same storage, where `head_size` is derived so the bytes fill the page
exactly.

### 2. Sub-paged full attention

vLLM unifies page sizes across hybrid groups by inflating the attention
*logical* block size (`vllm/platforms/interface.py:_align_hybrid_block_size`;
Qwen3.5-0.8B: 544), while the attention backend re-pages the physical tensor
at its own *kernel* block size (`vllm/v1/worker/utils.py:
prepare_kernel_block_sizes`; FlashAttention on hybrids: 32). Logical block `n`
then occupies the `k = logical/kernel` contiguous kernel pages
`n*k .. n*k+k-1` (vLLM expands the worker-side block table the same way,
`BlockTable.map_to_kernel_blocks`); the scheduler-side block IDs LMCache
receives stay logical.

Registering the raw kernel-paged tensor makes LMCache discover
`block_size == kernel < logical`, and `_derive_compression_metadata`
(`lmcache/v1/kv_layer_groups.py`) misclassifies the group as compressed:
only `1/k` of each chunk's KV is transferred, addressed against the kernel
page space. The edit re-views the tensor as
`(num_kernel_pages / k, 2, logical_block_size, 1, head_size)` — a pure
`view()`, valid because `k` kernel pages tile each logical page's bytes
exactly (enforced; see Invariants).

## Startup validation

`validate_kv_cache_groups` (called at connector init and again at
registration) rejects group specs the transfer path cannot serve correctly,
with one aggregated error: `CrossAttentionSpec`, and Mamba with
`mamba_cache_mode != "align"` (no reusable snapshots). Declared slot
compression (`compress_ratio > 1` / `tq_slot_size > 0`, e.g. DeepSeek-V4) is
*not* rejected — those groups are served by the compression path in
`lmcache/v1/kv_layer_groups.py` and only skipped by the edits here. Note the
compression path still derives per-group ratios from the unified vLLM block
size; switching it to per-group block sizes is pending in a separate PR.

Reference: vLLM PR #42828 (Mooncake store HMA support) uses the same
validate-and-reject-up-front pattern, and is the reference design for the
deferred follow-ups (per-group store/load masks; manager-mirroring hit
computation). Caveat for the latter: LMCache's lookup doubles as prefetch, so
vLLM's lookup-first-then-trim flow does not map directly. See the module
docstring for the full check-when list.

## Non-edit: declared compression

Groups whose spec *declares* slot compression — `MLAAttentionSpec.
compress_ratio > 1` (DeepSeek-V4 slot packing, `storage_block_size <
block_size`) or `TQFullAttentionSpec.tq_slot_size > 0` — genuinely store fewer
physical slots than logical tokens. They must reach the compression path in
`lmcache/v1/kv_layer_groups.py` unedited. (DeepSeek-V3.2's `fp8_ds_mla` cache
packs *bytes per slot*, not slots per block: its specs keep
`block_size == scheduler block size` and `compress_ratio == 1`, so it never
needs an edit either.)

The sub-paged rule's `matches` excludes declared-compression specs by their
own fields (`compress_ratio` / `tq_slot_size`), and its `apply` additionally
verifies by byte accounting that `k` kernel pages tile the logical page's
bytes exactly — any *undeclared* packed layout fails with a loud `ValueError`
rather than being transferred wrongly.

## The opaque-page contract

An edited view's dims are addressing metadata only (block id → byte range).
The named dims are **not** semantic: a Mamba view's "K plane" is conv/ssm
bytes, and a sub-paged attention view's "K plane" interleaves true K and V at
kernel-page granularity (true K is not contiguous across kernel pages, so no
logical-block view can have a pure-K plane). The synthetic head shape
`(1, page_bytes / (2 * block_size * elem))` signals this deliberately.

Byte transport round-trips correctly because store and retrieve share the same
bijective block-id → bytes mapping. Consequences:

- **Valid**: store/retrieve through the MP transfer path on the same engine
  configuration.
- **Not valid** for edited groups: content-aware processing (serde
  compression, blending, head resharding, layout conversion), and sharing
  cache entries across engines whose attention backends choose different
  kernel block sizes (the byte order inside a logical page is
  backend-dependent).

## Invariants

- Edits are pure tensor views over the registered storage — never copies.
- A sub-paged view is only produced when `kernel_page_bytes * k ==
  spec.page_size_bytes`; any mismatch raises `ValueError` (fail loudly rather
  than silently transfer a compressed layout).
- After edits, every registered tensor's block dim equals its group's
  `kv_cache_spec.block_size`, so the server derives `compress_ratio == 1` for
  these groups.

## Code map

| Area | File |
|---|---|
| Edits (this doc) | `lmcache/integration/vllm/kv_cache_group_edits.py` |
| Caller | `lmcache/integration/vllm/lmcache_mp_connector.py` (`register_kv_caches`) |
| Compression-ratio derivation (downstream consumer) | `lmcache/v1/kv_layer_groups.py` |
| vLLM block-size inflation | `vllm/platforms/interface.py` (`_align_hybrid_block_size`) |
| vLLM kernel-page split + block-table expansion | `vllm/v1/worker/utils.py`, `vllm/v1/worker/block_table.py` |
| End-to-end test | `.buildkite/k3_tests/multiprocess/scripts/run-single-test.sh` (`hma_lm_eval_qwen3_5`) |

Testing is end-to-end only (the `hma_lm_eval_qwen3_5` store-vs-retrieve gsm8k
check): the edit internals are expected to change as more group kinds are
covered, so tests pin the observable contract — faithful retrieve — rather
than view shapes.
