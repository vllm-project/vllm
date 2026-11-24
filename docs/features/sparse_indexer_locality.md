# Die-Local Sparse Indexer and Indexer Overlap

Two opt-in decode optimizations for the sparse-attention indexer of Kimi K3.1 and
DeepSeek-V4.1 (DeepSeek Sparse Attention), both off by default.

| Environment variable | Default | What it does |
| -------------------- | ------- | ------------ |
| `VLLM_INDEXER_LOCALIZE` | `0` | Die-local indexer decode on GPUs with two memory locality domains |
| `VLLM_INDEXER_OVERLAP` | `0` | Overlap the indexer's top-k with independent work of the same layer |

```bash
VLLM_INDEXER_LOCALIZE=1 VLLM_INDEXER_OVERLAP=1 vllm serve <Kimi K3.1 or DeepSeek-V4.1 checkpoint> ...
```

## Die-local indexer (`VLLM_INDEXER_LOCALIZE`)

On GPUs whose memory and SMs form two locality domains (dies), such as NVIDIA
Rubin, a die reads its own memory faster than the other die's. In decode, the
indexer scans every cached index key of every request, so it is bound by memory
bandwidth. With `VLLM_INDEXER_LOCALIZE=1`:

- The KV cache is placed in runs of whole KV blocks, close to 32 MiB each, that
  alternate between the two locality domains. Block ids and the cache-insert
  kernels are unchanged; this applies to the extensible KV cache (the default)
  and to the plain allocation.
- Every decode step, one kernel (captured in the decode CUDA graph) splits each
  request's index-key pages by domain and builds both domains' DeepGEMM
  schedules.
- Each domain's SMs (a green context per domain) scan the pages in their own
  memory with DeepGEMM's `fp8_fp4_paged_mqa_logits`. With top-k 512, 1024 or
  2048, each domain also runs the stock selector's top-k (`cooperative_topk` up
  to 64 rows, `persistent_topk` above) on its own scores and one kernel merges
  the two candidate lists; otherwise the two score arrays are joined into the
  stock full-length array and the stock top-k runs.
- A decode batch takes the die split only when each domain's top-k fits in one
  round of CTAs on the domain's SMs: at most as many rows as the domain holds
  clusters of the top-k's size at once (16, 8, 4 or 2 CTAs per row up to 4, 8,
  33 or 64 rows, then 1), as CUDA's occupancy query reports for the domain's
  green context. On VR200 (100 SMs per domain) that is 1-23, 34-50 and 65-100
  rows; other batches run the stock indexer. The rule depends only on the row
  count, which a FULL CUDA graph fixes per captured batch size.

With speculative decoding, a request has one decode row per query token (the
last accepted token and the draft tokens), and each row sees one key more than
the one before. DeepGEMM reads the page list of a request's first row and scans
as far as its last row needs, so the build splits each request's pages once, up
to its last row, gives every row of the request the same per-domain page lists,
and gives each row its own per-domain lengths.

The scores are bit-identical to the single-launch scan, and the kept tokens have
the same score multiset as the stock top-k (ties may keep other tokens). The CUDA
kernels ship as source and are compiled with NVRTC at their first use, in well
under a second.

Requirements: a GPU with two locality domains, DeepGEMM, `cuda-bindings` 13.4 or
later (driver API and NVRTC), and a PyTorch build with locality-domain green
contexts and `torch.cuda.execute_on_streams` (nightly from 2026-10-08, 2.16).
Where any of these is missing, vLLM logs why and runs the stock indexer. The die
split and the KV cache placement are also off for models whose class does not
set `supports_localized_indexer` (today Kimi K3 / K3.1 and DeepSeek-V4.1), with
decode or prefill context parallelism, a KV connector, and with the CuMem
allocator on a plain (non-extensible) KV cache.

`VLLM_INDEXER_LOCALIZE` is independent of
[`VLLM_USE_LOCALITY_DOMAINS`](../configuration/env_vars.md), which splits the
NVFP4 MoE layers by locality domain; with both on, they share the per-domain
green contexts.

When active, each rank logs `KV cache placed in ... runs alternating between 2
locality domains`, the decode batch sizes that take the die split and, once per
indexer cache group, `Die-local sparse indexer active for N layers`. Each
fallback is logged once as a warning.

## Indexer overlap (`VLLM_INDEXER_OVERLAP`)

Decode steps captured as FULL CUDA graphs reorder independent work of the same
layer around the indexer; the kernels and their inputs are unchanged, so the
outputs are bit-identical. Eager steps and piecewise CUDA graphs keep the stock
order.

- Kimi K3.1: the indexer's scan and top-k run on another stream (with the die
  split, on the domains' streams) while the layer issues the KV norm, the query
  projection, the output gate, the MLA cache insert and the query preparation;
  sparse attention waits for the kept tokens. The indexer's key projection runs
  on a side stream beside the query chain.
- DeepSeek-V4.1: the candidate source layer selects its candidate blocks, and
  expands them into the sparse block ids of its first consumer layer, on a side
  stream that the consumer joins. The main query projection and cache inserts
  run beside the indexer, and the indexer query beside the compressor and the
  index key. With the die split, the candidate source layer also keeps its top-k
  per domain and joins the two score arrays as part of the deferred candidate
  work.

The overlap works with and without `VLLM_INDEXER_LOCALIZE`, and with
speculative decoding. It is off with micro-batching (`--enable-dbo` or
`--ubatch-size`), and for Kimi K3.1 with an attention backend whose top-k
conversion does not go through the shared sparse MLA index group.
