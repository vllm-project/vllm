# MoE Expert Weight Caching

vLLM can run MoE models that exceed available GPU memory by keeping all expert
weights in CPU pinned memory and caching only the most-recently-used
experts in a fixed-size GPU scratch buffer.

| Option | Default | Description |
| --- | --- | --- |
| `--moe-expert-cache-size N` | `0` (disabled) | Number of expert slots to allocate in the GPU buffer per layer |
| `--moe-expert-cache-split` | `token` | How to evaluate a forward that needs more experts than the cache holds: `token` or `expert` |

!!! note
    Expert caching is not compatible with expert parallelism (EP > 1),
    data parallelism, or sequence parallelism.

## Quick start

```bash
# OLMoE-1B-7B: 64 experts, fits on 8 GB GPU with 16 cached per layer
vllm serve allenai/OLMoE-1B-7B-0924 \
    --moe-expert-cache-size 16
```

### Python API

`moe_expert_cache_size` is exposed as a direct `LLM` constructor parameter:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="allenai/OLMoE-1B-7B-0924",
    moe_expert_cache_size=16,
    moe_expert_cache_split="token",  # or "expert"
)
```

## Architecture (RFC #38256)

The cache is implemented as a `CachedWeightProvider` — the kernel does not
know or care where weights came from.

### How it works

```text
Everything fits (the common case, including every decode step):
  topk_ids -> provider.prepare():
    hit  -> bump the expert's frequency and recency
    miss -> evict the lowest-scoring expert, H2D copy into its slot
  -> kernel.apply(result.w1, result.w2, expert_map=result.expert_map)
```

Eviction is LFRU rather than plain LRU: an entry scores `freq / age`, so a
rarely used expert loses to a frequently used one even if the latter was
touched longer ago. Pure LRU behaves badly here, because MoE layers run in
sequence and early layers always look recently used.

The provider hands the kernel an `expert_map` — global expert id to buffer
slot, or `-1` for experts that are not resident — which is the same convention
expert parallelism uses, so `topk_ids` is passed through untouched. The map
lives in a GPU `int32` tensor with a pinned host mirror, rebuilt per call and
uploaded once.

### When the cache is smaller than a forward needs

A single forward can route to more distinct experts than the cache holds; a
batched prefill routinely routes to nearly all of them. `--moe-expert-cache-split`
picks how that forward is evaluated.

`token` (default) splits the batch by rows into chunks that fit, and
concatenates. Each token's full sum still happens inside one kernel call, so
**output is identical to running without the cache**. The cost is one kernel
launch per chunk, and as the cache approaches `top_k` no two tokens can share a
chunk — a long prefill can end up with one launch per token.

`expert` splits the expert set instead: run the whole batch once per group of
at most `--moe-expert-cache-size` experts, with the map hiding the rest, and
sum the results. That is `ceil(experts_used / cache_size)` launches whatever
the batch size, and each expert is fetched at most once per forward instead of
being refetched as the cache thrashes. **Output differs from the uncached path
at rounding level**, because each group's partial sum is rounded to the model
dtype before being accumulated.

Either way the cache must hold at least `top_k` experts — one token's experts
have to be resident together, which no split can avoid. That is checked at
startup.

## CUDA graphs

The cache is compatible with piecewise CUDA graphs, which is the default
when `--moe-expert-cache-size` is set without `--enforce-eager`. vLLM adds
the MoE op (`vllm::moe_forward`) to `splitting_ops` and caps
`cudagraph_mode` at `PIECEWISE`: every non-MoE segment of the model is
captured and replayed, while the MoE layer — routing, cache management,
H2D copies, and the grouped GEMM — runs eagerly between segments. The
cache's GPU buffers and its `expert_map` are allocated once and updated
in place, so captured segments never observe a stale address.

Full-graph capture (`FULL`, `FULL_AND_PIECEWISE`) is not supported: a
capture would freeze one forward's cache state into every replay.
`--enforce-eager` remains available and is required when compilation is
disabled (`-O0`).

## Known costs

Each MoE layer performs one device sync per forward to read the routing
decision onto the host (`topk_ids.unique()`); this is the latency floor of
the current design. Overlapping it with a routing-ahead prefetch is
planned as a follow-up (see RFC #38256).

## Observability

### DEBUG-level hit/miss log

Set `VLLM_LOGGING_LEVEL=DEBUG` to get a per-layer running hit/miss total,
logged every 1000 `prepare()` calls:

```text
DEBUG vllm...expert_weight_provider: Expert cache: 1234 hits, 56 misses (95.7% hit rate)
```

Read the hit rate as a locality signal only under `--moe-expert-cache-split
token`. Under `expert`, a forward wider than the cache walks the whole expert
set group by group, so the rate mostly reports how often that happened rather
than how well the cache is sized — and the eviction policy has little to do
either, since each group displaces the last.

## Sizing guidance

Set `--moe-expert-cache-size` to the number of experts that must fit on
GPU simultaneously per layer. For a model with `E` experts and `top_k`
routing:

- **Minimum useful**: `top_k` (one slot per active expert per token, no
  eviction during decode)
- **Typical decode**: `2 * top_k` – `4 * top_k` gives headroom for
  locality without wasting VRAM
- **Maximum** (no-op): `E` (all experts on GPU, equivalent to normal mode)

Below roughly `E / 2` a batched prefill will start needing more experts than
fit, and `--moe-expert-cache-split` starts to matter. Keep the default `token`
unless prefill latency is the problem; switch to `expert` when it is, and
verify the quality impact for your model rather than assuming it is negligible.

## GPU memory note

Expert weights in CPU pinned memory are invisible to the `--gpu-memory-utilization`
profiler. The profiler will underestimate available KV cache headroom by the
expert weight footprint (a safe margin, not a hazard), but exact
`gpu-memory-utilization`-based sizing will be off.

## Tests

```bash
# Unit tests: CachedWeightProvider
pytest tests/kernels/moe/test_expert_lru_cache.py -v
```
