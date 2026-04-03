# MoE Expert Weight Caching

vLLM can run MoE models that exceed available GPU memory by keeping all expert
weights in CPU pinned memory and caching only the most-recently-used
experts in a fixed-size GPU scratch buffer.

This feature is controlled by the `--moe-expert-cache-size` option.

| Option | Default | Description |
| --- | --- | --- |
| `--moe-expert-cache-size N` | `0` (disabled) | Number of expert slots to allocate in the GPU buffer per layer |

!!! note
    Expert caching requires `--enforce-eager`. CUDA graph capture is
    incompatible with the dynamic Python bookkeeping in `prepare()`.

!!! note
    Expert caching is not compatible with expert parallelism (EP > 1),
    data parallelism, or sequence parallelism.

## Quick start

```bash
# OLMoE-1B-7B: 64 experts, fits on 8 GB GPU with 16 cached per layer
vllm serve allenai/OLMoE-1B-7B-0924 \
    --moe-expert-cache-size 16 \
    --enforce-eager
```

### Python API

`moe_expert_cache_size` is exposed as a direct `LLM` constructor parameter:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="allenai/OLMoE-1B-7B-0924",
    moe_expert_cache_size=16,
    enforce_eager=True,
)
```

## Architecture (RFC #38256)

The cache is implemented as a `CachedWeightProvider` — the kernel does not
know or care where weights came from.

### How it works

```text
Decode (unique experts <= capacity) — GPU fast path:
  topk_ids -> provider.prepare():
    hit  -> move_to_end in OrderedDict  (O(1))
    miss -> evict LRU, H2D copy, update mapping[expert] = slot
  -> kernel.apply(result.w1, result.w2, result.topk_ids)
```

A persistent `_mapping` tensor (`int32`, GPU) holds the `expert_id -> slot`
mapping. It is updated in-place for misses and used for a vectorized remap —
no CPU tensor build or H2D transfer on the hot path.

The `CachedWeightProvider` uses `collections.OrderedDict` for LRU eviction
(no external dependencies). When unique experts exceed capacity, a
`RuntimeError` is raised — increase `--moe-expert-cache-size` to avoid this.

## Observability

### DEBUG-level hit/miss log

Set `VLLM_LOGGING_LEVEL=DEBUG` to get a per-layer hit/miss report every
60 seconds:

```text
DEBUG vllm...expert_weight_provider: Expert cache: 1234 hits, 56 misses (95.7% hit rate)
```

## Sizing guidance

Set `--moe-expert-cache-size` to the number of experts that must fit on
GPU simultaneously per layer. For a model with `E` experts and `top_k`
routing:

- **Minimum useful**: `top_k` (one slot per active expert per token, no
  eviction during decode)
- **Typical decode**: `2 * top_k` – `4 * top_k` gives headroom for
  locality without wasting VRAM
- **Maximum** (no-op): `E` (all experts on GPU, equivalent to normal mode)

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
