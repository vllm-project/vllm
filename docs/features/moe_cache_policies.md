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

## MXFP4 support

The expert cache supports MXFP4 (Microscaling FP4) expert weights, used by
DeepSeek V4 Flash and PRO. MXFP4 block scales (shape `[E, output_dim, input_blocks]`)
are handled natively — the `CachedWeightProvider` slices on dimension 0 regardless
of tensor rank.

### Monolithic kernel limitation

DeepSeek V4's default MXFP4 MoE kernel is **monolithic**: it performs expert
routing internally and expects `global_num_experts = total_experts`. The expert
cache remaps expert IDs to slot indices, which is incompatible with monolithic
routing.

When `--moe-expert-cache-size N > 0` is used with MXFP4, the backend automatically
re-selects with `prefer_modular=True` to obtain a modular kernel class that accepts
externally-provided `topk_ids`. For TRITON this selects `OAITritonExperts`; for
FLASHINFER TRTLLM this selects `TrtLlmMxfp4ExpertsModular`.

If no modular backend is available (e.g. MARLIN), a clear error is raised:
```
ValueError: --moe-expert-cache-size=N requires a modular MXFP4 MoE kernel,
but backend ... only provides monolithic kernels.
```

### 3D scale tensors

MXFP4 stores per-expert block scales as 3D tensors:
`[num_experts, output_dim, input_dim // block_size]`. The cache copies these
alongside weight tensors on a per-expert basis, preserving the full shape.

## Tests

```bash
# Unit tests: CachedWeightProvider
pytest tests/kernels/moe/test_expert_lru_cache.py -v
```
