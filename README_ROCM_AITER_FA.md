# ROCM_AITER_FA Attention Backend

The `ROCM_AITER_FA` backend accelerates attention on AMD MI300X (CDNA3+) GPUs
using kernels from [ROCm/aiter](https://github.com/ROCm/aiter). It is the
preferred backend for GQA/MHA workloads on ROCm when AITER is available.

## Enabling

```bash
vllm serve <model> --attention-backend ROCM_AITER_FA
```

Or via automatic selection (default when `VLLM_ROCM_USE_AITER_MHA=1`, which is
the default):

```bash
VLLM_ROCM_USE_AITER_MHA=1 vllm serve <model>
```

## Kernel Dispatch

The backend dispatches to different AITER kernels depending on the decode
query length:

### Single-token decode (`decode_max_query_len == 1`)

Normal autoregressive decoding. Dispatches to:

- **ASM PA kernel** (`pa_fwd_asm`) — hand-tuned assembly for MI300X, requires
  `head_size == 128`. Supports both NHD and shuffle KV cache layouts.
- **HIP PA kernel** (`paged_attention_rocm`) — fallback for other head sizes or
  workload heuristics.
- **`paged_attention_common`** — wrapper that selects between ASM and HIP based
  on head size, batch size, and precision settings.

### Multi-token decode (`decode_max_query_len > 1`)

Speculative decoding verification, sliding window, or attention sinks.
Dispatches to AITER Triton kernels:

- **Causal**: `aiter.ops.triton.unified_attention.unified_attention`
- **Non-causal**: `aiter.ops.triton.attention.mha_v3.flash_attn_with_kvcache`

These kernels handle arbitrary query lengths with standard NHD paged layout.

### Prefill / Extend

Uses AITER's CK-based flash attention:

- `aiter.flash_attn_varlen_func` — composable kernel flash attention with
  variable-length sequences.

## Shuffle KV Cache Layout

The shuffle layout rearranges K/V cache data into a tiled format optimized for
the ASM PA kernel:

```
K: [num_blocks, num_kv_heads, head_size // x, block_size, x]
V: [num_blocks, num_kv_heads, block_size // x, head_size, x]
```

where `x = 16 // element_size`.

### Enabling

```bash
VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1 vllm serve <model> --attention-backend ROCM_AITER_FA
```

Default: disabled (`0`).

### Limitations

The shuffle layout only works with single-token decode
(`decode_max_query_len == 1`). It is automatically disabled (with a warning)
when any of the following are active:

- **Speculative decoding** — verification requires multi-token decode
- **Sliding window attention**
- **Attention sinks**

The fused RoPE+KV cache update path is also disabled when shuffle is active
(uses a separate `reshape_and_cache_shuffle_triton` write path instead).

> **Note**: AITER's `unified_attention` Triton kernel does accept a
> `shuffled_kv_cache` parameter, but vLLM does not currently pass it in the
> multi-token decode path. This is a potential future optimization.

## Speculative Decoding

`ROCM_AITER_FA` supports speculative decoding (Eagle, Eagle3, etc.). During
verification, `decode_max_query_len = 1 + num_speculative_tokens`.

### Why not use the ASM PA kernel for verification?

The ASM kernel's [`pa_fwd_asm`](https://github.com/ROCm/aiter/blob/a16ffe4db50f3e7698539d9bbf664bc7131ac664/aiter/ops/attention.py#L129-L144)
accepts `max_qlen` and `qo_indptr` parameters for multi-token queries, but has
a tile constraint: [`mtp < PA_TILE_Q // gqa`](https://github.com/ROCm/aiter/blob/a16ffe4db50f3e7698539d9bbf664bc7131ac664/op_tests/test_pa_decode_bf16_asm.py#L45-L46).
For the common config (tq16, gqa=8), this limits `qlen ≤ 2`.

Eagle3 with `num_speculative_tokens: 3` produces `decode_max_query_len = 4`,
exceeding this limit. Therefore verification dispatches to AITER's Triton
`unified_attention` which handles arbitrary query lengths.

Additionally, vLLM's wrapper (`rocm_aiter_ops.paged_attention_common`) does not
pass `max_qlen`/`qo_indptr` to the kernel, defaulting to `max_qlen=1`.

### Acceptance rate with speculative decoding

ROCM_AITER_FA may show a slightly lower speculative token acceptance rate
(~5pp) compared to TRITON_ATTN. This is inherent to using different kernels
for generation (ASM PA, single-token) vs verification (Triton
`unified_attention`, multi-token) — different FP accumulation orders produce
slightly different logits. This tradeoff is accepted because the ASM PA kernel
provides significant single-token decode throughput benefits.

### Comparison with TRITON_ATTN backend

| Phase | ROCM_AITER_FA | TRITON_ATTN |
|-------|---------------|-------------|
| Prefill/Extend | AITER Triton `unified_attention` | vLLM Triton `unified_attention` |
| Decode (single-token) | AITER ASM `pa_fwd_asm` | vLLM Triton `unified_attention` |
| Decode (multi-token) | AITER Triton `unified_attention` | vLLM Triton `unified_attention` |
| Fused RoPE+KV cache | Supported | Not supported |
| QK-Norm+RoPE+KV fusion | Supported | Not supported |
| Shuffle KV layout | Supported (single-token only) | Not supported |

## Block Size Support

`get_supported_kernel_block_sizes()` returns `MultipleOf(16)`. The framework's
virtual block splitting (introduced in
[vllm-project/vllm#24486](https://github.com/vllm-project/vllm/pull/24486))
ensures the kernel only sees 16 or 32-sized blocks regardless of the model's
logical page size.

The AITER PA kernel has an [internal assertion](https://github.com/ROCm/aiter/blob/3728dcedf8e961abd32934b386ab729a72837b7a/csrc/cpp_itfs/pa/pa_kernels.cuh#L259)
that `block_size <= 32`. This is satisfied because the *kernel* block size
(post-splitting) is always ≤ 32, even when models use larger page sizes (e.g.
128 for MiniMax-M3 sparse attention, 544 for Qwen3-Next).

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_ROCM_USE_AITER` | `0` | Master switch for all AITER ops |
| `VLLM_ROCM_USE_AITER_MHA` | `1` | Enable AITER MHA (gates ROCM_AITER_FA backend selection) |
| `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT` | `0` | Enable shuffle KV cache tiling for ASM PA |

## Hardware Requirements

- AMD CDNA3+ (`get_cdna_version() > 2`):
  - **CDNA3 / gfx942**: MI300X, MI300A, MI308X, MI325X
  - **CDNA4 / gfx950**: MI355X
- ROCm with AITER installed (`pip install aiter` or built from source)
