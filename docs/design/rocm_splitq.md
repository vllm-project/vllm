# ROCm SplitQ KV cache

`ROCM_SPLITQ` is an attention backend for AMD GPUs that keeps the KV cache of
full-attention layers at 3-4 bits per value, with native kernels (HIP today). It is
selected with `--kv-cache-dtype splitq_k3v3_compact`, `splitq_k3v3` or
`splitq_k3v4` (the backend is picked from the cache dtype;
`--attention-backend ROCM_SPLITQ` also works).

## Why it exists

Long-context serving on AMD GPUs is bounded by KV-cache memory long before it
is bounded by compute. The existing options on ROCm were:

- fp16/bf16 KV (`TRITON_ATTN`, `ROCM_ATTN`): 16 bits per value.
- fp8 / int8 per-token-head: 8 bits per value.
- `TURBOQUANT`: 3-4 bits per value, but its kernels are Triton (store and
  decode) and its prefill goes through FlashAttention; it does not support the
  fused multi-step MTP draft.

SplitQ targets the same bit budget as TurboQuant with HIP kernels written for
the AMD matrix instructions (WMMA on RDNA3, MFMA on CDNA as a contract slot),
reading the packed cache directly in both decode and prefill, and with
multi-token verification (MTP / speculative decoding) as a first-class case.

## Supported models

The format exploits *partial rotary embeddings*: models whose attention heads
apply RoPE to a fraction of the head dims. Today the kernels are built for
head size 256 with 64 rotary dims (`partial_rotary_factor = 0.25`), the
attention shape of the Qwen3-Next / Qwen3.5 / Qwen3.6 / Qwen3.8 family (dense
and MoE, e.g. Qwen3.8-27B, Qwen3.6-35B-A3B, Qwen3.5-122B-A10B). Like the MLA
backends, which serve only models with multi-head latent attention, SplitQ is
tied to an attention shape, not to a model.

Models with full RoPE (Llama, Qwen3, Mistral, ...) or other head sizes are
rejected at startup with an explicit error. Sliding window, ALiBi, attention
sinks, soft-capping and encoder attention are not supported.

## Format

One slot per (token, KV head). V is always rotated as one 256-dim block
(random signs, then a Walsh-Hadamard transform) and gets 3 or 4 bits per dim.
K has two layouts:

- **Block K** (`splitq_k3v3`, `splitq_k3v4`): four 64-dim blocks, each rotated
  on its own. The first block holds exactly the 64 rotary dims and gets 4 bits
  per dim, the three NoPE blocks 3 bits; each block has its own scale.
- **Compact K** (`splitq_k3v3_compact`): rotated over all 256 dims like V,
  3 bits per dim, one scale. The rotation spreads the rotary dims over the
  whole head, so they need no extra bits; slots pad to 4 bytes instead of 8.

| Bytes (block K) | Content |
| --- | --- |
| `[0, 32)` | K RoPE block, 4-bit codes |
| `[32, 104)` | K NoPE blocks, 3-bit codes |
| next `256·v/8` | V, `v`-bit codes |
| next 10 | fp16 scales (K blocks 0-3, V) |

| Bytes (compact K) | Content |
| --- | --- |
| `[0, 96)` | K, 3-bit codes |
| `[96, 192)` | V, 3-bit codes |
| `[192, 196)` | fp16 scales (K, V) |

| Cache dtype | Slot bytes | Bits per value |
| --- | --- | --- |
| `splitq_k3v3_compact` | 196 | 3.06 |
| `splitq_k3v3` | 216 | 3.38 |
| `splitq_k3v4` | 248 | 3.88 |

For reference, `turboquant_3bit_nc` uses 198 bytes and `turboquant_k3v4_nc`
230.

After the rotation the coordinates are close to Gaussian, so each code
indexes the Lloyd-Max codebook of N(0, 1) for its width, stored as int8
(`LUT` in the reference): a code `c` of a block with scale `s` stands for
`LUT[c] · s / 64`. The scale is chosen so that `x · x̂ = |x|²`. A
least-squares fit minimizes the reconstruction error but shrinks every
reconstructed vector (by about 3% at 3 bits), which flattens the softmax and
scales down the attention output; at 380k tokens it cost 2-3× more NLL than
this scale, at the same size. The quantizer is dynamic: no calibration, no per-model
dictionaries. The kernels map four codes to their int8 values with one
`v_perm_b32` (two for 4-bit codes), so the codebook costs nothing in
bandwidth and the dot products stay integer. The byte layout's source of
truth is the PyTorch reference in `vllm/v1/attention/ops/rocm_splitq.py`.

Why these layouts: measured on real Q/K/V of Qwen3.8-27B (32k tokens, every
full-attention layer, attention-output error), both K layouts sit on the same
bytes-versus-error curve, about 25-30% below TurboQuant at equal size; block
K reaches the 4-bit rotary block, compact K the smallest slot. Rotating the
rotary block alone at 3 bits does not work (one layer reaches 32% error);
rotating it with the rest of the head does. Moving bits between K and V or
weighting K channels by query energy did not help at a fixed size. The int8
codebook loses nothing against the floating-point one.

Attention runs in the rotated space: the query is rotated like K and the
output is rotated back once per query. The rotations are orthogonal, so the
scores are those of the unrotated vectors.

## Kernels and data flow

| Step | Kernel | File |
| --- | --- | --- |
| Cache write | `splitq_cache_store`: one wave per slot, rotate + quantize + pack | `csrc/attention/splitq_attn.cu` |
| Decode / MTP verify (≤128 query tokens per request) | `splitq_decode`: split-KV; WMMA tier on gfx11, portable dot4 tier elsewhere; consecutive query tokens of a request share each K/V read | `splitq_decode_wmma_rdna3.cu`, `splitq_attn.cu` |
| Split reduce + inverse V rotation | `reduce_kernel` | `splitq_attn.cu` |
| Prefill | `splitq_prefill`: phase 1 reads the cached prefix from the packed cache (int8 WMMA for QK with per-segment scales, fp16 WMMA for PV), phase 2 attends to the chunk's own unquantized K/V and merges | `splitq_prefill_rdna3.cu` |
| Q/K/V chunk rotation for prefill | `splitq_rotate` | `splitq_attn.cu` |
| Codebooks (`v_perm_b32` tables) and byte layout | — | `splitq_format.cuh` |

The metadata builder builds, on the device and without host syncs, the
per-query request and causal-length maps the kernels index by. They are what
makes MTP verification batches (several query tokens per request, each with
its own causal length) work without special cases, and they give cudagraph
padding queries a K length of 0.

## Architecture contract

Kernels are native: HIP or FlyDSL, never Triton. There is no fallback: an
architecture without a required kernel is rejected at startup with an error
naming what is missing. The kernels in this tree are HIP; a FlyDSL kernel
(for example a CDNA prefill) must keep the same signature and semantics, pass
the same tests, and report a clear error when `flydsl` is not installed.

Every AMD target vLLM builds (gfx906, gfx908, gfx90a, gfx942, gfx950, gfx1030,
gfx11xx, gfx12xx, gfx1250) compiles every SplitQ source; architecture-specific
code is guarded and the portable tier uses only 32-lane shuffles, so it is
correct on wave64 (CDNA) as well as wave32 (RDNA).

| Piece | gfx11 (RDNA3) | gfx12 (RDNA4) | gfx90a / gfx942 / gfx950 (MI200/MI300/MI350) | gfx1030 (RDNA2) |
| --- | --- | --- | --- | --- |
| int8 dot4 (`splitq_arch.cuh`) | `v_dot4_i32_iu8` | `v_dot4_i32_iu8` | `v_dot4_i32_i8` | `v_dot4_i32_i8` |
| Codebook lookup (`splitq_format.cuh`) | `v_perm_b32` | `v_perm_b32` | `v_perm_b32` | `v_perm_b32` |
| Cache store, reduce, rotate | portable | portable | portable | portable |
| Decode | WMMA tier | portable dot4 tier | portable dot4 tier | portable dot4 tier |
| Prefill | WMMA | **missing** | **missing** | **missing** |

To add an architecture:

1. Implement `splitq_prefill` for it, in HIP or FlyDSL (same signature and semantics as
   `splitq_prefill_rdna3.cu`: phase-1 partials over the packed prefix, phase-2
   chunk attention and merge, rotated space in and out). On CDNA the natural
   building blocks are `v_mfma_i32_16x16x32_i8` (gfx942/gfx950) or
   `v_mfma_f32_16x16x16f16` (gfx90a) in place of the WMMA calls.
2. Optionally add a decode tier (the portable dot4 kernel already works).
3. Register the tiers in `_ARCH_TIERS` in
   `vllm/v1/attention/backends/rocm_splitq_attn.py`.
4. Run `tests/kernels/attention/test_rocm_splitq.py`: every kernel is checked
   against the PyTorch reference, and `test_decode_matches_reference` runs
   both the fast and the portable decode tier.

## Results

All measurements on 4× RX 7900 XTX (gfx1100), Qwen3.8-27B W4A16, tensor
parallel 4, MTP k=3, 8 GB of KV cache per GPU.

### Capacity

| KV cache | Tokens that fit |
| --- | --- |
| fp16 | TBD |
| `turboquant_k3v4_nc` | 1,790,625 |
| `splitq_k4v4` | 1,402k |
| `splitq_k3v3` | 1,679k |

### Speed

TBD: `vllm bench serve` (random prompts, no prefix reuse) TTFT and TPOT at
8k / 32k / 100k / 380k input, and per-layer kernel times from
`benchmarks/attention_benchmarks`.

### Quality

TBD: GSM8K (`tests/evals/gsm8k/gsm8k_eval.py`, 1319 questions, 5-shot,
temperature 0.6) and next-token NLL over a 380k-token document.

## Reproducing

```bash
vllm serve Qwen/Qwen3.8-27B --kv-cache-dtype splitq_k3v4 ...
pytest tests/kernels/attention/test_rocm_splitq.py
python benchmarks/attention_benchmarks/benchmark.py \
    --backends ROCM_SPLITQ --kv-cache-dtype splitq_k3v4 \
    --model Qwen/Qwen3.8-27B --head-dim 256 --num-q-heads 6 --num-kv-heads 1 \
    --batch-specs q1s100k q4s100k q2ks100k
```
