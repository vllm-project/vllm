# Octave KV: a 3-bit KV cache for ROCm

`ROCM_OCTAVE` is an attention backend for AMD GPUs that keeps the KV cache of
full-attention layers at 3-4 bits per value, with native kernels (HIP today). It is
selected with `--kv-cache-dtype octave_k3v3_compact`, `octave_k3v3` or
`octave_k3v4` (the backend is picked from the cache dtype;
`--attention-backend ROCM_OCTAVE` also works). The name comes from the 3-bit
codes: each one indexes an eight-level codebook.

## RDNA4 serving

`ROCM_OCTAVE` dispatches to independent RDNA4 kernels on `gfx1200`/`gfx1201`.
Select an `octave_*` cache dtype, or also specify
`--attention-backend ROCM_OCTAVE` explicitly.

```bash
HIP_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3.8-27B-FP8 \
    --tensor-parallel-size 2 --dtype bfloat16 \
    --attention-backend ROCM_OCTAVE --kv-cache-dtype octave_k3v4 \
    --max-model-len 32768 --max-num-batched-tokens 8192 --max-num-seqs 4
```

The RDNA4 path is validated with ROCm 10. CMake compiles the HIP kernels
for the selected `gfx1200` and/or `gfx1201` targets into vLLM's `_C` extension.
Python calls cache writes, prefill, decode, and QSA via `torch.ops._C`, as on
RDNA3. Native prefill takes the packed prefix and the current chunk's
unquantized K/V as inputs.

RDNA4 currently supports head size 256 with 64 rotary dimensions. This applies
to dense attention and the QSA path. Other head-size/RoPE combinations are
rejected during geometry validation. The specialized HIP kernels provide
cache storage and a small fixed set of scalar and WMMA kernels for decode,
verification, and prefill. This includes grouped-prefill fast paths for GQA
groups 1 and 2 and GQA-6 reuse paths for all three cache formats. Packed
prefill can also use the geometry HIP kernels, which handle sparse QSA
attention. Both use the existing Octave wire formats and causal
metadata. QSA supports tagged exact K/V for selected tokens among the first
four and last 64. A missing tag uses packed KV.

To limit the number of compiled variants, scalar kernels also handle unsafe
decode and matrix rows, and geometry prefill selects request tiling at runtime.
The grouped-prefill guard remains separate to preserve ragged-request
performance. These choices retain the tuned WMMA paths and numerical checks.

At startup, RDNA4 reserves shared GPU workspace using the layer's local query
and KV head counts, the scheduler's token and request limits, and the maximum
sequence length. The scratch budget covers the larger of FP16 expansion and
streamed prefill with its split buffers, subject to a 2 GiB ceiling and the
packed kernel's 256-split limit. The FP32 output is reserved alongside it.
Decode and prefill choose splits within that budget. Captured graphs reuse
the reserved addresses, and native kernels resolve live lengths on the device.

Streamed and FP16-expanded dense prefill support any positive integer ratio of
query heads to KV heads per GPU. Dispatch chooses between them using the
chunk size, prefix length, number of query/head tiles, and available workspace.
Small chunks, ragged batches, and long, thin prefixes can stream directly from
packed pages. When the scheduler supplies no sequence-length hint, dispatch uses
the page table capacity. For standard serving geometry and a positive finite
scale, the whole batch uses packed attention only when neither FP16 expansion
nor a single streamed split fits the scratch budget. Packed attention can run
with one split and no split scratch.

In streamed prefill, loader waves stage 16 cached keys at a time. Each
rotated query uses two signed base-256 digits. Integer WMMA accumulates their dot products with the
original codebook entries before applying the stored K scale. A partner wave
computes PV using FP16 WMMA and FP32 accumulation. The live chunk uses its raw
K/V and the finish kernel combines both softmax normalizers.

When the expanded buffers fit, larger batches can expand the prefix once and
reuse it across query positions. Each compute wave covers the full head
dimension. For short single requests, dispatch selects 12, 16 or 20 waves per
workgroup to avoid an extra pass over the prefix for a partial query tile.
Expansion preserves the FP16 key rounding residual and measures
its squared norm. A query workgroup includes the residual in QK when the
Cauchy–Schwarz bound on the absolute logit error exceeds 1/64. Dispatch
charges the residuals and their bounds to the same workspace budget. Both
routes retain the FP32 row repair for numerically unsafe outputs. Dense PV
scales probabilities by a power of two before FP16 conversion so small weights
on large values survive, then removes that factor from the normalized output.

Sliding windows, ALiBi, attention sinks, soft caps, unequal QK/V dimensions
and fused output quantization are unsupported. Runtime validation was on
`gfx1201`; `gfx1200` was build-tested.

## Why it exists

Long-context serving on AMD GPUs is bounded by KV-cache memory long before it
is bounded by compute. The existing options on ROCm were:

- fp16/bf16 KV (`TRITON_ATTN`, `ROCM_ATTN`): 16 bits per value.
- fp8 / int8 per-token-head: 8 bits per value.
- `TURBOQUANT`: 3-4 bits per value, but its kernels are Triton (store and
  decode) and its prefill goes through FlashAttention; it does not support the
  fused multi-step MTP draft.

Octave KV targets the same bit budget as TurboQuant with HIP kernels written for
the AMD matrix instructions (WMMA on RDNA3, MFMA on CDNA as a contract slot),
reading the packed cache directly in decode, and with
multi-token verification (MTP / speculative decoding) as a first-class case.

## Supported models on RDNA3/CDNA3

The format exploits *partial rotary embeddings*: models whose attention heads
apply RoPE to a fraction of the head dims. The RDNA3/CDNA3 kernels are built for
head size 256 with 64 rotary dims (`partial_rotary_factor = 0.25`), the
attention shape of the Qwen3-Next / Qwen3.5 / Qwen3.6 / Qwen3.8 family (dense
and MoE, e.g. Qwen3.8-27B, Qwen3.6-35B-A3B, Qwen3.5-122B-A10B). Like the MLA
backends, which serve only models with multi-head latent attention, Octave KV is
tied to an attention shape, not to a model.

Models with full RoPE (Llama, Qwen3, Mistral, ...) or other head sizes are
rejected by these tiers with an explicit error. RDNA4 currently has the same
256/64 Octave attention shape. Sliding window, ALiBi, attention
sinks, soft-capping and encoder attention are not supported.

The decode kernels cover GQA groups of 1, 2, 4, 6 or 8 query heads per KV
head on each GPU. Under tensor parallelism the group is counted per GPU, so
models with 32 query heads and 2 KV heads (Qwen3.5-122B-A10B) need TP ≥ 4.

## Format

One slot per (token, KV head). V is always rotated as one 256-dim block
(random signs, then a Walsh-Hadamard transform) and gets 3 or 4 bits per dim.
K has two layouts:

- **Block K** (`octave_k3v3`, `octave_k3v4`): four 64-dim blocks, each rotated
  on its own. The first block holds exactly the 64 rotary dims and gets 4 bits
  per dim, the three NoPE blocks 3 bits; each block has its own scale.
- **Compact K** (`octave_k3v3_compact`): rotated over all 256 dims like V,
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
| `octave_k3v3_compact` | 196 | 3.06 |
| `octave_k3v3` | 216 | 3.38 |
| `octave_k3v4` | 248 | 3.88 |

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
truth is the PyTorch reference in `vllm/v1/attention/ops/rocm_octave.py`.

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
| Cache write | `octave_cache_store`: one wave per slot, rotate + quantize + pack | `csrc/attention/octave_attn.cu` |
| Decode / MTP verify (≤128 query tokens per request) | `octave_decode`: split-KV; WMMA tier on gfx11, portable dot4 tier elsewhere; consecutive query tokens of a request share each K/V read | `octave_decode_wmma_rdna3.cu`, `octave_attn.cu` |
| Split reduce + inverse V rotation | `reduce_kernel` | `octave_attn.cu` |
| Prefill | `octave_prefill`: phase 1 reads the cached prefix from the packed cache (int8 WMMA for QK with per-segment scales, fp16 WMMA for PV), phase 2 attends to the chunk's own unquantized K/V and merges | `octave_prefill_rdna3.cu` |
| Q/K/V chunk rotation for prefill | `octave_rotate` | `octave_attn.cu` |
| Codebooks (`v_perm_b32` tables) and byte layout | — | `octave_format.cuh` |

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

`ROCM_OCTAVE` selects the native implementation for RDNA3, RDNA4 or CDNA3;
other families are rejected at selection. The RDNA3/CDNA3 tiers are validated
on gfx1100 and gfx942 (MI300). RDNA4 uses the separate HIP runtime described
above. CMake builds the RDNA3/CDNA3 sources and registers their ops only when
the target list includes
gfx1100 or gfx942; other targets in the same build compile the
architecture-specific kernels as empty stubs, and the kernel dispatch reports
any other RDNA3 chip as not validated.

| Piece | gfx1100 (RDNA3) | gfx942 (MI300) |
| --- | --- | --- |
| int8 dot4 (`octave_arch.cuh`) | `v_dot4_i32_iu8` | `v_dot4_i32_i8` |
| Codebook lookup (`octave_format.cuh`) | `v_perm_b32` | `v_perm_b32` |
| Cache store, reduce, rotate | portable | portable (decode reduce: gfx942 kernel) |
| Decode | WMMA | MFMA (`v_mfma_i32_16x16x32_i8`) |
| Prefill | WMMA | MFMA |

The portable dot4 decode stays as the reference tier the tests compare the
native tiers against.

To add an architecture:

1. Implement `octave_prefill` for it, in HIP or FlyDSL (same signature and semantics as
   `octave_prefill_rdna3.cu`: phase-1 partials over the packed prefix, phase-2
   chunk attention and merge, rotated space in and out).
2. Add a decode tier, or validate the portable dot4 kernel there.
3. Add the architecture to the CMake guard, to `_kernel_tier` and
   `supports_compute_capability` in
   `vllm/v1/attention/backends/rocm_octave_attn.py`, and to the test gate.
4. Run `tests/kernels/attention/test_rocm_octave.py`: every kernel is checked
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
| `octave_k4v4` | 1,402k |
| `octave_k3v3` | 1,679k |

### Speed

TBD: `vllm bench serve` (random prompts, no prefix reuse) TTFT and TPOT at
8k / 32k / 100k / 380k input, and per-layer kernel times from
`benchmarks/attention_benchmarks`.

### Quality

TBD: GSM8K (`tests/evals/gsm8k/gsm8k_eval.py`, 1319 questions, 5-shot,
temperature 0.6) and next-token NLL over a 380k-token document.

## Reproducing

```bash
vllm serve Qwen/Qwen3.8-27B --kv-cache-dtype octave_k3v4 ...
pytest tests/kernels/attention/test_rocm_octave.py
python benchmarks/attention_benchmarks/benchmark.py \
    --backends ROCM_OCTAVE --kv-cache-dtype octave_k3v4 \
    --model Qwen/Qwen3.8-27B --head-dim 256 --num-q-heads 6 --num-kv-heads 1 \
    --batch-specs q1s100k q4s100k q2ks100k
```
