# Upstream PR #221: DeepSeek V4.1 kernels and the fused norm + RoPE + attn + RoPE + cast megakernel

| | |
| :-- | :-- |
| PR | [deepseek-ai/FlashMLA#221](https://github.com/deepseek-ai/FlashMLA/pull/221) "Add kernels for DeepSeek v4.1" |
| Author | Shengyu Liu (interestingLSY) |
| Merged | 2026-09-10 into `main` (`07a1089`) |
| Size | 162 files, +7058 / -1859 |
| Local ref | `pr-221` (fetched from `upstream pull/221/head`) |

This document describes what the PR changes, what the new fused "megakernel" is, how it works internally, what its interface looks like, and how to use it end to end. Unless stated otherwise, every file path refers to the tree **after** the PR.

---

## 1. TL;DR

- Adds attention kernels for **DeepSeek V4.1** (`d_qk = d_v = 512`, MQA, token-level sparse attention) on SM100 / SM103.
- Introduces two new paged KV cache formats: **V4.1 fp8** (528 B/token, RoPE quantized too, per-32 ue8m0 scales) and **V4.1 fp4** (288 B/token, e2m1 with per-16 e4m3 scales, only usable as the secondary `extra_k_cache`).
- Adds a new fused kernel, `flash_mla.fused_norm_rope_attn_rope_cast`, that replaces five launches (Q RMSNorm, Q RoPE, sparse core attention, conjugate O RoPE, FP8 cast of O) with one persistent kernel. It has a **prefill** and a **decode** entry point. Its output feeds DeepGEMM's FP8 einsum for the `Wv` projection directly.
- The fusion is free in FLOPs terms (1430 TFlops prefill, 670 TFlops decode on B200 per upstream) but requires the `Q_b` and `Wv` weights to be **permuted once, offline**, with two helper kernels shipped in the PR.
- Restructures `csrc/` into `csrc/kernels/{sm90,sm100,smxx}` plus a `csrc/api/*.cpp` layer, renames `MODEL1` to `V4`, and builds `sm_100a` + `sm_103a` instead of `sm_100f`.

---

## 2. What changed

### 2.1 Source tree restructure

Everything under `csrc/sm90`, `csrc/sm100`, `csrc/smxx` moved to `csrc/kernels/<arch>/...`. The API layer became one `.cpp` per operator with a `register_*(pybind11::module_&)` function, all collected in `csrc/api/api.cpp`:

```text
csrc/api/
  api.cpp                              # PYBIND11_MODULE, calls the register_* functions
  sparse_prefill.cpp                   # flash_mla_sparse_fwd
  sparse_decode.cpp                    # flash_mla_with_kvcache (sparse path)
  dense_fwd.cpp / dense_bwd.cpp        # SM100 MHA prefill fwd / bwd
  dense_decode.cpp                     # SM90 dense MLA decode
  fused_norm_rope_attn_rope_cast_fwd.cpp   # NEW: the megakernel + permute helpers
  common.h                             # dispatch macros, Arch, KV format detection
csrc/kernels/
  params.h                             # ModelType {V32, V4, V41, V41_FP4}, param structs
  kv_cache_format.h                    # NEW: compile-time KVCacheFormat<ModelType>
  defines.h, utils.h
  sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/
    core_attn/{config.h,kernel.h,kernel.cuh,instantiations/*.cu}   # the megakernel
    permute_q_b_proj/kernel.{h,cu}                                  # weight permute helper
    permute_wv_proj/kernel.{h,cu}                                   # weight permute helper
  sm100/decode/sparse/head64/          # renamed from sm100/decode/head64, + V41 / V41_FP4
  sm100/prefill/sparse/fwd/head64/     # NEW h_q=64 sparse prefill (d_qk 512 / 576)
  sm100/prefill/sparse/fwd_for_small_topk/head128/   # + V41 / V41_FP4 decode instantiations
  sm90/decode/sparse/                  # renamed from sm90/decode/sparse_fp8, model1 -> v4
```

`ModelType::MODEL1` is now `ModelType::V4` everywhere. The test-side enum `FP8KVCacheLayout` became `quant.KVCacheLayout` with members `V32_FP8`, `V32_FP8Sparse`, `V4_FP8Sparse`, `V41_FP8Sparse`, `V41_FP4`.

### 2.2 New KV cache formats

All `d_qk = 512` formats store a page block as `page_block_size` **data rows** followed by `page_block_size` **scale rows** (scales are not interleaved into the token row, unlike V3.2). The format is auto-detected from `k_cache.shape[-1]` (bytes per token).

| Format | `ModelType` | Bytes/token | Data row | Scale row | Scale granule |
| :-- | :-- | --: | :-- | :-- | --: |
| DeepSeek V3.2 | `V32` | 656 | 512 B e4m3 NoPE, 16 B fp32 scales, 128 B bf16 RoPE (all inline) | none | 128 |
| DeepSeek V4 | `V4` | 584 | 448 B e4m3 NoPE + 128 B bf16 RoPE | 8 B: 7 ue8m0 + 1 pad | 64 |
| DeepSeek V4.1 fp8 | `V41` | 528 | 512 B e4m3 (RoPE quantized too) | 16 B ue8m0 | 32 |
| DeepSeek V4.1 fp4 | `V41_FP4` | 288 | 256 B e2m1, 2 per byte, even index in low nibble | 32 B e4m3 | 16 |

Rules enforced by the API (`csrc/api/common.h`, `kv_cache_format.h`):

- `V41_FP4` is only valid for `extra_k_cache`, and only when `k_cache` is `V41`. Otherwise `extra_k_cache` must have the same format as `k_cache`.
- Upstream's intended use in V4.1: the sliding-window (SWA) cache in fp8, the compressed-attention (CA) cache in fp4.
- fp4 scale is `amax / 6` rounded to e4m3 with no per-tensor scale. The e2m1 x e4m3 product is exact in bf16, and the kernels dequantize to bf16 in shared memory, so all MMAs stay bf16.
- Reference quantizer / dequantizer: `tests/quant.py` (`quantize_k_cache`, `dequantize_k_cache`).

### 2.3 Standard (non-fused) kernel coverage after the PR

| Kernel | Arch | Heads | `d_qk` | KV formats |
| :-- | :-- | :-- | :-- | :-- |
| Sparse decode | SM90 | 64, 128 | 512, 576 | V32, V4 |
| Sparse decode | SM100 head64 | 64 | 512, 576 | V32, V4, V41, V41+V41_FP4 |
| Sparse decode | SM100 head64x2 (2 launches) | 128 | 512, 576 | V32, V4 |
| Sparse decode | SM100 head128 (native, new) | 128 | 512 | V4, V41, V41+V41_FP4, with and without split-KV |
| Sparse prefill | SM90 | 64, 128 | 512, 576 | bf16 |
| Sparse prefill | SM100 head64 (new) | 64 | 512, 576 | bf16 |
| Sparse prefill | SM100 head128 | 128 | 512, 576 | bf16 |
| Fused megakernel | SM100 / SM103 | 64, 128 | 512 | prefill bf16; decode V4, V41, V41+V41_FP4 |

Sparse decoding for V4.1 formats is SM100-only. The public `flash_mla_with_kvcache` / `flash_mla_sparse_fwd` signatures are unchanged.

### 2.4 Build and tests

- `setup.py` now emits `-gencode arch=compute_100a,code=sm_100a` and `-gencode arch=compute_103a,code=sm_103a` instead of `sm_100f` (better SASS). It also adds the CUDA 13 `cccl` include directories.
- The kerutils submodule copy is synced with upstream.
- New test: `tests/test_fused_norm_rope_attn_rope_cast.py`. It needs `tilelang`, `tile_kernels` (TileKernels), `deep_gemm`, and `kernelkit`.
- `tests/test_flash_mla_sparse_decoding.py` and `tests/lib.py` gained `kvcache_layout` / `extra_kvcache_layout` parameters.

---

## 3. What the megakernel is

Upstream calls it the "fused norm + RoPE + attn + RoPE + cast kernel". Python module: `flash_mla.fused_norm_rope_attn_rope_cast`. C++ namespace: `sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn`.

### 3.1 The problem it solves

In the DeepSeek V4 / V4.1 attention block the segment between the two big GEMMs looks like this:

```text
hidden --Q_a--> q_lora [q_lora_rank]
       --Q_b--> q      [h_q, 512]                      (FP8 GEMM, DeepGEMM)
                 |  (1) Q RMSNorm            (V4 only, weightless)
                 |  (2) Q RoPE               (last 64 dims, non-neox)
                 |  (3) sparse core attention over top-k KV tokens   (h_kv = 1, d = 512)
                 |  (4) O RoPE, conjugate    (last 64 dims)
                 |  (5) cast O to FP8 + per-32 ue8m0 scales
                 v
       o_fp8 [n_wv_group, 8 * 512]
       --Wv--> [n_wv_group, o_lora_rank]               (FP8 grouped einsum, DeepGEMM)
       --Wo--> hidden
```

Steps 1, 2, 4, 5 are tiny memory-bound kernels. Each one costs a launch plus a full read and write of the `[s_q, h_q, 512]` activation. The megakernel does all five inside the attention kernel's existing load and epilogue phases, so the activation is read once (Q, in bf16) and written once (O, already in FP8).

### 3.2 The trade

The kernel is only free if its input and output can be streamed in the layout the tensor cores want, not the model's `[h, d]` layout. Rather than transposing at runtime, the PR **permutes the weights once** so the GEMMs produce and consume those layouts natively:

- `permute_q_b_proj` reorders the rows of `Q_b` so the Q GEMM writes each token as 16-element head-dim chunks interleaved across heads.
- `permute_wv_proj` reorders the input columns of `Wv` so it consumes the kernel's transposed 32-element output chunks.

Both permutations also reorder the FP8 scale factors. See section 5 for the exact layouts.

Upstream reports up to 1430 TFlops prefill and 670 TFlops decode on B200 for the fused kernel, against 1450 and 700 for the unfused sparse kernels measured on their own. Upstream describes the fused kernel as keeping "the same or even slightly higher TFlops" while removing the four surrounding small kernels, so the end-to-end win is the launches and memory passes saved, not raw attention throughput.

### 3.3 What is and is not supported

| Constraint | Value |
| :-- | :-- |
| GPU | SM100 / SM103 only (`Arch::is_sm100f()`) |
| `h_q` | 64 or 128 |
| `h_kv` | 1 (MQA) |
| `d_qk`, `d_v` | 512, 512 |
| RoPE | non-neox only, `rope_dim = 64`, applied to the last 64 dims |
| Q norm | weightless RMSNorm over all 512 dims, `rsqrt(sum(q^2)/512 + eps)`; any learned gamma must be folded into `Q_b` |
| O quant | `num_per_channels = 32`, `use_tma_aligned_col_major_sf = round_sf = use_packed_ue8m0 = True` (DeepGEMM's layout) |
| `wv_group_size` | fixed at 8, so `n_wv_group = h_q / 8` |
| Decode batch | 1. Flatten `b * s_q` into `s_q` since indices are page-absolute and `h_kv = 1` |
| Decode split-KV | not supported (the RoPE + quant epilogue cannot be combined across splits) |
| Decode KV format | `k_cache` in V4 or V41; `extra_k_cache` same as `k_cache`, or V41_FP4 when `k_cache` is V41 |
| `topk` | `indices.stride(0) * 4` must be a multiple of 32 B, so `topk % 8 == 0` for a contiguous `indices`. With `extra_k_cache`: `topk % 4 == 0` (fp8 extra) or `topk % 8 == 0` (fp4 extra) |
| `q` layout | permuted (section 5.1). Each token's `h_q * 512` values must be contiguous (`q.stride(1) == 512`) |

---

## 4. How the megakernel works

Source: `csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/kernel.cuh` (about 1600 lines). The header comment in `config.h` is the authoritative summary. Key points:

**Scheduling.** Persistent kernel driven by Cluster Launch Control (CLC). Grid is `(s_q * CLUSTER_SIZE, 1, 1)`. Each cluster processes one query token per job and pulls the next job from CLC.

**Cluster shape.** `CLUSTER_SIZE = ceil(h_q / 64)`. `h_q = 64` runs as one CTA with `B_TOPK = 64` tokens per KV block. `h_q = 128` runs as a 2-CTA cluster with dual-CTA UMMA and `B_TOPK = 96`; CTA0 owns `V[:, 0:256]`, CTA1 owns `V[:, 256:512]`. Each CTA has 512 threads in four warpgroups.

**Warp roles.**

| Warpgroup | Role |
| :-- | :-- |
| WG0 | Q fetch and O epilogue. Loads Q with 256-bit LDG, accumulates `sum(q^2)` per head on the fly, applies RoPE to the last 64 dims, stores Q to TMEM. Later loads O from TMEM, applies conjugate RoPE, quantizes to FP8, writes to global. Timeline `Q0 Q1 O0 Q2 O1 ... Qn O(n-1) On`, so Q of job i+1 overlaps O of job i. |
| WG1 | KV producer. Prefill: gathers the bf16 KV block with TMA `gather4`. Decode: loads fp8 / fp4 rows from the paged cache, dequantizes them in registers, writes bf16 into the shared-memory KV slot. |
| WG2 | Warp 8 issues UMMAs (CTA0 only). Warp 9 queries CLC. Warp 10 builds the validity mask from `indices` / `topk_length` and, in decode, the TMA coordinates and scale pointers for WG1. Warp 11 (decode, V4 only) TMA-loads the bf16 RoPE part. |
| WG3 | Softmax. Reads P from TMEM, maintains online-softmax state (`mi`, `li`), produces bf16 S into shared memory, rescales O in TMEM when the running max moves by more than 6 (in log2 units). |

**Multi-rail GEMM.** Q is folded by `FOLD_FACTOR = 2` into a 128-row operand and QK^T runs as a 2-rail batched GEMM; the two partial P tiles are reduced in shared memory. Three KV slots are software-pipelined.

**Q norm folding.** The RMSNorm denominator is a scalar per (token, head), so it is folded into the softmax scale: `score_multiplier = sm_scale / ln2 * rsqrt(sum(q^2)/512 + eps)`. The result is exact and needs no extra pass over Q.

**Epilogue quantization.** For each 32-element chunk of a head's output row: `amax` is clamped to at least `1e-4`, `sf = amax / 448` is rounded **up** to a power of two (ue8m0), the values are multiplied by `1/sf` and converted to e4m3 with round-to-nearest saturate-finite. The attention sink, when present, enters here as `output_scale = 1 / (li + exp2(sink * log2e - mi))`. `lse` is `mi * ln2 + log(li)` and becomes `+inf` when no valid token was attended. Prefill also writes `max_logits = real_mi * ln2`.

**Decode unified slot space.** With `extra_k_cache`, the kernel walks a virtual index array `[topk orig slots; extra_topk extra slots]` tiled by `B_TOPK`. Since 128 is not a multiple of 96, one block may straddle the boundary (`KVLocation::ORIG_AND_EXTRA`). That is why `topk` must be 4-aligned (8 with fp4 extra): one TMA `gather4` covers 4 rows from one tensor map. This also keeps decode numerically aligned with a prefill over the concatenated indices.

---

## 5. Data layouts

### 5.1 Permuted Q (kernel input)

Nominal tensor shape is `[s_q, h_q, 512]`, bf16, but the kernel only uses `q.stride(0)`. Within a token the `h_q * 512` elements are ordered as

```text
[ chunk 0: h0 d0..15 | h1 d0..15 | ... | h(H-1) d0..15 ]
[ chunk 1: h0 d16..31 | h1 d16..31 | ... ]
...
[ chunk 31: h0 d496..511 | ... | h(H-1) d496..511 ]
```

i.e. index `= (d / 16) * (h_q * 16) + h * 16 + (d % 16)`. `permute_q_b_proj` produces this by moving `Q_b` row `(h, d)` to row `(d % 16) + h * 16 + (d / 16) * (h_q * 16)` and moving the corresponding scale-factor row identically. Scale granularity 32 or 128 is accepted.

### 5.2 Permuted O (kernel output)

`out_fp8` has shape `[s_q, n_wv_group, 8 * 512]`, fp8_e4m3. Within one wv group of 8 heads the `8 * 512` elements are ordered as

```text
[ chunk 0: h0 d0..31 | h1 d0..31 | ... | h7 d0..31 ]
[ chunk 1: h0 d32..63 | ... | h7 d32..63 ]
...
[ chunk 15: h0 d480..511 | ... | h7 d480..511 ]
```

i.e. 32-element chunk `(h, c)` sits at chunk position `c * 8 + h`. `permute_wv_proj` applies the same permutation to the input columns of each wv group's `Wv` slice, so `deep_gemm.fp8_einsum("bhr,hdr->bhd", ...)` contracts the permuted `r` dimension directly.

`out_sf` has shape `[s_q, n_wv_group, 8 * 512 / 32 / 4] = [s_q, n_wv_group, 32]`, int32, four ue8m0 bytes packed per int32 in the same chunk order. It is stored token-column-major (`out_sf.stride(0) == 1`) with `s_q` padded up to a multiple of 4, which is exactly DeepGEMM's `use_tma_aligned_col_major_sf=True, round_sf=True, use_packed_ue8m0=True` scale layout. Scale granularity is always 32 regardless of `num_per_channels`, because the weight is per-32 scaled and DeepGEMM requires A and B to match.

### 5.3 Permuted weights

| Helper | Input | Output |
| :-- | :-- | :-- |
| `permute_q_b_proj((w, sf), h_q, d_q)` | `w: [h_q*d_q, q_lora_rank]` fp8; `sf: [h_q*d_q, q_lora_rank/gran/4]` int32, `sf.stride(0) == 1` | same shapes, rows permuted |
| `permute_wv_proj((w, sf), wv_group_size, d_o)` | `w: [n_wv_group, d_proj_out, wv_group_size*d_o]` fp8; `sf: [n_wv_group, d_proj_out, wv_group_size*d_o/32/4]` int32, `sf.stride(1) == 1` | same shapes, columns permuted; `gran` must be 32 |

---

## 6. Interface reference

### 6.1 Python: `flash_mla.fused_norm_rope_attn_rope_cast.prefill`

```python
out_fp8, out_sf, max_logits, lse = prefill(
    enable_q_norm: bool, rms_norm_eps: float,
    token_positions: Tensor, is_rope_neox_style: bool, rope_dim: int, cos_sin_cache: Tensor,
    n_wv_group: int, num_per_channels: int,
    use_tma_aligned_col_major_sf: bool, round_sf: bool, use_packed_ue8m0: bool,
    q: Tensor, kv: Tensor, indices: Tensor, sm_scale: float,
    d_v: int = 512, attn_sink: Tensor | None = None, topk_length: Tensor | None = None,
)
```

| Group | Arg | Type / shape | Notes |
| :-- | :-- | :-- | :-- |
| Norm | `enable_q_norm` | bool | `True` for V4, `False` for V4.1 |
| | `rms_norm_eps` | float | e.g. `1e-4` |
| RoPE | `token_positions` | `[s_q]` int32 | position of each query token |
| | `is_rope_neox_style` | bool | must be `False` |
| | `rope_dim` | int | must be 64 |
| | `cos_sin_cache` | `[max_pos, 64]` fp32 | `[:, :32]` cos, `[:, 32:]` sin; vLLM `RotaryEmbedding.cos_sin_cache` layout |
| Cast | `n_wv_group` | int | `h_q / 8` |
| | `num_per_channels` | int | must be 32 |
| | three DeepGEMM flags | bool | must all be `True` |
| Attn | `q` | `[s_q, h_q, 512]` bf16 | **permuted layout**, pre-RoPE, pre-norm |
| | `kv` | `[s_kv, 1, 512]` bf16 | non-paged |
| | `indices` | `[s_q, 1, topk]` int32 | invalid entries `< 0` or `>= s_kv` |
| | `sm_scale` | float | |
| | `d_v` | int | must be 512 |
| | `attn_sink` | `[h_q]` fp32, optional | output scaled by `exp(lse) / (exp(lse) + exp(sink))`; no effect on `lse`, `max_logits` |
| | `topk_length` | `[s_q]` int32, optional | query `i` uses only `indices[i, :, :topk_length[i]]` |

Returns `out_fp8 [s_q, n_wv_group, 4096]` e4m3, `out_sf [s_q, n_wv_group, 32]` int32, `max_logits [s_q, h_q]` fp32, `lse [s_q, h_q]` fp32. A query with no valid token gets `max_logits = -inf`, `lse = +inf`, zero output.

### 6.2 Python: `flash_mla.fused_norm_rope_attn_rope_cast.decode`

```python
out_fp8, out_sf, lse = decode(
    enable_q_norm, rms_norm_eps,
    token_positions, is_rope_neox_style, rope_dim, cos_sin_cache,
    n_wv_group, num_per_channels, use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0,
    q: Tensor, k_cache: Tensor, indices_in_kvcache: Tensor, sm_scale: float,
    d_v: int = 512, attn_sink=None, topk_length=None,
    extra_k_cache=None, extra_indices_in_kvcache=None, extra_topk_length=None,
)
```

Differences from `prefill`:

| Arg | Type / shape | Notes |
| :-- | :-- | :-- |
| `q` | `[s_q, h_q, 512]` bf16 | `s_q` is the flattened `b * s_q`; batch is always 1 |
| `k_cache` | `[num_blocks, page_block_size, 1, bytes_per_token]` fp8_e4m3fn / int8 / uint8 | format detected from `bytes_per_token`: 584 (V4) or 528 (V4.1) |
| `indices_in_kvcache` | `[s_q, topk]` int32 | `block_idx * page_block_size + offset`; `-1` for invalid |
| `topk_length` | `[s_q]` int32, optional | per query token |
| `extra_k_cache` | `[extra_num_blocks, extra_page_block_size, 1, bytes]`, optional | same format as `k_cache`, or 288 (V4.1 fp4) when `k_cache` is V4.1 |
| `extra_indices_in_kvcache` | `[s_q, extra_topk]` int32 | required iff `extra_k_cache` is given |
| `extra_topk_length` | `[s_q]` int32, optional | |

Returns `out_fp8`, `out_sf` as above, and `lse [s_q, h_q]`. No `max_logits`.

Each page block must be contiguous (`k_cache.stride(1) == bytes_per_token`), the base pointer 16 B aligned, and `k_cache.stride(0)` a multiple of the format's TMA stride (576 for V4, 512 for V4.1, 256 for fp4). `tests/quant.py` pads blocks accordingly.

### 6.3 Python: weight permutation helpers

```python
q_b_fp8_perm, q_b_sf_perm = fused_norm_rope_attn_rope_cast.permute_q_b_proj((q_b_fp8, q_b_sf), h_q, d_q)
wv_fp8_perm,  wv_sf_perm  = fused_norm_rope_attn_rope_cast.permute_wv_proj((wv_fp8, wv_sf), wv_group_size, d_o)
```

Shapes and constraints are in section 5.3. These are one-time preprocessing kernels, not per-step.

### 6.4 pybind level (`flash_mla.cuda`)

| Function | Registered in |
| :-- | :-- |
| `fused_norm_rope_attn_rope_cast_fwd(q, kv, indices, sm_scale, d_v, attn_sink, topk_length, enable_q_norm, rms_norm_eps, token_positions, is_rope_neox_style, rope_dim, cos_sin_cache, n_wv_group, num_per_channels, use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0)` | `csrc/api/fused_norm_rope_attn_rope_cast_fwd.cpp` |
| `fused_norm_rope_attn_rope_cast_decode(q, kv, indices, sm_scale, d_v, attn_sink, topk_length, extra_kv, extra_indices, extra_topk_length, enable_q_norm, ..., use_packed_ue8m0)` | same |
| `permute_q_b_proj(q_b_proj, scale_factors, h_q, d_q)` | same |
| `permute_wv_proj(wv_proj, scale_factors, wv_group_size, d_o)` | same |

### 6.5 C++ kernel entry

```cpp
namespace sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn {
struct Config { SparseAttnFwdMode FWD_MODE; ModelType MODEL_TYPE; ModelType EXTRA_MODEL_TYPE; uint32_t H_Q; bool ENABLE_Q_NORM; };
template<SparseAttnFwdMode M> using ParamT = /* ParamsTemplate<SparseAttnFwdParams> or ParamsTemplate<SparseAttnDecodeParams> */;
template<Config CONFIG> void run_fused_norm_rope_attn_rope_cast_fwd_kernel(const ParamT<CONFIG.FWD_MODE>& params);
}
```

`ParamsTemplate<Base>` extends the existing sparse prefill / decode param struct with the norm, RoPE, cast, and FP8-output fields. 16 instantiations are compiled: `{V4 prefill, V4 decode, V41 decode, V41+fp4 decode} x {h64, h128} x {norm, nonorm}`.

---

## 7. How to use it

### 7.1 Build

```bash
git submodule update --init --recursive
pip install -v .          # needs CUDA 12.9+ for SM100; emits sm_100a and sm_103a
```

### 7.2 One-time weight preparation

```python
import torch, tile_kernels, deep_gemm
from flash_mla import fused_norm_rope_attn_rope_cast as fused

h_q, d_q, q_lora_rank = 64, 512, 1536
n_wv_group, wv_group_size, d_o, o_lora_rank = h_q // 8, 8, 512, 512

# Q_b: [h_q*d_q, q_lora_rank] bf16 -> FP8 in DeepGEMM layout -> permuted
q_b_fp8, q_b_sf = tile_kernels.quant.per_token_cast(
    q_b_proj_bf16, 'e4m3', 128,                        # gran 32 or 128 both OK
    use_tma_aligned_col_major_sf=True, round_sf=True, use_packed_ue8m0=True)
q_b_fp8, q_b_sf = fused.permute_q_b_proj((q_b_fp8, q_b_sf), h_q, d_q)

# Wv: [n_wv_group*o_lora_rank, wv_group_size*d_o] bf16 -> FP8 (gran 32) -> einsum SF layout -> permuted
wv_fp8, wv_sf = tile_kernels.quant.per_token_cast(
    wv_proj_bf16, 'e4m3', 32,
    use_tma_aligned_col_major_sf=False, round_sf=True, use_packed_ue8m0=False)
wv_sf = deep_gemm.transform_sf_into_required_layout(
    wv_sf.view(n_wv_group, o_lora_rank, wv_group_size * d_o // 32),
    o_lora_rank, wv_group_size * d_o, num_groups=n_wv_group, recipe=(1, 1, 32), is_sfa=False)
wv_fp8 = wv_fp8.view(n_wv_group, o_lora_rank, wv_group_size * d_o)
wv_fp8, wv_sf = fused.permute_wv_proj((wv_fp8, wv_sf), wv_group_size, d_o)
```

### 7.3 Per-layer forward, prefill

```python
# q_lora = (q_lora_fp8, q_lora_sf): FP8 [s_q, q_lora_rank] with per-token SF from tile_kernels.quant.per_token_cast.
# The permuted Q_b makes q land directly in the kernel's layout.
q = torch.empty((s_q, h_q * d_q), dtype=torch.bfloat16, device='cuda')
deep_gemm.fp8_gemm_nt(q_lora, (q_b_fp8, q_b_sf), q, recipe_a=(1, 128), recipe_b=(1, 128))
q = q.view(s_q, h_q, d_q)                              # nominal shape only; data is permuted

out_fp8, out_sf, max_logits, lse = fused.prefill(
    False, 1e-4,                                       # V4.1: no Q norm
    token_positions, False, 64, cos_sin_cache,
    n_wv_group, 32, True, True, True,
    q, kv, indices, sm_scale=sm_scale,
    attn_sink=attn_sink, topk_length=topk_length)

wv_out = torch.empty((s_q, n_wv_group, o_lora_rank), dtype=torch.bfloat16, device='cuda')
deep_gemm.fp8_einsum("bhr,hdr->bhd", (out_fp8, out_sf), (wv_fp8, wv_sf), wv_out, recipe=(1, 1, 32))
# wv_out.view(s_q, n_wv_group * o_lora_rank) -> Wo projection
```

### 7.4 Per-layer forward, decode

```python
# k_cache: V4.1 fp8 paged cache (528 B/token), e.g. the sliding-window cache
# extra_k_cache: V4.1 fp4 paged cache (288 B/token), e.g. the compressed-attention cache
out_fp8, out_sf, lse = fused.decode(
    False, 1e-4,
    token_positions, False, 64, cos_sin_cache,
    n_wv_group, 32, True, True, True,
    q.view(b * s_q, h_q, d_q), k_cache, indices_in_kvcache.view(b * s_q, topk), sm_scale,
    attn_sink=attn_sink, topk_length=topk_length,
    extra_k_cache=extra_k_cache,
    extra_indices_in_kvcache=extra_indices_in_kvcache.view(b * s_q, extra_topk),
    extra_topk_length=extra_topk_length)
# then the same fp8_einsum as in prefill
```

Unlike `flash_mla_with_kvcache`, the fused decode path has no scheduler metadata and no split-KV, so nothing needs to be cached between decode steps.

### 7.5 Test and benchmark

```bash
python tests/test_fused_norm_rope_attn_rope_cast.py            # correctness + perf, prefill and decode
python tests/test_flash_mla_sparse_decoding.py                 # standard decode incl. V41 / V41_FP4 layouts
python tests/test_flash_mla_sparse_prefill.py
```

The fused test builds a random `Q_b` and `Wv`, runs the fused kernel on the permuted weights, runs a TileLang-backed reference on the original weights, and compares the dequantized output, the `Wv` result, `max_logits`, and `lse`. It covers `h_q in {64, 128}`, `s_q in {1, 184, 2123}`, V4 / V4.1 / V4.1+fp4 cache pairs, attention sink, `topk_length`, all-invalid indices, and out-of-bounds top-k.

---

## 8. Notes for this fork (`vllm-project/FlashMLA`)

- This fork branched before the PR and carries its own changes: libtorch ABI-stable API (`csrc/api/*.h` with `STD_TORCH_CHECK`), optional output buffers, `num_sm_parts` clamping, and an **NVFP4 KV cache format** (#18). Merging PR #221 will conflict heavily because of the `csrc/` -> `csrc/kernels/` move and the `.h` -> `.cpp` API split.
- The fork's NVFP4 format is **not** upstream's `V41_FP4`. Fork: 352 B/token, V3.2 geometry (`d_qk = 576`), e2m1 NoPE + unscaled e4m3 RoPE + 32 permuted e4m3 scales inline, `ModelType::V32_NVFP4_FP8ROPE`. Upstream: 288 B/token, `d_qk = 512`, all 512 dims e2m1, scales in a separate row region, only allowed as `extra_k_cache`. They do not collide on bytes-per-token detection because `d_qk` differs, but any merge must keep both `ModelType` entries and both dequant paths.
- The PR syncs the vendored kerutils with upstream. Most primitives the megakernel uses (CLC helpers, `tma_gather4_cta_group_2`, 256-bit LDG/STG wrappers) already exist in this fork's kerutils. `tmem_ld_red_32dp32bNx` (TMEM load with fused reduction, used on SM103 in the epilogue) does not, so the kerutils diff must come along with the kernel.
- The megakernel needs the model side to hand over pre-permuted `Q_b` and `Wv` FP8 weights and to run the `Wv` projection with DeepGEMM's `fp8_einsum`. That is an integration change in the serving engine, not just a kernel swap.
