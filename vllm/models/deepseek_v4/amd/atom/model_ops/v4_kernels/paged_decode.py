# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sparse decode attention over a unified KV pool with per-token paged indices.

Designed for V4 decode + CUDAGraph: replaces the per-fwd `kv_flat_sa`
materialization (whose shape depends on `n_committed_per_seq` → varies per
fwd → blocks CG capture) with a single unified KV pool indexed via paged
indices, mirroring `aiter.mla.mla_decode_fwd`'s API style.

Caller contract:
  unified_kv:       [total_pages, D] BF16  (page_size=1)
    Conceptually merges the SWA ring buffer and the compressor paged cache
    of a single V4 layer. Slots in `[0, swa_pages)` reference SWA entries
    (state_slot * win + ring); slots in `[swa_pages, ...)` reference
    compressed-K entries (block_id * K_PER_BLOCK + slot_in_block).
  kv_indices: [total_indices] int32 — per-token slot lists, flat.
    Per-token entries live in
    `kv_indices[kv_indptr[t] : kv_indptr[t+1]]`.
    **All entries MUST be valid slot ids in [0, unified_kv.shape[0]).**
    The production decode index builder (``write_v4_paged_decode_indices``)
    emits ragged-packed indices with no sentinels; CG-padded tokens get
    a zero-length slice via ``indptr[t+1] == indptr[t]``. The kernel no
    longer carries a per-iter ``slot >= 0`` sentinel check.
  kv_indptr:  [N+1] int32 — true prefix sum (variable per-token len).
  attn_sink:        [H] per-head learnable softmax-denom bias (V4 specific).
  softmax_scale:    float.

Returns:
  out: [N, H, D] same dtype as q.

Numerics: online-softmax in log2 domain (exp2 with qk_scale = softmax_scale *
LOG2E), with attention sink folded as a virtual K. Bit-close to the PyTorch
reference (``_sparse_attn_ragged_torch``) within fp32-accumulation tolerance.

Architecture:
  - Small T (T * ceil(H/block_h) < ~1.7×CU): split-K + reduce-kernel path.
    Split kernel emits (m, l, acc) fp32 partials; reduce combines splits and
    folds attn_sink.
  - Large T: single-pass FUSED kernel that does softmax + sink + write in one
    shot (no partial-buffer alloc, no second kernel launch).

CUDAGraph-safe: kv_splits and tile config depend only on capture-time
shapes (``T``, ``H``); the kernels' early-return / segment-mask logic is
driven by ``kv_indptr`` values, which are runtime data but don't affect
the captured launch sequence.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.utils.device_info import get_num_sms
from vllm.models.deepseek_v4.amd.atom.model_ops.sparse_attn_v4 import _sparse_attn_ragged_torch
from aiter.ops.triton.attention.pa_decode_sparse import pa_decode_sparse

LOG2E = 1.4426950408889634  # log2(e); folded into qk_scale so softmax can use exp2.
_MAX_KV_SPLITS = 64  # Hard cap on kv_splits (see _kv_splits_heuristic).

# FP8 KV cache (1xGROUP_SIZE block-scale quantization).
#
# Storage: unified_kv[total_pages, D] in e4m3fnuz + kv_scales[total_pages,
# D // GROUP_SIZE] in fp32. Per-slot, D is split into NUM_GROUPS chunks of
# GROUP_SIZE elements; each chunk shares one fp32 scale.
# Dequant in-kernel: kv_bf16 = kv_fp8.to(fp32) * scale[d // GROUP_SIZE], cast
# back to q.dtype before the second dot.
#
# GROUP_SIZE=64 matches the user's 1x64 quant spec. V4-Pro: D=512 → 8 scales
# per slot, 4 bytes each → +6.25% storage on top of the fp8 pool (vs the
# halving from bf16→fp8 = 2× saving — net ~46% read bandwidth reduction).
_FP8_GROUP_SIZE = 64
_FP8_DTYPE = torch.float8_e4m3fnuz


@functools.lru_cache(maxsize=1)
def _cu_count() -> int:
    """Compute-unit count of the active GPU, queried once via aiter.

    Wrapped in ``lru_cache`` so the first decode call pays the device-property
    lookup and all subsequent calls hit the cache — important inside a hot
    decode loop and CUDAGraph capture (no data-dependent host work).
    """
    return get_num_sms()


# ---------------------------------------------------------------------------
# Heuristics — pure-Python, capture-time deterministic.
# ---------------------------------------------------------------------------


def _kernel_config(block_h: int) -> tuple[int, int, int]:
    """Pick (BLOCK_K, num_warps, num_stages) without autotune.

    Depends ONLY on ``block_h`` (a function of H, capture-time shape) so the
    config is identical between CUDAGraph capture and replay regardless of
    per-token K. In production ``kv_indices.shape[0]`` is a padded bucket
    whose value is unrelated to the true per-token kv_len, so any heuristic
    that reads it would mis-tune at capture time.

    Derived from autotune statistics over ~150 shapes on MI355:
      - BLOCK_K=16 dominated (~78% of best configs for D=512); D=512 has
        enough load width that wider K tiles spill regs more than they buy.
      - num_warps:
          block_h ≤ 32 → 4 warps; block_h ≥ 64 → 8 warps.
        Pre-v11 the threshold was 16, which gave H=32 nw=8 — that was a
        regression worth +30% geomean (max +57%) on H=32 across all T,
        from a Phase-1 fine-tune sweep on top of v05+v09. With block_h=32
        the MFMA tile is 32×16×D; 4 waves (256 threads) is the sweet
        spot for AMD wave64 register budget. 8 warps over-distribute and
        leave each warp with too little MFMA to amortize pipeline fill.
        block_h=64 still wants 8 warps (one wave per row tile is too
        little ILP).
      - num_stages=2 is the safe default — deeper pipelining (3) helps only
        when the K loop has many iterations; we cannot know per-token K at
        capture time, so the conservative pick avoids regressing short-K.
    """
    block_k = 16
    num_warps = 4 if block_h <= 32 else 8
    num_stages = 2
    return block_k, num_warps, num_stages


def _prev_pow2(n: int) -> int:
    """Largest power of two ≤ ``n``. For n < 1 returns 1."""
    if n < 1:
        return 1
    return 1 << (n.bit_length() - 1)


def _kv_splits_heuristic(
    T: int,
    H: int,
    block_h: int,
    num_cu: int | None = None,
    target_wg_per_cu: float = 2.0,
    max_kv_splits: int = _MAX_KV_SPLITS,
) -> int:
    """Pick KV_SPLITS to fill the GPU. CUDAGraph-safe: depends ONLY on
    capture-time scalars (``T``, ``H``, ``block_h``). Never reads any
    tensor value or shape — production callers must not assume kv_indices
    layout encodes per-token kv_len.

    Split-K trades a single-kernel pass for (split, reduce) two-kernel +
    partial-buffer allocation. The trade is worth it when the base grid
    ``T * ceil(H/block_h)`` underfills the device.

      base_ctas  = T * ceil(H / block_h)
      target_wg  = target_wg_per_cu * num_cu     (≈ 1.7x to hide load-imbalance)
      if base_ctas >= target_wg:  splits = 1     (grid already saturates GPU)
      else:                       splits = prev_pow2(min(target_wg/base_ctas,
                                                          max_kv_splits))

    ``max_kv_splits`` (default 64) caps the number of split-kernel CTAs per
    token. Higher values would buy more parallelism for bs=1 long-ctx, but
    when per-token K is short most splits fall-through and the launch
    overhead dominates. 64 is the sweet spot for MI300/MI355.

    Rounded DOWN to a power of two — rounding up over-splits when
    splits_to_fill isn't already pow2 (e.g. T=2 → 258 → 512 doubles the wg
    count past target, halves per-split work → 4× slowdown on bs=2 ctx=16384).
    """
    if num_cu is None:
        num_cu = _cu_count()
    target_wg = max(1, int(target_wg_per_cu * num_cu))
    head_blocks = max(1, (H + block_h - 1) // block_h)
    base_ctas = max(1, T * head_blocks)
    if base_ctas >= target_wg:
        return 1

    splits_to_fill = max(1, target_wg // base_ctas)
    return _prev_pow2(min(splits_to_fill, max_kv_splits))


# ---------------------------------------------------------------------------
# Kernels.
# ---------------------------------------------------------------------------


@triton.jit
def _paged_decode_fused_kernel(
    q_ptr,  # [N, H, D]
    unified_kv_ptr,  # [total_pages, D] bf16/fp16, or fp8 when QUANT_KV
    kv_scales_ptr,  # [total_pages, NUM_GROUPS] fp32 when QUANT_KV (dummy otherwise)
    kv_indices_ptr,  # [total_indices] int32
    kv_indptr_ptr,  # [N+1] int32
    attn_sink_ptr,  # [H]
    out_ptr,  # [N, H, D]
    q_stride_t,
    q_stride_h,
    q_stride_d,
    kv_stride_n,
    kv_stride_d,
    ks_stride_n,  # row stride of kv_scales (groups are contiguous, stride=1)
    out_stride_t,
    out_stride_h,
    out_stride_d,
    qk_scale,  # = softmax_scale * LOG2E
    log2e,  # = LOG2E, to lift natural-log sink into log2 domain
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    QUANT_KV: tl.constexpr,  # True → dequant fp8 KV via kv_scales
    GROUP_SIZE: tl.constexpr,  # scale block width along D (e.g. 64)
    NUM_GROUPS: tl.constexpr,  # D // GROUP_SIZE (constexpr; D % GROUP_SIZE == 0)
):
    """Single-pass online-softmax with sink folded inline — fast path for
    cases where ``kv_splits = 1`` (base grid already saturates the GPU). Skips
    the partial-buffer alloc + reduce-kernel launch that the 2-kernel
    split-K path needs.

    Grid: ``(N, ceil(H / BLOCK_H))``. One CTA owns one token and one head-tile.
    """
    t = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, BLOCK_D)
    h_mask = h_offs < H
    d_mask = d_offs < D

    q = tl.load(
        q_ptr
        + t * q_stride_t
        + h_offs[:, None] * q_stride_h
        + d_offs[None, :] * q_stride_d,
        mask=h_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    kv_start = tl.load(kv_indptr_ptr + t)
    kv_end = tl.load(kv_indptr_ptr + t + 1)
    kv_len = kv_end - kv_start
    num_tiles = tl.cdiv(kv_len, BLOCK_K)

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    k_offs = tl.arange(0, BLOCK_K)
    if QUANT_KV:
        # Compile-time per-D-element group index: d_offs // GROUP_SIZE has
        # NUM_GROUPS distinct values; redundant scale loads at the same
        # address are coalesced through L1, no per-element scalar issue.
        g_idx_per_d = d_offs // GROUP_SIZE
    # num_stages=3 on the inner K loop overrides the launch-time default
    # (2) for this loop only. Deeper SW pipeline keeps 2 in-flight KV
    # gathers (vs 1 with stages=2) while the current MFMA runs — better
    # hides AMD HBM gather latency on D=512. Cost: ~1 extra KV tile
    # (BLOCK_K*BLOCK_D*2 = 16KB at bf16) staged in regs/LDS per CTA.
    for j in tl.range(0, num_tiles, num_stages=3):
        k_start = j * BLOCK_K
        k_pos = k_start + k_offs
        valid = k_pos < kv_len  # in_range; no sentinel check (see contract)
        slot = tl.load(
            kv_indices_ptr + kv_start + k_pos,
            mask=valid,
            other=0,  # any in-bounds slot; the read is masked out below
        )

        kv_raw = tl.load(
            unified_kv_ptr
            + slot[:, None] * kv_stride_n
            + d_offs[None, :] * kv_stride_d,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        if QUANT_KV:
            # 1xGROUP_SIZE block-scale dequant via direct broadcast load —
            # avoids the explicit reshape + 3D intermediate that pinned
            # too many bf16 tiles in flight at once. The masked load with
            # d_offs // GROUP_SIZE as the column index produces a virtual
            # [BLOCK_K, BLOCK_D] scales tile but in IR is a coalesced
            # NUM_GROUPS-wide load per row.
            scales_full = tl.load(
                kv_scales_ptr + slot[:, None] * ks_stride_n + g_idx_per_d[None, :],
                mask=valid[:, None] & d_mask[None, :],
                other=0.0,
            ).to(q.dtype)
            kv = kv_raw.to(q.dtype) * scales_full
        else:
            kv = kv_raw

        scores = tl.dot(q, tl.trans(kv)) * qk_scale
        # K: drop h_mask from the per-iter where. In V4-Pro every realistic
        # (H, BLOCK_H) pair has H % BLOCK_H == 0 → h_mask is statically
        # all-True, but the runtime ``h_offs < H`` compare prevents Triton
        # from constant-folding. Masking only on ``valid`` is sufficient:
        # ``neg_large`` on invalid k positions makes exp2(scores - m) ≈ 0
        # in the subsequent dot, and the masked store at the end gates
        # invalid h rows from polluting the output.
        scores = tl.where(valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    # Fold attn_sink as a virtual K of weight 1. sink is a natural-log bias;
    # multiply by log2e so it lives in the same log2 domain as our online m_i.
    sink_raw = tl.load(attn_sink_ptr + h_offs, mask=h_mask, other=neg_large).to(
        tl.float32
    )
    sink = sink_raw * log2e
    m_final = tl.maximum(m_i, sink)
    alpha_kv = tl.exp2(m_i - m_final)
    alpha_sink = tl.exp2(sink - m_final)
    l_final = l_i * alpha_kv + alpha_sink

    denom = tl.maximum(l_final, 1.0e-30)
    out = tl.where(
        l_final[:, None] > 0.0, (acc * alpha_kv[:, None]) / denom[:, None], 0.0
    )
    tl.store(
        out_ptr
        + t * out_stride_t
        + h_offs[:, None] * out_stride_h
        + d_offs[None, :] * out_stride_d,
        out.to(out_ptr.dtype.element_ty),
        mask=h_mask[:, None] & d_mask[None, :],
    )


@triton.jit
def _paged_decode_split_kernel(
    q_ptr,  # [N, H, D]
    unified_kv_ptr,  # [total_pages, D] bf16/fp16, or fp8 when QUANT_KV
    kv_scales_ptr,  # [total_pages, NUM_GROUPS] fp32 when QUANT_KV (dummy otherwise)
    kv_indices_ptr,  # [total_indices] int32
    kv_indptr_ptr,  # [N+1] int32
    m_partial_ptr,  # [N, KV_SPLITS, H_padded] fp32
    l_partial_ptr,  # [N, KV_SPLITS, H_padded] fp32
    acc_partial_ptr,  # [N, KV_SPLITS, H_padded, D] fp32
    q_stride_t,
    q_stride_h,
    q_stride_d,
    kv_stride_n,
    kv_stride_d,
    ks_stride_n,  # row stride of kv_scales (groups are contiguous, stride=1)
    mp_stride_t,
    mp_stride_k,
    mp_stride_h,
    lp_stride_t,
    lp_stride_k,
    lp_stride_h,
    ap_stride_t,
    ap_stride_k,
    ap_stride_h,
    ap_stride_d,
    H: tl.constexpr,
    D: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    qk_scale,  # = softmax_scale * LOG2E
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    QUANT_KV: tl.constexpr,  # True → dequant fp8 KV via kv_scales
    GROUP_SIZE: tl.constexpr,  # scale block width along D (e.g. 64)
    NUM_GROUPS: tl.constexpr,  # D // GROUP_SIZE
):
    """3D split-K + exp2-softmax sparse paged-decode. Grid: (N, ceil(H/BLOCK_H), KV_SPLITS).

    Emits pre-sink (m, l, acc) partials in fp32. The reduce kernel folds
    ``attn_sink`` and combines splits.
    """
    t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_k = tl.program_id(2)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, BLOCK_D)
    h_mask = h_offs < H
    d_mask = d_offs < D

    q = tl.load(
        q_ptr
        + t * q_stride_t
        + h_offs[:, None] * q_stride_h
        + d_offs[None, :] * q_stride_d,
        mask=h_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    kv_start = tl.load(kv_indptr_ptr + t)
    kv_end = tl.load(kv_indptr_ptr + t + 1)
    kv_len = kv_end - kv_start

    # tiles_per_segment pattern from aiter's kernel_unified_attention_3d:
    # KV_SPLITS is constexpr; tiles_per_segment derived at runtime. Splits
    # whose tile range is past kv_len early-return WITHOUT writing — the
    # reduce kernel uses the same constexpr BLOCK_K/KV_SPLITS to compute
    # ``act_num_segments = cdiv(kv_len, tiles_per_segment * BLOCK_K)`` and
    # masks unwritten slots out of its load. This saves a per-empty-split
    # write of BLOCK_H*BLOCK_D fp32 (16 KB at BLOCK_H=16, BLOCK_D=512) which
    # dominated short-K + many-splits cases.
    tiles_per_segment = tl.cdiv(kv_len, KV_SPLITS * BLOCK_K)
    if pid_k * tiles_per_segment * BLOCK_K >= kv_len:
        return
    num_tiles = tl.cdiv(kv_len, BLOCK_K)
    tile_start = pid_k * tiles_per_segment
    tile_end = tl.minimum((pid_k + 1) * tiles_per_segment, num_tiles)

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    k_offs = tl.arange(0, BLOCK_K)
    if QUANT_KV:
        g_idx_per_d = d_offs // GROUP_SIZE
    # num_stages=3 (see fused kernel comment for rationale).
    for j in tl.range(tile_start, tile_end, num_stages=3):
        k_start = j * BLOCK_K
        k_pos = k_start + k_offs
        valid = k_pos < kv_len  # in_range; no sentinel check (see contract)
        slot = tl.load(
            kv_indices_ptr + kv_start + k_pos,
            mask=valid,
            other=0,  # any in-bounds slot; masked out below
        )

        kv_raw = tl.load(
            unified_kv_ptr
            + slot[:, None] * kv_stride_n
            + d_offs[None, :] * kv_stride_d,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        if QUANT_KV:
            scales_full = tl.load(
                kv_scales_ptr + slot[:, None] * ks_stride_n + g_idx_per_d[None, :],
                mask=valid[:, None] & d_mask[None, :],
                other=0.0,
            ).to(q.dtype)
            kv = kv_raw.to(q.dtype) * scales_full
        else:
            kv = kv_raw

        scores = tl.dot(q, tl.trans(kv)) * qk_scale
        # K (same as fused kernel): drop h_mask from per-iter where.
        scores = tl.where(valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    m_base = t * mp_stride_t + pid_k * mp_stride_k
    tl.store(m_partial_ptr + m_base + h_offs * mp_stride_h, m_i, mask=h_mask)
    l_base = t * lp_stride_t + pid_k * lp_stride_k
    tl.store(l_partial_ptr + l_base + h_offs * lp_stride_h, l_i, mask=h_mask)
    a_base = t * ap_stride_t + pid_k * ap_stride_k
    tl.store(
        acc_partial_ptr
        + a_base
        + h_offs[:, None] * ap_stride_h
        + d_offs[None, :] * ap_stride_d,
        acc,
        mask=h_mask[:, None] & d_mask[None, :],
    )


@triton.jit
def _paged_decode_reduce_kernel(
    m_partial_ptr,  # [N, KV_SPLITS, H_padded] fp32
    l_partial_ptr,  # [N, KV_SPLITS, H_padded] fp32
    acc_partial_ptr,  # [N, KV_SPLITS, H_padded, D] fp32
    attn_sink_ptr,  # [H]
    kv_indptr_ptr,  # [N+1] int32
    out_ptr,  # [N, H, D]
    mp_stride_t,
    mp_stride_k,
    mp_stride_h,
    lp_stride_t,
    lp_stride_k,
    lp_stride_h,
    ap_stride_t,
    ap_stride_k,
    ap_stride_h,
    ap_stride_d,
    out_stride_t,
    out_stride_h,
    out_stride_d,
    log2e,  # = LOG2E, used to convert natural-log sink → log2 domain
    H: tl.constexpr,
    D: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D_CHUNK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """2D-tile reduce: combine KV_SPLITS partials, fold attn_sink, write
    final output. Grid: ``(T, H, ceil(D / D_CHUNK))`` — one CTA owns one
    (token, single-head, D-chunk) tuple.

    Rewrite of the prior 3D-load reduce inspired by:
      - aiter ``_fwd_kernel_stage2`` (mla_decode_rope.py): scalar control
        flow + one D-tile per CTA, online accumulation across splits.
      - AKO4X ``hybrid_2d_reduce`` (B200 reference, +3.37x): merged 2D
        tile load ``[KV_SPLITS, D_CHUNK]`` replaces strided 3D access.

    Why this wins on MI355 split path (T ≤ ~256, kv_splits > 1):

    1. **2D tile fits registers**. The old 3D load tile
       ``[KV_SPLITS=64, BLOCK_H=1, BLOCK_D=512]`` = 32K fp32 = 128 KB —
       didn't fit in registers, spilled to LDS, and used a strided 3D
       address compute. New ``[KV_SPLITS, D_CHUNK]`` = 64×64×4 = 16 KB
       fits one wave's VGPR with headroom.
    2. **D-chunked grid widens occupancy**. Old grid was ``(T, H)`` —
       e.g. T=1 H=16 → 16 CTAs into 256 CUs (6% occupancy). New grid
       ``(T, H, D/D_CHUNK)`` → 16 × (512/64) = 128 CTAs at T=1 H=16
       (50% occupancy). Reduce was the latency bottleneck at small T.
    3. **Scalar control flow for sink fold**. Sink computation needs
       (m_max, l_combined) which only depend on (t, h) — same value
       across all dc programs for the same (t, h). They recompute it
       redundantly but it's a tiny scalar reduce; cheaper than passing
       through LDS.
    """
    t = tl.program_id(0)
    h = tl.program_id(1)
    dc = tl.program_id(2)

    d_offs = dc * D_CHUNK + tl.arange(0, D_CHUNK)
    k_offs = tl.arange(0, KV_SPLITS)
    d_mask = d_offs < D

    neg_large = -3.4028234663852886e38

    kv_start = tl.load(kv_indptr_ptr + t)
    kv_end = tl.load(kv_indptr_ptr + t + 1)
    kv_len = kv_end - kv_start
    # CTA-level early return for empty tokens (CUDAGraph padding, or any
    # caller-supplied zero-length slice). Split kernel skipped these without
    # writing partials → partial buffers hold garbage; the segm_mask path
    # below would still mask it correctly but consumes the masked-load BW
    # and the sink-fold arithmetic. Skipping the whole CTA also halves the
    # reduce-kernel cost on mixed-kv batches with many padded tokens.
    if kv_len == 0:
        out_off = t * out_stride_t + h * out_stride_h + d_offs * out_stride_d
        tl.store(
            out_ptr + out_off,
            tl.zeros([D_CHUNK], dtype=out_ptr.dtype.element_ty),
            mask=d_mask,
        )
        return
    tiles_per_segment = tl.cdiv(kv_len, KV_SPLITS * BLOCK_K)
    act_num_segments = tl.cdiv(kv_len, tl.maximum(tiles_per_segment, 1) * BLOCK_K)
    segm_mask = k_offs < act_num_segments

    # 1D loads for (m, l) along splits — single head h.
    m_p = tl.load(
        m_partial_ptr + t * mp_stride_t + k_offs * mp_stride_k + h * mp_stride_h,
        mask=segm_mask,
        other=neg_large,
    )  # [KV_SPLITS]
    l_p = tl.load(
        l_partial_ptr + t * lp_stride_t + k_offs * lp_stride_k + h * lp_stride_h,
        mask=segm_mask,
        other=0.0,
    )  # [KV_SPLITS]

    # 2D-tile load for acc partials — the key change vs old 3D-strided load.
    a_p = tl.load(
        acc_partial_ptr
        + t * ap_stride_t
        + k_offs[:, None] * ap_stride_k
        + h * ap_stride_h
        + d_offs[None, :] * ap_stride_d,
        mask=segm_mask[:, None] & d_mask[None, :],
        other=0.0,
    )  # [KV_SPLITS, D_CHUNK]

    # Combine across splits.
    m_max = tl.max(m_p, axis=0)  # scalar
    alpha_split = tl.exp2(m_p - m_max)  # [KV_SPLITS]
    l_combined = tl.sum(l_p * alpha_split, axis=0)  # scalar
    acc_combined = tl.sum(a_p * alpha_split[:, None], axis=0)  # [D_CHUNK]

    # Fold attn_sink (recomputed across dc — scalar work, negligible).
    sink_raw = tl.load(attn_sink_ptr + h).to(tl.float32)
    sink = sink_raw * log2e
    m_final = tl.maximum(m_max, sink)
    alpha_kv = tl.exp2(m_max - m_final)
    alpha_sink = tl.exp2(sink - m_final)
    l_final = l_combined * alpha_kv + alpha_sink

    denom = tl.maximum(l_final, 1.0e-30)
    # Direct divide (acc*alpha_kv)/denom, matching the single-CTA reference.
    # The prior `acc * (alpha_kv/denom)` precomputed a reciprocal-scaled scalar
    # (~1 extra ulp per element). Under split-K that per-kv_splits rounding
    # diverges across batch shapes (T) and, via MTP greedy spec-acceptance,
    # flips tokens — breaking MTP losslessness. The D-chunked multi-CTA layout
    # (perf) is untouched; only the final normalize arithmetic changes.
    acc_final = acc_combined * alpha_kv
    out = tl.where(l_final > 0.0, acc_final / denom, 0.0)

    tl.store(
        out_ptr + t * out_stride_t + h * out_stride_h + d_offs * out_stride_d,
        out.to(out_ptr.dtype.element_ty),
        mask=d_mask,
    )


# ---------------------------------------------------------------------------
# Wrapper.
# ---------------------------------------------------------------------------


def _sparse_attn_v4_paged_decode_triton(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    kv_scales: torch.Tensor | None = None,
    block_h: int | None = None,
    kv_splits: int | None = None,
    block_k: int | None = None,
) -> torch.Tensor:
    """V4 sparse decode Triton implementation: split-K with FUSED fast path,
    exp2 softmax, CG-safe heuristic. ``block_h`` and ``kv_splits`` are
    escape hatches for benchmarks; production callers pass neither.

    When ``kv_scales`` is provided, ``unified_kv`` must be e4m3fnuz and
    ``kv_scales`` must be ``[total_pages, D // GROUP_SIZE]`` fp32 — 1xGROUP_SIZE
    block-scale quantization. Dequant happens in-kernel; the dot still runs
    in q.dtype.
    """
    if not q.is_cuda:
        raise RuntimeError(
            "Triton sparse_attn_v4_paged_decode requires CUDA/HIP tensors"
        )
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(
            f"sparse_attn_v4_paged_decode expects fp16/bf16 q, got {q.dtype}"
        )

    quant_kv = kv_scales is not None
    if quant_kv:
        if unified_kv.dtype != _FP8_DTYPE:
            raise RuntimeError(
                f"kv_scales supplied but unified_kv is {unified_kv.dtype}, "
                f"expected {_FP8_DTYPE}"
            )
        if kv_scales.dtype != torch.float32:
            raise RuntimeError(f"kv_scales must be fp32, got {kv_scales.dtype}")
        D_check = unified_kv.shape[-1]
        if D_check % _FP8_GROUP_SIZE != 0:
            raise RuntimeError(
                f"D={D_check} must be divisible by GROUP_SIZE={_FP8_GROUP_SIZE}"
            )
        expected_g = D_check // _FP8_GROUP_SIZE
        if kv_scales.shape != (unified_kv.shape[0], expected_g):
            raise RuntimeError(
                f"kv_scales shape {tuple(kv_scales.shape)} does not match "
                f"expected ({unified_kv.shape[0]}, {expected_g})"
            )
        if kv_scales.stride(-1) != 1:
            kv_scales = kv_scales.contiguous()
    else:
        if unified_kv.dtype != q.dtype:
            raise RuntimeError(
                f"unified_kv dtype mismatch: kv={unified_kv.dtype}, q={q.dtype}"
            )

    T, H, D = q.shape
    out = torch.empty_like(q)

    if block_h is None:
        block_h = triton.next_power_of_2(min(H, 64))
    else:
        block_h = triton.next_power_of_2(block_h)
    block_h = max(block_h, 16)  # AMD MFMA min tile

    n_head_blocks = (H + block_h - 1) // block_h
    h_padded = n_head_blocks * block_h
    block_d = triton.next_power_of_2(D)

    if kv_splits is None:
        kv_splits = _kv_splits_heuristic(T, H, block_h)

    qk_scale = float(softmax_scale) * LOG2E
    _bk, num_warps, num_stages = _kernel_config(block_h)
    if block_k is None:
        # fp8 dequant inflates per-tile ALU work ~4×; a wider K tile amortizes
        # the per-tile dequant cost (scale load + cast + multiply) over more
        # MFMA work. Empirically BLOCK_K=32 wins ~20% over BLOCK_K=16 on fp8
        # (bs=512 ctx=4096: 3000µs → 2300µs) without hurting bf16.
        block_k = 32 if quant_kv else _bk

    # Kernel reads (kv_scales_ptr, ks_stride_n) only when QUANT_KV — supply a
    # dummy 1-element fp32 tensor on the bf16 path so the launch signature
    # stays uniform (avoids a separate JIT specialization per call).
    if quant_kv:
        kv_scales_arg = kv_scales
        ks_stride_n_arg = kv_scales.stride(0)
        num_groups_arg = D // _FP8_GROUP_SIZE
    else:
        kv_scales_arg = q.new_empty(1, dtype=torch.float32)
        ks_stride_n_arg = 1
        # NUM_GROUPS still needs a constexpr value (unused at compile time
        # because the QUANT_KV=False branch elides the dequant code).
        num_groups_arg = 1

    # Fast path: when the base grid (T * n_head_blocks) already saturates the
    # GPU, kv_splits=1 and a single-pass fused kernel beats split+reduce by
    # skipping the partial-buffer alloc and the second kernel launch.
    if kv_splits == 1:
        grid_fused = (T, n_head_blocks)
        _paged_decode_fused_kernel[grid_fused](
            q,
            unified_kv,
            kv_scales_arg,
            kv_indices,
            kv_indptr,
            attn_sink,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            unified_kv.stride(0),
            unified_kv.stride(1),
            ks_stride_n_arg,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            qk_scale,
            LOG2E,
            H,
            D,
            BLOCK_H=block_h,
            BLOCK_D=block_d,
            BLOCK_K=block_k,
            QUANT_KV=quant_kv,
            GROUP_SIZE=_FP8_GROUP_SIZE,
            NUM_GROUPS=num_groups_arg,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out

    # Split-K path: split kernel writes (m, l, acc) partials in log2 domain;
    # reduce kernel combines them, folds attn_sink, writes final output.
    # Empty splits early-return without writing — reduce masks them out using
    # the same constexpr BLOCK_K to derive ``act_num_segments``.
    m_partial = torch.empty(
        (T, kv_splits, h_padded), dtype=torch.float32, device=q.device
    )
    l_partial = torch.empty_like(m_partial)
    acc_partial = torch.empty(
        (T, kv_splits, h_padded, D), dtype=torch.float32, device=q.device
    )

    grid_split = (T, n_head_blocks, kv_splits)
    _paged_decode_split_kernel[grid_split](
        q,
        unified_kv,
        kv_scales_arg,
        kv_indices,
        kv_indptr,
        m_partial,
        l_partial,
        acc_partial,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        unified_kv.stride(0),
        unified_kv.stride(1),
        ks_stride_n_arg,
        m_partial.stride(0),
        m_partial.stride(1),
        m_partial.stride(2),
        l_partial.stride(0),
        l_partial.stride(1),
        l_partial.stride(2),
        acc_partial.stride(0),
        acc_partial.stride(1),
        acc_partial.stride(2),
        acc_partial.stride(3),
        H,
        D,
        kv_splits,
        qk_scale,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        QUANT_KV=quant_kv,
        GROUP_SIZE=_FP8_GROUP_SIZE,
        NUM_GROUPS=num_groups_arg,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # 2D-tile reduce: grid = (T, H, ceil(D/D_CHUNK)). One CTA per
    # (token, single-head, D-chunk). Adaptive D_CHUNK based on whether the
    # base reduce grid (T*H) already saturates the GPU:
    #   - large T*H (≥ 2*num_CU=512 on MI355): the grid is already at or
    #     above the launch target; further D-splitting just over-fragments
    #     (T=128 H=128 with D_CHUNK=64 → 131K CTAs → 2× slowdown). Use
    #     D_CHUNK = block_d (1 d-block per CTA, grid = T*H).
    #   - small T*H: D-split widens the grid to fill CUs at small-T
    #     latency-critical decode. The 2D tile [KV_SPLITS, D_CHUNK] should
    #     stay ≤ 16 KB fp32 to fit registers.
    base_grid_t_h = T * H
    target_reduce_wg = 2 * _cu_count()
    if base_grid_t_h >= target_reduce_wg:
        d_chunk = block_d
    else:
        d_chunks_needed = max(1, target_reduce_wg // base_grid_t_h)
        d_chunks_needed = min(d_chunks_needed, block_d // 32)
        d_chunk = max(32, triton.next_power_of_2(block_d // d_chunks_needed))
    grid_reduce = (T, H, (D + d_chunk - 1) // d_chunk)
    _paged_decode_reduce_kernel[grid_reduce](
        m_partial,
        l_partial,
        acc_partial,
        attn_sink,
        kv_indptr,
        out,
        m_partial.stride(0),
        m_partial.stride(1),
        m_partial.stride(2),
        l_partial.stride(0),
        l_partial.stride(1),
        l_partial.stride(2),
        acc_partial.stride(0),
        acc_partial.stride(1),
        acc_partial.stride(2),
        acc_partial.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        LOG2E,
        H,
        D,
        kv_splits,
        BLOCK_D=block_d,
        D_CHUNK=d_chunk,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return out


def sparse_attn_v4_paged_decode_reference(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    kv_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure-torch reference. Materialises per-token KV via gather and reuses
    `_sparse_attn_ragged_torch`. Slow but correct — for unit tests / dump-bisect.

    Uses the longest per-token span as the K dimension for the dense
    `topk_idxs` tensor; shorter spans are tail-padded with `-1`. When
    ``kv_scales`` is given, dequantizes unified_kv (fp8) → q.dtype up front
    via per-slot 1xGROUP_SIZE block-scale multiply.
    """
    if kv_scales is not None:
        # Dequant the whole pool once — only the gather sees a real tensor,
        # so this is just a reference path; not optimized.
        total_pages, D = unified_kv.shape
        num_groups = D // _FP8_GROUP_SIZE
        kv_fp32 = unified_kv.to(torch.float32)
        scales_expanded = (
            kv_scales.view(total_pages, num_groups, 1)
            .expand(total_pages, num_groups, _FP8_GROUP_SIZE)
            .reshape(total_pages, D)
        )
        unified_kv = (kv_fp32 * scales_expanded).to(q.dtype)
    T = q.size(0)
    indptr = kv_indptr.to(torch.int64)
    spans = (indptr[1:] - indptr[:T]).clamp(min=0)
    k_dim = int(spans.max().item()) if T > 0 else 1
    if k_dim == 0:
        k_dim = 1
    topk_idxs = torch.full((T, k_dim), -1, device=q.device, dtype=torch.int32)
    for t in range(T):
        s = int(indptr[t].item())
        n = int(spans[t].item())
        if n > 0:
            topk_idxs[t, :n] = kv_indices[s : s + n].to(torch.int32)
    return _sparse_attn_ragged_torch(q, unified_kv, attn_sink, topk_idxs, softmax_scale)


def sparse_attn_v4_paged_decode(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    kv_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """V4 decode sparse attention over a unified KV pool with paged indices.

    When ``kv_scales`` is provided, ``unified_kv`` must be fp8 (e4m3fnuz) and
    will be dequantized in-kernel using 1xGROUP_SIZE (default 64) block scales.
    """
    if get_gfx() == "gfx1250":
        return pa_decode_sparse(
            q,
            unified_kv,
            kv_indices,
            kv_indptr,
            attn_sink,
            softmax_scale,
            has_invalid=False,
            kv_scales=kv_scales,
        )
    return _sparse_attn_v4_paged_decode_triton(
        q,
        unified_kv,
        kv_indices,
        kv_indptr,
        attn_sink,
        softmax_scale,
        kv_scales=kv_scales,
    )
