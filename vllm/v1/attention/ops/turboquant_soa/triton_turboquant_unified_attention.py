# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified (prefill + decode) Triton attention kernel for TurboQuant.

Structure is ported from ``vllm/v1/attention/ops/triton_unified_attention.py``
(the AITER unified attention kernel already upstreamed into vLLM). Only the K
and V load sites are replaced: instead of reading raw fp16 keys/values from
two contiguous caches, this kernel reads TurboQuant-packed bytes from a single
combined cache and dequantizes on the fly inside the tile loop.

Benefits over the v1/v2 decode-only kernels:

1. GQA heads are stacked into ``BLOCK_M`` and the Q·K and P·V ops are proper
   ``tl.dot`` tensor-core operations (MFMA on MI300X).
2. The same kernel handles decode (``BLOCK_Q=1``) and prefill
   (``BLOCK_Q>1``) -- no more Python per-request for-loop for continuation
   chunks.
3. Only the subset of features exercised by the current TQ paths is kept
   (causal, GQA). Sinks / softcap / ALiBi / sliding-window / qq-bias /
   mm-prefix are deferred to follow-ups.

This is an opt-in v3 path behind ``VLLM_TQ_DECODE_V3``.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .triton_turboquant_decode import _use_fp8_e4b15
from .triton_turboquant_decode_v2 import build_pair_lut

# reduce_segments is KV-format-agnostic: by the time it runs, K/V have been
# consumed and only (max, expsum, partial_output) triples remain. Reuse the
# baseline's implementation verbatim to avoid code duplication.
from vllm.v1.attention.ops.triton_unified_attention import reduce_segments

_is_hip = current_platform.is_rocm()


# ---------------------------------------------------------------------------
# Helper: find which sequence in the batch a global q-block index belongs to.
# Identical to the unified kernel's binary search.
# ---------------------------------------------------------------------------


@triton.jit
def _find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val
        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid
    return left - 1


# ---------------------------------------------------------------------------
# Fused Q-rotation prologue (Opt: eliminates the launcher-side
#   q_rot = (query.float() @ PiT).to(dtype).contiguous()
# chain, which cost ~50-60 us/step at D=64 — all launch-bound, not compute-
# bound. The fused version does one small MFMA inside the attention kernel,
# once per program, before the KV-tile loop. PiT is [D,D] fp32 and fits in
# L1 easily (16KB at D=64, 64KB at D=128).
#
# Algebra is identical to the launcher path: both compute Q_rot = Q @ PiT
# with fp32 accumulate. Differences are rounding-order only (MFMA tiling
# vs rocBLAS tiling), well within the 4-bit KV quant noise floor.
# ---------------------------------------------------------------------------


@triton.jit
def _tq_fuse_q_rotation(
    Q,  # [BLOCK_M, HEAD_SIZE_PADDED] — raw query in Q.dtype
    PiT_ptr,
    PiT_stride_0: tl.int64,
    PiT_stride_1: tl.int64,
    dim_mask,  # [HEAD_SIZE_PADDED] int1 — valid head dims
    HEAD_SIZE_PADDED: tl.constexpr,
):
    """Fused Q @ PiT prologue. Called once per program for the MSE-key path
    when the launcher has passed the raw (un-rotated) query. Returns the
    rotated Q in the original dtype.
    """
    d_offs = tl.arange(0, HEAD_SIZE_PADDED)
    pit_offsets = d_offs[:, None] * PiT_stride_0 + d_offs[None, :] * PiT_stride_1
    pit_mask = dim_mask[:, None] & dim_mask[None, :]
    PiT_tile = tl.load(PiT_ptr + pit_offsets, mask=pit_mask, other=0.0).to(tl.float32)
    # input_precision="ieee" pins both inputs to full fp32 MFMA (no TF32
    # truncation). allow_tf32 is intentionally omitted — Triton rejects
    # passing both. This matches the launcher's rocBLAS fp32 GEMM in
    # algebra; rounding-order differs, diff is <= a few fp16 ulp.
    Q_rot = tl.dot(
        Q.to(tl.float32),
        PiT_tile,
        input_precision="ieee",
    )
    return Q_rot.to(Q.dtype)


# ---------------------------------------------------------------------------
# TQ K-tile dequant: returns K_T : [HEAD_SIZE_PADDED, TILE_SIZE] in Q.dtype
# ready for tl.dot(Q, K_T).
#
# Packed slot layout (per (block, pos, kv_head)) is the same as v1/v2:
#   MSE path:   [ key_idx_bytes (MSE_BYTES) | key_norm_fp16 (2B) | ...V... ]
#   FP8 path:   [ fp8_key_bytes (D)                             | ...V... ]
# ---------------------------------------------------------------------------


@triton.jit
def _tq_load_k_tile(
    KV_cache_ptr,
    KV_cache_u16_ptr,  # uint16 view of KV_cache (same storage)
    data_bases,  # [TILE_SIZE] int64 — byte offset to each token's DATA region
    knorm_u16_addrs,  # [TILE_SIZE] int64 — u16 element index for each token's K-norm
    d_offs,  # [HEAD_SIZE_PADDED]
    d_mask,  # [HEAD_SIZE_PADDED] int1
    tile_mask,  # [TILE_SIZE] int1
    Centroids_ptr,
    Pair_lut_ptr,
    OUT_DTYPE: tl.constexpr,  # tl.float16 or tl.bfloat16
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,  # HEAD_SIZE_PADDED — needed for pair-LUT reshape
    MSE_BITS: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    KEY_FP8: tl.constexpr,
    USE_PAIR_LUT: tl.constexpr,
    NORM_CORRECTION: tl.constexpr,
    FP8_E4B15: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    """Load + dequantize a TILE_SIZE × HEAD_SIZE block of keys and return
    the transposed tile K_T : [HEAD_SIZE_PADDED, TILE_SIZE].

    Opt#3 SoA layout: packed K data is at `data_bases[t] + [0, MSE_BYTES)`
    for MSE keys (or `[0, D)` for FP8). The per-token K-norm lives in the
    per-block SoA metadata region; `knorm_u16_addrs` already encodes its
    u16 element index. For decode tiles aligned with blocks, these addresses
    are contiguous → one coalesced wide load replaces TILE_SIZE scattered
    loads (the whole point of Opt#3).
    """
    if KEY_FP8:
        k_addrs = data_bases[:, None] + d_offs[None, :]
        k_raw = tl.load(
            KV_cache_ptr + k_addrs,
            mask=tile_mask[:, None] & d_mask[None, :],
            other=0,
        )
        if FP8_E4B15:
            k_f32 = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
        else:
            k_f32 = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        K = k_f32  # [TILE_SIZE, HEAD_SIZE_PADDED]
    else:
        # MSE path: gather packed key indices + centroid LUT.
        if MSE_BITS == 4 and USE_PAIR_LUT:
            # FLUTE pair-LUT fast path — load each packed byte once, decode
            # both nibbles, single 3-D gather returns (T[lo], T[hi]) per byte.
            HALF_D: tl.constexpr = BLOCK_D // 2
            half_offs = tl.arange(0, HALF_D)
            byte_mask = (half_offs * 2) < HEAD_DIM
            byte_addrs = data_bases[:, None] + half_offs[None, :]
            byte_raw = tl.load(
                KV_cache_ptr + byte_addrs,
                mask=tile_mask[:, None] & byte_mask[None, :],
                other=0,
            ).to(tl.int32)
            lo_idx = byte_raw & 0xF
            hi_idx = (byte_raw >> 4) & 0xF
            pair_key = lo_idx * N_CENTROIDS + hi_idx
            pair_slot = tl.arange(0, 2)
            c_pair = tl.load(
                Pair_lut_ptr + pair_key[:, :, None] * 2 + pair_slot[None, None, :],
                mask=(tile_mask[:, None, None] & byte_mask[None, :, None]),
                other=0.0,
            )
            c_vals = tl.reshape(c_pair, [TILE_SIZE, BLOCK_D])
        elif MSE_BITS == 4:
            half_idx = d_offs // 2
            nibble_shift = (d_offs % 2) * 4
            mse_addrs = data_bases[:, None] + half_idx[None, :]
            mse_raw = tl.load(
                KV_cache_ptr + mse_addrs,
                mask=tile_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            mse_idx = (mse_raw >> nibble_shift[None, :]) & 0xF
            c_vals = tl.load(
                Centroids_ptr + mse_idx,
                mask=tile_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
        else:
            # Generic bit extraction (3-bit, etc.)
            mse_bit_off = d_offs * MSE_BITS
            mse_byte_idx = mse_bit_off // 8
            mse_bit_shift = mse_bit_off % 8
            mse_mask_val = (1 << MSE_BITS) - 1
            mse_addrs0 = data_bases[:, None] + mse_byte_idx[None, :]
            mse_raw0 = tl.load(
                KV_cache_ptr + mse_addrs0,
                mask=tile_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            mse_raw1 = tl.load(
                KV_cache_ptr + mse_addrs0 + 1,
                mask=tile_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            raw16 = mse_raw0 | (mse_raw1 << 8)
            mse_idx = (raw16 >> mse_bit_shift[None, :]) & mse_mask_val
            c_vals = tl.load(
                Centroids_ptr + mse_idx,
                mask=tile_mask[:, None] & d_mask[None, :],
                other=0.0,
            )

        # Opt#1: 1/||c_vec|| is pre-folded into the stored K-norm at store
        # time, so the kernel doesn't recompute norm-correction here.
        # Opt#3: K-norms for a tile are contiguous in the per-block SoA
        # region when the tile lies within one block (always true for
        # aligned decode tiles with TILE_SIZE == BLOCK_SIZE) — one coalesced
        # u16 load replaces TILE_SIZE scattered 2-byte loads.
        norm_u16 = tl.load(KV_cache_u16_ptr + knorm_u16_addrs, mask=tile_mask, other=0)
        vec_norms = norm_u16.to(tl.float16, bitcast=True).to(tl.float32)
        K = c_vals * vec_norms[:, None]  # [TILE_SIZE, HEAD_SIZE_PADDED]

    K_T = tl.trans(K.to(OUT_DTYPE))  # [HEAD_SIZE_PADDED, TILE_SIZE]
    _ = HEAD_DIM
    return K_T


# ---------------------------------------------------------------------------
# TQ V-tile dequant: returns V : [TILE_SIZE, HEAD_SIZE_PADDED] in Q.dtype
# ready for tl.dot(P, V).
# ---------------------------------------------------------------------------


@triton.jit
def _tq_load_v_tile(
    KV_cache_ptr,
    KV_cache_u16_ptr,  # uint16 view of KV_cache (same storage)
    val_bases,  # [TILE_SIZE] int64 — byte offset to each token's V-data
    vscale_u16_addrs,  # [TILE_SIZE] int64 — u16 element index for V-scale
    vzero_u16_addrs,  # [TILE_SIZE] int64 — u16 element index for V-zero
    d_offs,
    d_mask,
    tile_mask,
    OUT_DTYPE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VQB: tl.constexpr,
):
    """Load + dequantize a TILE_SIZE × HEAD_SIZE block of values.

    Opt#3 SoA layout: packed V data is at `val_bases[t] + [0, VAL_DATA_BYTES)`
    (V-data immediately follows K-data within the slot's data region, and
    `val_bases` is precomputed by the caller as `data_base + KEY_DATA_BYTES`).
    V-scale / V-zero live in the per-block SoA metadata region at indices
    `vscale_u16_addrs` / `vzero_u16_addrs`. For tiles aligned to block
    boundaries, those addresses are contiguous → one coalesced wide load
    per field instead of TILE_SIZE scattered 2-byte loads.
    """
    if VQB == 4:
        vb_idx = d_offs // 2
        vb_shift = (d_offs % 2) * 4
        val_addrs = val_bases[:, None] + vb_idx[None, :]
        val_raw = tl.load(
            KV_cache_ptr + val_addrs,
            mask=tile_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        v_idx = ((val_raw >> vb_shift[None, :]) & 0xF).to(tl.float32)
    else:  # VQB == 3
        val_bit_off = d_offs * 3
        val_byte_idx = val_bit_off // 8
        val_bit_shift = val_bit_off % 8
        val_addrs0 = val_bases[:, None] + val_byte_idx[None, :]
        val_raw0 = tl.load(
            KV_cache_ptr + val_addrs0,
            mask=tile_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        val_raw1 = tl.load(
            KV_cache_ptr + val_addrs0 + 1,
            mask=tile_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        raw16 = val_raw0 | (val_raw1 << 8)
        v_idx = ((raw16 >> val_bit_shift[None, :]) & 0x7).to(tl.float32)

    # SoA scale / zero loads — coalesced on aligned decode tiles.
    scale_u16 = tl.load(KV_cache_u16_ptr + vscale_u16_addrs, mask=tile_mask, other=0)
    zero_u16 = tl.load(KV_cache_u16_ptr + vzero_u16_addrs, mask=tile_mask, other=0)
    v_scales = scale_u16.to(tl.float16, bitcast=True).to(tl.float32)
    v_zeros = zero_u16.to(tl.float16, bitcast=True).to(tl.float32)

    V = v_idx * v_scales[:, None] + v_zeros[:, None]  # [TILE_SIZE, HEAD_SIZE_PADDED]
    _ = HEAD_DIM
    return V.to(OUT_DTYPE)


# ---------------------------------------------------------------------------
# Unified 2D attention kernel over a TurboQuant KV cache (causal only, GQA).
# Structure follows ``kernel_unified_attention_2d`` from
# triton_unified_attention.py with:
#   * Q-head GQA stacking into BLOCK_M
#   * tensor-core tl.dot(Q, K) and tl.dot(P, V)
#   * causal + query-padding masking
#   * online softmax
# and K/V loads replaced by the TQ helpers above.
# ---------------------------------------------------------------------------


@triton.jit
def kernel_tq_unified_attention_2d(
    output_ptr,  # [num_tokens, Hq, D]
    query_ptr,  # [num_tokens, Hq, D] — raw if FUSE_Q_ROT else rotated
    KV_cache_ptr,  # [num_blocks, block_size, Hk, padded_slot] uint8 - TQ packed
    KV_cache_u16_ptr,  # uint16 view of same storage — Opt#2 wide metadata loads
    Centroids_ptr,  # [n_centroids] fp32
    Pair_lut_ptr,  # [N*N, 2] fp32 (pair-LUT for 4-bit MSE) or dummy
    PiT_ptr,  # [D, D] fp32 — only dereferenced when FUSE_Q_ROT == 1
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    query_start_len_ptr,  # [num_seqs+1]
    sinks_ptr,  # [Hq] fp32 — per-head sink logits; dereferenced only when USE_SINKS
    scale,  # float32
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    stride_cache_block: tl.int64,
    pit_stride_0: tl.int64,
    pit_stride_1: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    num_seqs: tl.int32,
    # TQ layout constants
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    KEY_FP8: tl.constexpr,
    USE_PAIR_LUT: tl.constexpr,
    # Opt#3 SoA layout constants
    NUM_KV_HEADS: tl.constexpr,
    KEY_DATA_BYTES: tl.constexpr,  # MSE_BYTES for MSE, HEAD_SIZE for FP8
    META_REGION_OFFSET: tl.constexpr,  # bytes: bs * H * (KD+VD)
    NUM_SOA_FIELDS: tl.constexpr,  # 3 for MSE, 2 for FP8
    SOA_K_NORM: tl.constexpr,  # 0 for MSE, unused for FP8
    SOA_V_SCALE: tl.constexpr,  # 1 (MSE) / 0 (FP8)
    SOA_V_ZERO: tl.constexpr,  # 2 (MSE) / 1 (FP8)
    NORM_CORRECTION: tl.constexpr = 0,
    FP8_E4B15: tl.constexpr = 0,
    FUSE_Q_ROT: tl.constexpr = 0,  # Fused Q @ PiT prologue when 1 (MSE path)
    USE_SINKS: tl.constexpr = 0,  # Per-head sink logit folded into softmax denom
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_idx = _find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : [BLOCK_M, HEAD_SIZE_PADDED]
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    if FUSE_Q_ROT:
        Q = _tq_fuse_q_rotation(
            Q,
            PiT_ptr,
            pit_stride_0,
            pit_stride_1,
            dim_mask,
            HEAD_SIZE_PADDED,
        )

    block_table_offset = seq_idx * block_table_stride

    # Online softmax state.
    # When USE_SINKS=1 we fold a per-head sink logit s_h into the softmax
    # denominator by initializing M = s_h, L = 1.0 (== exp(s_h - s_h)).
    # The standard online-softmax update then correctly rescales the sink
    # contribution as real tokens update M, and the final acc / L gives
    # out_h = sum_i exp(q.k_i)*V_i / (exp(s_h) + sum_i exp(q.k_i)).
    # Masked rows (query_mask_1 == 0) get -inf so the sink cannot pollute
    # invalid (q_token, q_head) positions outside the real query range.
    if USE_SINKS:
        M = tl.load(
            sinks_ptr + query_offset_1, mask=query_mask_1, other=float("-inf")
        ).to(tl.float32)
    else:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - cur_batch_query_len

    # Longest key prefix any query row in this q-block can attend to.
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    num_tiles = tl.cdiv(max_seq_prefix_len, TILE_SIZE)

    for j in range(0, num_tiles):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        # Opt#3 SoA addressing: compute data + SoA metadata addresses once
        # per tile; the load helpers below issue coalesced wide loads from
        # these addresses for decode tiles that align with block boundaries.
        slot_within_block = (seq_offset % BLOCK_SIZE).to(tl.int64)
        block_base = physical_block_idx * stride_cache_block
        DATA_BYTES_PER_SLOT: tl.constexpr = KEY_DATA_BYTES + VAL_DATA_BYTES
        data_bases = (
            block_base
            + slot_within_block * (NUM_KV_HEADS * DATA_BYTES_PER_SLOT)
            + tl.cast(kv_head_idx, tl.int64) * DATA_BYTES_PER_SLOT
        )
        val_bases = data_bases + KEY_DATA_BYTES

        head_meta_u16_base = (block_base + META_REGION_OFFSET) // 2 + tl.cast(
            kv_head_idx, tl.int64
        ) * (NUM_SOA_FIELDS * BLOCK_SIZE)
        knorm_u16_addrs = (
            head_meta_u16_base + SOA_K_NORM * BLOCK_SIZE + slot_within_block
        )
        vscale_u16_addrs = (
            head_meta_u16_base + SOA_V_SCALE * BLOCK_SIZE + slot_within_block
        )
        vzero_u16_addrs = (
            head_meta_u16_base + SOA_V_ZERO * BLOCK_SIZE + slot_within_block
        )

        # ---- K : [HEAD_SIZE_PADDED, TILE_SIZE] in Q.dtype ----
        K_T = _tq_load_k_tile(
            KV_cache_ptr,
            KV_cache_u16_ptr,
            data_bases,
            knorm_u16_addrs,
            offs_d,
            dim_mask,
            tile_mask,
            Centroids_ptr,
            Pair_lut_ptr,
            OUT_DTYPE=Q.dtype,
            HEAD_DIM=HEAD_SIZE,
            BLOCK_D=HEAD_SIZE_PADDED,
            MSE_BITS=MSE_BITS,
            N_CENTROIDS=N_CENTROIDS,
            KEY_FP8=KEY_FP8,
            USE_PAIR_LUT=USE_PAIR_LUT,
            NORM_CORRECTION=NORM_CORRECTION,
            FP8_E4B15=FP8_E4B15,
            TILE_SIZE=TILE_SIZE,
        )

        # ---- V : [TILE_SIZE, HEAD_SIZE_PADDED] in Q.dtype ----
        V = _tq_load_v_tile(
            KV_cache_ptr,
            KV_cache_u16_ptr,
            val_bases,
            vscale_u16_addrs,
            vzero_u16_addrs,
            offs_d,
            dim_mask,
            tile_mask,
            OUT_DTYPE=Q.dtype,
            HEAD_DIM=HEAD_SIZE,
            VQB=VQB,
        )

        # S : [BLOCK_M, TILE_SIZE]
        S = scale * tl.dot(Q, K_T)

        # Causal + query-padding mask
        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = seq_offset[None, :] <= query_abs_pos
        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
            S,
            float("-inf"),
        )

        # Online softmax
        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j

        # acc += P · V
        acc += tl.dot(P.to(V.dtype), V)

    # Epilogue: normalize and store
    acc = acc / L[:, None]

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )
    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


# ---------------------------------------------------------------------------
# Unified 3D (split-KV) attention kernel over a TurboQuant KV cache.
#
# Same loop body as the 2D kernel above, but with an extra ``segm_idx`` grid
# axis: each program walks only ``cdiv(seq_len, NUM_SEGMENTS_PER_SEQ * TILE)``
# tiles of the KV history and writes a *partial* (max, expsum, acc) triple to
# a scratch buffer. A subsequent ``reduce_segments`` kernel merges the
# ``NUM_SEGMENTS_PER_SEQ`` partials per (q_token, q_head) via the standard
# online-softmax merge.
#
# This gives the decode path ``num_seqs × Hk × NUM_SEGMENTS_PER_SEQ`` CTAs
# instead of just ``num_seqs × Hk``, which is what v2 already uses (and what
# lets v2 run flat at ~0.15 ms across KV lengths on MI300X).
# ---------------------------------------------------------------------------


@triton.jit
def kernel_tq_unified_attention_3d(
    segm_output_ptr,  # [num_tokens, Hq, NUM_SEGMENTS_PER_SEQ, HEAD_SIZE_PADDED] fp32
    segm_max_ptr,  # [num_tokens, Hq, NUM_SEGMENTS_PER_SEQ] fp32
    segm_expsum_ptr,  # [num_tokens, Hq, NUM_SEGMENTS_PER_SEQ] fp32
    query_ptr,  # [num_tokens, Hq, D] — raw if FUSE_Q_ROT else rotated
    KV_cache_ptr,  # [num_blocks, block_size, Hk, padded_slot] uint8
    KV_cache_u16_ptr,  # uint16 view of same storage — Opt#2 wide metadata loads
    Centroids_ptr,
    Pair_lut_ptr,
    PiT_ptr,  # [D, D] fp32 — only dereferenced when FUSE_Q_ROT == 1
    block_tables_ptr,
    seq_lens_ptr,
    query_start_len_ptr,
    sinks_ptr,  # [Hq] fp32 — per-head sink logits; dereferenced only when USE_SINKS
    scale,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    stride_cache_block: tl.int64,
    pit_stride_0: tl.int64,
    pit_stride_1: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    num_seqs: tl.int32,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    # TQ layout constants (same as 2D)
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    KEY_FP8: tl.constexpr,
    USE_PAIR_LUT: tl.constexpr,
    # Opt#3 SoA layout constants
    NUM_KV_HEADS: tl.constexpr,
    KEY_DATA_BYTES: tl.constexpr,
    META_REGION_OFFSET: tl.constexpr,
    NUM_SOA_FIELDS: tl.constexpr,
    SOA_K_NORM: tl.constexpr,
    SOA_V_SCALE: tl.constexpr,
    SOA_V_ZERO: tl.constexpr,
    NORM_CORRECTION: tl.constexpr = 0,
    FP8_E4B15: tl.constexpr = 0,
    FUSE_Q_ROT: tl.constexpr = 0,  # Fused Q @ PiT prologue when 1 (MSE path)
    USE_SINKS: tl.constexpr = 0,  # Per-head sink logit folded into softmax denom
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    seq_idx = _find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    tiles_per_segment = tl.cdiv(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)

    # Segment empty for this sequence — skip.
    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    if FUSE_Q_ROT:
        Q = _tq_fuse_q_rotation(
            Q,
            PiT_ptr,
            pit_stride_0,
            pit_stride_1,
            dim_mask,
            HEAD_SIZE_PADDED,
        )

    block_table_offset = seq_idx * block_table_stride

    # Online softmax state. For the split-KV (3D) kernel we must fold the
    # sink logit into *only* segment 0's (M, L) so that the stage-2 reducer
    # — which is a standard online-softmax merge — counts the sink exactly
    # once. Segments with segm_idx > 0 keep the plain -inf / 1.0 init and
    # therefore carry no sink contribution; stage-2 then combines them.
    # This mirrors the v1 sink pattern from PR #40663 (sid==0 init trick).
    if USE_SINKS and segm_idx == 0:
        M = tl.load(
            sinks_ptr + query_offset_1, mask=query_mask_1, other=float("-inf")
        ).to(tl.float32)
    else:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    context_len = seq_len - cur_batch_query_len
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    num_tiles = tl.cdiv(max_seq_prefix_len, TILE_SIZE)

    # Segment bounds: clip to the causal prefix range.
    tile_lo = segm_idx * tiles_per_segment
    tile_hi = tl.minimum((segm_idx + 1) * tiles_per_segment, num_tiles)

    for j in range(tile_lo, tile_hi):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        # Opt#3 SoA addressing (see 2D kernel for the layout invariants).
        slot_within_block = (seq_offset % BLOCK_SIZE).to(tl.int64)
        block_base = physical_block_idx * stride_cache_block
        DATA_BYTES_PER_SLOT: tl.constexpr = KEY_DATA_BYTES + VAL_DATA_BYTES
        data_bases = (
            block_base
            + slot_within_block * (NUM_KV_HEADS * DATA_BYTES_PER_SLOT)
            + tl.cast(kv_head_idx, tl.int64) * DATA_BYTES_PER_SLOT
        )
        val_bases = data_bases + KEY_DATA_BYTES

        head_meta_u16_base = (block_base + META_REGION_OFFSET) // 2 + tl.cast(
            kv_head_idx, tl.int64
        ) * (NUM_SOA_FIELDS * BLOCK_SIZE)
        knorm_u16_addrs = (
            head_meta_u16_base + SOA_K_NORM * BLOCK_SIZE + slot_within_block
        )
        vscale_u16_addrs = (
            head_meta_u16_base + SOA_V_SCALE * BLOCK_SIZE + slot_within_block
        )
        vzero_u16_addrs = (
            head_meta_u16_base + SOA_V_ZERO * BLOCK_SIZE + slot_within_block
        )

        K_T = _tq_load_k_tile(
            KV_cache_ptr,
            KV_cache_u16_ptr,
            data_bases,
            knorm_u16_addrs,
            offs_d,
            dim_mask,
            tile_mask,
            Centroids_ptr,
            Pair_lut_ptr,
            OUT_DTYPE=Q.dtype,
            HEAD_DIM=HEAD_SIZE,
            BLOCK_D=HEAD_SIZE_PADDED,
            MSE_BITS=MSE_BITS,
            N_CENTROIDS=N_CENTROIDS,
            KEY_FP8=KEY_FP8,
            USE_PAIR_LUT=USE_PAIR_LUT,
            NORM_CORRECTION=NORM_CORRECTION,
            FP8_E4B15=FP8_E4B15,
            TILE_SIZE=TILE_SIZE,
        )

        V = _tq_load_v_tile(
            KV_cache_ptr,
            KV_cache_u16_ptr,
            val_bases,
            vscale_u16_addrs,
            vzero_u16_addrs,
            offs_d,
            dim_mask,
            tile_mask,
            OUT_DTYPE=Q.dtype,
            HEAD_DIM=HEAD_SIZE,
            VQB=VQB,
        )

        S = scale * tl.dot(Q, K_T)

        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = seq_offset[None, :] <= query_abs_pos
        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
            S,
            float("-inf"),
        )

        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j

        acc += tl.dot(P.to(V.dtype), V)

    # -------- Write partial (acc, M, L) to scratch; no normalization --------
    # Layout matches baseline reduce_segments so we can reuse it.
    segm_output_offset = (
        query_offset_0[:, None].to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )
    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------


_layout_cache: dict[tuple, dict[str, int]] = {}


def _get_layout(D: int, mse_bits: int, value_quant_bits: int) -> dict[str, int]:
    key = (D, mse_bits, value_quant_bits)
    cfg = _layout_cache.get(key)
    if cfg is None:
        cfg = {
            "mse_bytes": math.ceil(D * mse_bits / 8),
            "val_data_bytes": math.ceil(D * value_quant_bits / 8),
            "BLOCK_D": triton.next_power_of_2(D),
        }
        _layout_cache[key] = cfg
    return cfg


def _get_pair_lut(centroids: torch.Tensor) -> torch.Tensor:
    """Return a fresh pair-LUT for ``centroids`` on each call.

    Previously cached by ``(data_ptr, device)`` — that key is unsafe
    because the CUDA allocator can reuse freed addresses for different
    centroid tensors, yielding a stale (wrong-values) LUT. The LUT is tiny
    (N*N*2 fp32, ~2KB at N=16) so unconditional rebuild is essentially
    free compared to attention work. If this shows up on a profile,
    replace with a hash-of-values fingerprint, not data_ptr.
    """
    return build_pair_lut(centroids)


# PiT in Q's dtype cached per (PiT-id, dtype) to avoid a host cast on every
# launcher call. Keyed by the fp32 PiT's data_ptr so ownership/lifetime is
# tied to the caller's tensor.
_pit_qdtype_cache: dict = {}


def _get_pit_in_query_dtype(PiT: torch.Tensor, qdtype: torch.dtype) -> torch.Tensor:
    key = (PiT.data_ptr(), qdtype)
    cached = _pit_qdtype_cache.get(key)
    if cached is None:
        cached = PiT.to(qdtype).contiguous()
        _pit_qdtype_cache[key] = cached
    return cached


def triton_turboquant_unified_attention(
    query: torch.Tensor,  # [num_tokens, Hq, D] - fp16/bf16
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, padded_slot] uint8
    block_table: torch.Tensor,  # [num_seqs, max_num_blocks] int32
    seq_lens: torch.Tensor,  # [num_seqs] int32
    query_start_loc: torch.Tensor,  # [num_seqs+1] int32
    Pi: torch.Tensor,  # [D, D] fp32
    centroids: torch.Tensor,  # [n_centroids] fp32
    scale: float,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    value_packed_size: int,  # unused; kept for signature parity with v2
    key_fp8: bool = False,
    norm_correction: bool = False,
    PiT: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    tile_size: int | None = None,
    max_query_len: int | None = None,
    max_seq_len: int | None = None,
    num_kv_splits: int | None = None,
    force_2d: bool = False,
    fuse_q_rot: bool = True,
    sinks: torch.Tensor | None = None,  # [Hq] float — per-head sink logits
) -> torch.Tensor:
    """Launch unified TQ attention (v3).

    ``query`` carries *raw* query vectors. For the MSE-key path the query
    has to be rotated by ``PiT`` before it can be multiplied against the
    (already-rotated) stored K. By default (``fuse_q_rot=True``) that
    rotation is done inside the attention kernel prologue as a single
    small MFMA — no extra dispatch, no HBM round-trip. Setting
    ``fuse_q_rot=False`` restores the original launcher path (fp32 rocBLAS
    GEMM + casts + .contiguous()), which is kept as an A/B toggle for
    bench harnesses; numerical results match the fused path to within a
    few ulp of fp32 round-off. The FP8-key path never rotates Q regardless
    of this flag.

    ``tile_size`` defaults to ``32`` for prefill (``max_query_len > 1``) and
    ``16`` for pure decode (``max_query_len == 1``). Callers may override.

    Dispatch rule:
      * Prefill / chunked-prefill (any query block with ``BLOCK_Q > 1``) and
        ``num_tokens > num_seqs``: 2D kernel (plenty of CTAs already).
      * Pure decode with long KV: 3D split-KV kernel + ``reduce_segments``
        for extra parallelism; this is the same pattern v2 uses.
      * ``force_2d=True``: force the 2D path regardless (used by the
        apples-to-apples bench to isolate pure TQ dequant overhead).

    ``num_kv_splits`` controls the 3D-split segment count (default ``16``).

    Returns ``output`` of shape ``[num_tokens, Hq, D]`` in ``query.dtype``.
    """
    assert query.dim() == 3, f"query must be [N, Hq, D], got {query.shape}"
    num_tokens, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    kv_group_size = Hq // Hk
    num_seqs = int(query_start_loc.shape[0] - 1)
    device = query.device

    cfg = _get_layout(D, mse_bits, value_quant_bits)
    _ = value_packed_size  # unused

    # Q-rotation strategy:
    #   * FP8-key path: no rotation at all (keys stored as fp8, no codebook).
    #   * MSE-key path + fuse_q_rot=True (default): pass the raw query to the
    #     kernel together with PiT and let the kernel's prologue do one
    #     fp32 tl.dot(Q, PiT). Eliminates the launcher-side
    #     (cast, rocBLAS GEMM, cast, contiguous) chain (~50-60 us/step at
    #     D=64, all launch-bound — see bottleneck report §10).
    #   * MSE-key path + fuse_q_rot=False: legacy launcher rotation. Kept
    #     as an A/B toggle for bench harnesses.
    #
    # PiT is always materialized as fp32 D x D contiguous; the kernel loads
    # it as fp32 (cheap, 16KB at D=64) and computes Q @ PiT with IEEE fp32
    # accumulate via tl.dot.
    if key_fp8:
        q_rot = query.contiguous()
        apply_fuse_q_rot = False
    else:
        if PiT is None:
            PiT = Pi.T.contiguous()
        apply_fuse_q_rot = bool(fuse_q_rot)
        if apply_fuse_q_rot:
            q_rot = query.contiguous()
        else:
            q_rot = (query.float() @ PiT).to(query.dtype).contiguous()

    # PiT in fp32, contiguous. For the fused path this is what the kernel
    # loads; for the legacy path it's passed through as a harmless tensor
    # (never dereferenced under the FUSE_Q_ROT constexpr guard). On FP8
    # path Pi may be unused at the call site, so we fall back to an
    # arbitrary non-null tensor (reuse centroids) to satisfy Triton's
    # non-null pointer requirement.
    if (not key_fp8) and PiT is not None:
        PiT_f32 = PiT if PiT.dtype == torch.float32 else PiT.to(torch.float32)
        if not PiT_f32.is_contiguous():
            PiT_f32 = PiT_f32.contiguous()
        pit_stride_0 = PiT_f32.stride(0)
        pit_stride_1 = PiT_f32.stride(1)
    else:
        PiT_f32 = centroids  # harmless dummy; not dereferenced when FUSE_Q_ROT=0
        pit_stride_0 = 0
        pit_stride_1 = 0

    # Sinks: per-head fp32 logits, contiguous. The kernel only dereferences
    # this pointer when USE_SINKS=1, so when the caller passes None we bind
    # a harmless non-null tensor (centroids) to satisfy Triton's non-null
    # pointer requirement. See the sink design notes in the 2D/3D kernel
    # bodies and the v1 precedent from PR #40663 (sid==0 init trick).
    if sinks is not None:
        sinks_f32 = sinks if sinks.dtype == torch.float32 else sinks.to(torch.float32)
        if not sinks_f32.is_contiguous():
            sinks_f32 = sinks_f32.contiguous()
        assert sinks_f32.numel() == Hq, (
            f"sinks must have shape [Hq={Hq}], got numel={sinks_f32.numel()}"
        )
        use_sinks = True
    else:
        sinks_f32 = centroids  # harmless dummy; not dereferenced when USE_SINKS=0
        use_sinks = False

    if output is None:
        output = torch.empty_like(query)

    # BLOCK_M heuristic (TQ-specific; diverges from stock unified_attention).
    #
    # Stock fp16 picks BLOCK_M=16 for small GQA because fp16 is ALU-saturated
    # at any tile size and smaller BLOCK_M = better occupancy. TQ is a
    # DIFFERENT story: the dequant chain (packed-byte loads + centroid gather
    # + pair-LUT + norm/scale bitcasts) has big per-block fixed overhead, and
    # the pair-LUT gather pattern is latency-bound (NOT HBM-bandwidth-bound,
    # confirmed via rocprof: MemUnitStalled=0.01%, VALUBusy=56% at BM=16).
    #
    # For prefill/chunked (max_query_len > 1) we want BLOCK_M as large as
    # possible to (a) amortize the fixed dequant overhead over more query
    # tokens per CTA and (b) let MFMA saturate with a 128x32 tile shape.
    # Empirical sweep on MI300X across gpt-oss (D=64) and llama (D=128):
    #     prefill B=1 Q=4k     : BM=128 is 3.4x faster than BM=16
    #     chunked B=64 Q=1k C=8: BM=128 is 5.6x faster than BM=16
    # With BM=128 v3 matches or BEATS the fp16 baseline on prefill/chunked.
    #
    # For pure decode (max_query_len == 1) BLOCK_M stays small because
    # BLOCK_Q > 1 just pads rows — each sequence contributes at most 1 query
    # token and larger BLOCK_M wastes lanes on masked-out rows.
    if max_query_len is not None:
        is_prefill_like = max_query_len > 1
    else:
        is_prefill_like = num_tokens > num_seqs

    if is_prefill_like:
        # Prefill / chunked: use BLOCK_M=128 (rounded up to a multiple of
        # kv_group_size). Each CTA does very large work (BLOCK_M x TILE_SIZE
        # x KV_len MMA + dequant), so even small grids (~17 CTAs) saturate
        # MI300X just fine. We verified empirically: BM=128 wins across all
        # prefill/chunked sweep points, including small (B=1 Q=256 C=8k) and
        # large (B=64 Q=1k C=8k; 5.6x speedup vs BM=16). BLOCK_M must be a
        # multiple of kv_group_size so BLOCK_Q is an integer.
        BLOCK_M = max(128, triton.next_power_of_2(kv_group_size))
    else:
        # Decode: BLOCK_Q > 1 in decode just pads rows (at most 1 query
        # token per sequence), so there's little to amortize. Inherit
        # stock's small-BLOCK_M heuristic.
        BLOCK_M = 16 if kv_group_size <= 16 else triton.next_power_of_2(kv_group_size)
    BLOCK_Q = BLOCK_M // kv_group_size

    # Grid: at most ceil(N / BLOCK_Q) + num_seqs q-blocks total, like unified.
    total_num_q_blocks = num_tokens // BLOCK_Q + num_seqs

    # TILE_SIZE heuristic (matches stock unified_attention's _get_tile_size):
    #   prefill (max_query_len > 1): 32
    #   decode  (max_query_len == 1): 16
    if tile_size is None:
        tile_size = 32 if is_prefill_like else 16

    # Pair-LUT fast path for 4-bit MSE keys. Skipped for FP8 and non-4-bit.
    # When USE_PAIR_LUT==0 the kernel never dereferences pair_lut, but Triton
    # still requires a tensor pointer, so we fall back to reusing `centroids`.
    use_pair_lut = (not key_fp8) and (mse_bits == 4)
    pair_lut = _get_pair_lut(centroids) if use_pair_lut else centroids

    fp8_e4b15 = _use_fp8_e4b15(device.index or 0)
    num_stages = 1 if _is_hip else 2

    # uint16-aliased view of the cache. Under the Opt#3 SoA layout, K-norm /
    # V-scale / V-zero live in a contiguous per-block metadata region and are
    # always 2-byte aligned, so single-instruction u16 loads replace the
    # original 2× uint8 + OR sequence for every per-token metadata fetch.
    kv_cache_u16 = kv_cache.view(torch.uint16)

    # Opt#3 SoA layout constants (derived locally; matches the store-side
    # computation so the launcher signature stays unchanged). Invariant:
    # data_bytes_per_slot + meta_bytes_per_slot == slot_size_aligned.
    mse_bytes = cfg["mse_bytes"]
    val_data_bytes = cfg["val_data_bytes"]
    key_data_bytes = D if key_fp8 else mse_bytes
    data_bytes_per_slot = key_data_bytes + val_data_bytes
    meta_region_offset = block_size * Hk * data_bytes_per_slot
    num_soa_fields = 2 if key_fp8 else 3
    soa_k_norm = 0  # unused for FP8; harmless constant
    soa_v_scale = 0 if key_fp8 else 1
    soa_v_zero = 1 if key_fp8 else 2

    # ------------------------------------------------------------------
    # Dispatch: 2D for prefill / chunked; 3D split-KV for pure decode with
    # long KV. Falls back to 2D if the sequences are short or if the caller
    # forces it. ``is_prefill_like`` was already computed above for the
    # BLOCK_M / tile_size heuristic; reuse it here.
    # ------------------------------------------------------------------

    # Use 3D only for pure decode where the KV history is long enough that
    # splitting meaningfully increases SM occupancy. Threshold is empirically
    # tuned on MI300X:
    #   seq <  1024 -> 2D path (short KV, reduce overhead > parallelism gain)
    #   seq >= 1024 -> 3D path (v3/v2 ratio 1.04-1.27x across shapes)
    # Below 1024 KV tokens the reduce_segments + scratch-alloc overhead
    # exceeds the parallelism gain from splitting. Above it, the single-CTA
    # 2D path is serial over the KV history and starves MI300X's 304 CUs.
    #
    # IMPORTANT: max_seq_len is expected to come from the caller (backends
    # track it from the block table). We avoid calling seq_lens.max().item()
    # here because that forces a GPU->CPU sync on every launcher call, which
    # adds ~150us of overhead to pure-decode steps.
    if max_seq_len is None:
        # Fallback: infer from block_table (host-side, no sync). This is an
        # upper bound, sufficient for the dispatch decision.
        max_seq_len_hint = int(block_table.shape[1]) * int(block_size)
    else:
        max_seq_len_hint = int(max_seq_len)
    use_3d = (not force_2d) and (not is_prefill_like) and max_seq_len_hint >= 1024

    if not use_3d:
        kernel_tq_unified_attention_2d[(total_num_q_blocks, Hk)](
            output_ptr=output,
            query_ptr=q_rot,
            KV_cache_ptr=kv_cache,
            KV_cache_u16_ptr=kv_cache_u16,
            Centroids_ptr=centroids,
            Pair_lut_ptr=pair_lut,
            PiT_ptr=PiT_f32,
            block_tables_ptr=block_table,
            seq_lens_ptr=seq_lens,
            query_start_len_ptr=query_start_loc,
            sinks_ptr=sinks_f32,
            scale=scale,
            num_query_heads=Hq,
            num_queries_per_kv=kv_group_size,
            block_table_stride=block_table.stride(0),
            query_stride_0=q_rot.stride(0),
            query_stride_1=q_rot.stride(1),
            output_stride_0=output.stride(0),
            output_stride_1=output.stride(1),
            stride_cache_block=kv_cache.stride(0),
            pit_stride_0=pit_stride_0,
            pit_stride_1=pit_stride_1,
            BLOCK_SIZE=block_size,
            TILE_SIZE=tile_size,
            HEAD_SIZE=D,
            HEAD_SIZE_PADDED=cfg["BLOCK_D"],
            BLOCK_Q=BLOCK_Q,
            BLOCK_M=BLOCK_M,
            num_seqs=num_seqs,
            MSE_BITS=mse_bits,
            MSE_BYTES=mse_bytes,
            VQB=value_quant_bits,
            VAL_DATA_BYTES=val_data_bytes,
            N_CENTROIDS=int(centroids.numel()),
            KEY_FP8=1 if key_fp8 else 0,
            USE_PAIR_LUT=1 if use_pair_lut else 0,
            NUM_KV_HEADS=Hk,
            KEY_DATA_BYTES=key_data_bytes,
            META_REGION_OFFSET=meta_region_offset,
            NUM_SOA_FIELDS=num_soa_fields,
            SOA_K_NORM=soa_k_norm,
            SOA_V_SCALE=soa_v_scale,
            SOA_V_ZERO=soa_v_zero,
            NORM_CORRECTION=1 if norm_correction else 0,
            FP8_E4B15=fp8_e4b15,
            FUSE_Q_ROT=1 if apply_fuse_q_rot else 0,
            USE_SINKS=1 if use_sinks else 0,
            num_warps=4,
            num_stages=num_stages,
        )
        return output

    # ---------------- 3D split-KV path ----------------
    # Pick segment count. 16 is a good default for MI300X (304 CUs): at
    # Hk=8, num_seqs=1, total CTAs = 1 * 8 * 16 = 128, which saturates when
    # combined with warp-level parallelism. Capped at ceil(max_seq_len /
    # TILE_SIZE) so we don't launch empty segments.
    if num_kv_splits is None:
        num_kv_splits = 16
    max_possible_splits = max(1, (max_seq_len_hint + tile_size - 1) // tile_size)
    num_segments = max(1, min(num_kv_splits, max_possible_splits))

    HEAD_SIZE_PADDED = cfg["BLOCK_D"]
    # Scratch buffers for segment partials. fp32 to match the kernel's
    # accumulator dtype. Allocated per-call for now; a production backend
    # would hoist these to persistent buffers reused across layers.
    segm_output = torch.empty(
        (num_tokens, Hq, num_segments, HEAD_SIZE_PADDED),
        dtype=torch.float32,
        device=device,
    )
    segm_max = torch.empty(
        (num_tokens, Hq, num_segments), dtype=torch.float32, device=device
    )
    segm_expsum = torch.empty(
        (num_tokens, Hq, num_segments), dtype=torch.float32, device=device
    )

    kernel_tq_unified_attention_3d[(total_num_q_blocks, Hk, num_segments)](
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        query_ptr=q_rot,
        KV_cache_ptr=kv_cache,
        KV_cache_u16_ptr=kv_cache_u16,
        Centroids_ptr=centroids,
        Pair_lut_ptr=pair_lut,
        PiT_ptr=PiT_f32,
        block_tables_ptr=block_table,
        seq_lens_ptr=seq_lens,
        query_start_len_ptr=query_start_loc,
        sinks_ptr=sinks_f32,
        scale=scale,
        num_query_heads=Hq,
        num_queries_per_kv=kv_group_size,
        block_table_stride=block_table.stride(0),
        query_stride_0=q_rot.stride(0),
        query_stride_1=q_rot.stride(1),
        stride_cache_block=kv_cache.stride(0),
        pit_stride_0=pit_stride_0,
        pit_stride_1=pit_stride_1,
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE=D,
        HEAD_SIZE_PADDED=HEAD_SIZE_PADDED,
        BLOCK_Q=BLOCK_Q,
        BLOCK_M=BLOCK_M,
        num_seqs=num_seqs,
        NUM_SEGMENTS_PER_SEQ=num_segments,
        MSE_BITS=mse_bits,
        MSE_BYTES=mse_bytes,
        VQB=value_quant_bits,
        VAL_DATA_BYTES=val_data_bytes,
        N_CENTROIDS=int(centroids.numel()),
        KEY_FP8=1 if key_fp8 else 0,
        USE_PAIR_LUT=1 if use_pair_lut else 0,
        NUM_KV_HEADS=Hk,
        KEY_DATA_BYTES=key_data_bytes,
        META_REGION_OFFSET=meta_region_offset,
        NUM_SOA_FIELDS=num_soa_fields,
        SOA_K_NORM=soa_k_norm,
        SOA_V_SCALE=soa_v_scale,
        SOA_V_ZERO=soa_v_zero,
        NORM_CORRECTION=1 if norm_correction else 0,
        FP8_E4B15=fp8_e4b15,
        FUSE_Q_ROT=1 if apply_fuse_q_rot else 0,
        USE_SINKS=1 if use_sinks else 0,
        num_warps=4,
        num_stages=num_stages,
    )

    # Reduce: merge num_segments partials per (q_token, q_head) into the
    # final output via online-softmax merge. Reusing the baseline kernel
    # (KV-format-agnostic).
    reduce_segments[(num_tokens, Hq)](
        output_ptr=output,
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        seq_lens_ptr=seq_lens,
        num_seqs=num_seqs,
        num_query_heads=Hq,
        out_scale_inv=1.0,
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        block_table_stride=block_table.stride(0),
        TILE_SIZE=tile_size,
        HEAD_SIZE=D,
        HEAD_SIZE_PADDED=HEAD_SIZE_PADDED,
        query_start_len_ptr=query_start_loc,
        BLOCK_Q=BLOCK_Q,
        NUM_SEGMENTS_PER_SEQ=num_segments,
        USE_FP8=False,
    )
    return output


# ---------------------------------------------------------------------------
# Thin decode adapter: lets callers who currently use v1/v2's decode launcher
# signature switch to v3 without rebuilding query_start_loc themselves.
# ---------------------------------------------------------------------------


def triton_turboquant_decode_attention_v3(
    query: torch.Tensor,  # [B, Hq, D]
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    Pi: torch.Tensor,
    centroids: torch.Tensor,
    scale: float,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    value_packed_size: int,
    key_fp8: bool = False,
    norm_correction: bool = False,
    PiT: torch.Tensor | None = None,
    # kept for signature parity with v1/v2 decode launchers; unused here.
    max_seq_len: int = 0,
    mid_o_buf: Any = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: Any = None,
    buf_holder: Any = None,
    max_num_kv_splits: int = 32,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode-only convenience wrapper around ``triton_turboquant_unified_attention``.

    Treats a rank-3 ``[B, Hq, D]`` query as one token per request (query_len = 1)
    and synthesizes a ``query_start_loc`` of ``[0, 1, 2, ..., B]``.
    ``max_num_kv_splits`` is forwarded to the unified launcher as the 3D
    split-KV segment count (capped per-call against ``max_seq_len``).
    ``sinks`` (optional ``[Hq]`` fp32) are forwarded to the kernel which
    folds them into the softmax denominator via the init-time trick.
    """
    del mid_o_buf, lse_buf, buf_holder
    B = query.shape[0]
    cu_seqlens_q = torch.arange(B + 1, device=query.device, dtype=seq_lens.dtype)

    out = triton_turboquant_unified_attention(
        query=query.contiguous(),
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        query_start_loc=cu_seqlens_q,
        Pi=Pi,
        centroids=centroids,
        scale=scale,
        mse_bits=mse_bits,
        key_packed_size=key_packed_size,
        value_quant_bits=value_quant_bits,
        value_packed_size=value_packed_size,
        key_fp8=key_fp8,
        norm_correction=norm_correction,
        PiT=PiT,
        output=output_buf[:B] if output_buf is not None else None,
        max_query_len=1,
        max_seq_len=max_seq_len if max_seq_len > 0 else None,
        num_kv_splits=max_num_kv_splits,
        sinks=sinks,
    )
    return out
