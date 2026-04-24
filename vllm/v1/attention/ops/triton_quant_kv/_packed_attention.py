# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention (read-path) kernel + launcher for the sub-byte packed modes.

A single :func:`_attn_packed` kernel handles both INT4
(``PACKING_FACTOR=2``) and INT2 (``PACKING_FACTOR=4``) — the constexpr
gates all mode-specific branches and Triton only traces the taken one,
so each concrete launch compiles to the same code a bespoke per-mode
kernel would.  Mode-specific pieces handled by the branch:

* How many Q/KV streams are split (2 for INT4, 4 for INT2).
* KV dequantization (nibble unpack vs Lloyd-Max centroid lookup).
* Score correction (asymmetric zero-point subtraction vs plain).

Everything else (prologue, masking, online softmax, tile loop, 2D/3D
epilogue) is shared.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_attention_helpers import (
    apply_alibi_to_score,
    apply_softcap,
    cdiv_fn,
    compute_kv_seq_mask,
    compute_tile_loop_bounds,
    init_softmax_M,
    load_qq_bias_tile,
    resolve_seq_and_query_len,
    softmax_step,
    store_segm_reduce_scalars,
)
from vllm.v1.attention.ops.triton_quant_kv._pack_unpack import (
    unpack_int2_quartet,
    unpack_int4_nibbles,
)
from vllm.v1.attention.ops.triton_quant_kv._packed_reshape import _lloyd_max_dequant_4
from vllm.v1.attention.ops.triton_unified_attention import reduce_segments

float8_info = torch.finfo(current_platform.fp8_dtype())


@triton.jit
def _attn_packed(
    # Output destinations.  In 2D mode the final result is written into
    # ``output_ptr``; in 3D mode per-segment partials go into the three
    # ``segm_*`` tensors and ``output_ptr`` is unused.
    output_ptr,
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    sink_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    alibi_slopes_ptr,
    qq_bias_ptr,
    scale,
    out_scale,
    softcap,
    k_scale_cache_ptr,
    v_scale_cache_ptr,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    qq_bias_stride_0: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    PACKED_HEAD_PADDED: tl.constexpr,  # HEAD_SIZE / PACKING_FACTOR, rounded up
    USE_ALIBI_SLOPES: tl.constexpr,
    USE_ALIBI_SQRT: tl.constexpr,
    USE_QQ_BIAS: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    USE_SINKS: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    USE_MM_PREFIX: tl.constexpr,
    MAX_MM_RANGES: tl.constexpr,
    mm_prefix_range_ptr,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.constexpr,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.constexpr,
    stride_ks_blk: tl.int64,
    stride_ks_slot: tl.int64,
    stride_ks_head: tl.int64,
    stride_vs_blk: tl.int64,
    stride_vs_slot: tl.int64,
    stride_vs_head: tl.int64,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    USE_FP8: tl.constexpr,
    IS_3D: tl.constexpr,
    # 2 → INT4 nibble pair (asymmetric + zp); 4 → INT2 quartet
    # (Lloyd-Max centroids).  All mode-specific branches below gate on
    # this value.
    PACKING_FACTOR: tl.constexpr,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    # -----------------------------------------------------------------
    # Shared prologue: sequence lookup, q-block bounds, early returns.
    # -----------------------------------------------------------------
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2) if IS_3D else 0

    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    if IS_3D:
        tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
        if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
            return
    else:
        tiles_per_segment = 0

    offs_m = tl.arange(0, BLOCK_M)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv

    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # -----------------------------------------------------------------
    # Split-Q prologue: PACKING_FACTOR interleaved streams of Q.
    # INT4 uses 2 streams (even / odd); INT2 uses 4.  The packed KV
    # cache stores one byte per ``packed_offs``, which holds
    # PACKING_FACTOR values.
    # -----------------------------------------------------------------
    packed_offs = tl.arange(0, PACKED_HEAD_PADDED)
    offs_s0 = packed_offs * PACKING_FACTOR
    offs_s1 = packed_offs * PACKING_FACTOR + 1
    mask_s0 = tl.where(offs_s0 < HEAD_SIZE, 1, 0).to(tl.int1)
    mask_s1 = tl.where(offs_s1 < HEAD_SIZE, 1, 0).to(tl.int1)
    packed_dim_mask = tl.where(packed_offs < HEAD_SIZE // PACKING_FACTOR, 1, 0).to(
        tl.int1
    )
    q_base = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
    )
    q_mask = query_mask_0[:, None] & query_mask_1[:, None]
    Q_s0 = tl.load(
        query_ptr + q_base + offs_s0[None, :],
        mask=mask_s0[None, :] & q_mask,
        other=0.0,
    ).to(tl.float32)
    Q_s1 = tl.load(
        query_ptr + q_base + offs_s1[None, :],
        mask=mask_s1[None, :] & q_mask,
        other=0.0,
    ).to(tl.float32)
    if PACKING_FACTOR == 4:
        offs_s2 = packed_offs * 4 + 2
        offs_s3 = packed_offs * 4 + 3
        mask_s2 = tl.where(offs_s2 < HEAD_SIZE, 1, 0).to(tl.int1)
        mask_s3 = tl.where(offs_s3 < HEAD_SIZE, 1, 0).to(tl.int1)
        Q_s2 = tl.load(
            query_ptr + q_base + offs_s2[None, :],
            mask=mask_s2[None, :] & q_mask,
            other=0.0,
        ).to(tl.float32)
        Q_s3 = tl.load(
            query_ptr + q_base + offs_s3[None, :],
            mask=mask_s3[None, :] & q_mask,
            other=0.0,
        ).to(tl.float32)

    # INT4 asymmetric correction needs sum(Q) per row.
    if PACKING_FACTOR == 2:
        Q_sum = tl.sum(Q_s0, axis=1) + tl.sum(Q_s1, axis=1)

    block_table_offset = seq_idx * block_table_stride

    # -----------------------------------------------------------------
    # Online-softmax state + optional feature loads.
    # -----------------------------------------------------------------
    M = init_softmax_M(
        sink_ptr, query_offset_1, query_mask_1, segm_idx, BLOCK_M, USE_SINKS, IS_3D
    )
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc_s0 = tl.zeros([BLOCK_M, PACKED_HEAD_PADDED], dtype=tl.float32)
    acc_s1 = tl.zeros([BLOCK_M, PACKED_HEAD_PADDED], dtype=tl.float32)
    if PACKING_FACTOR == 4:
        acc_s2 = tl.zeros([BLOCK_M, PACKED_HEAD_PADDED], dtype=tl.float32)
        acc_s3 = tl.zeros([BLOCK_M, PACKED_HEAD_PADDED], dtype=tl.float32)

    context_len = seq_len - cur_batch_query_len

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    if USE_QQ_BIAS:
        qq_bias_row_ptrs = qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0

    loop_lo, loop_hi, max_seq_prefix_len = compute_tile_loop_bounds(
        context_len,
        seq_len,
        cur_batch_query_len,
        q_block_local_idx,
        segm_idx,
        tiles_per_segment,
        TILE_SIZE,
        BLOCK_M,
        BLOCK_Q,
        num_queries_per_kv,
        SLIDING_WINDOW,
        USE_MM_PREFIX,
        IS_3D,
    )

    # -----------------------------------------------------------------
    # Tile loop.  Per-tile: load packed KV + scales, dequantize into
    # PACKING_FACTOR streams, compute the split dot, run the shared
    # softmax step, accumulate per stream.
    # -----------------------------------------------------------------
    for j in range(loop_lo, loop_hi):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        slot_in_blk = seq_offset % BLOCK_SIZE
        k_off = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + packed_offs[:, None] * stride_k_cache_3
            + slot_in_blk[None, :] * stride_k_cache_1
        )
        K_packed = tl.load(
            key_cache_ptr + k_off,
            mask=packed_dim_mask[:, None] & tile_mask[None, :],
            other=0,
        )
        v_off = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + packed_offs[None, :] * stride_v_cache_3
            + slot_in_blk[:, None] * stride_v_cache_1
        )
        V_packed = tl.load(
            value_cache_ptr + v_off,
            mask=packed_dim_mask[None, :] & tile_mask[:, None],
            other=0,
        )
        # Dequantize KV.  INT4 unpacks nibbles as plain uint [0..15];
        # the zero-point is applied on the score side.  INT2 unpacks
        # quartet indices and looks up Lloyd-Max centroids (N(0, 1)).
        if PACKING_FACTOR == 2:
            K_s0_u, K_s1_u = unpack_int4_nibbles(K_packed)
            K_s0 = K_s0_u.to(tl.float32)
            K_s1 = K_s1_u.to(tl.float32)
            V_s0_u, V_s1_u = unpack_int4_nibbles(V_packed)
            V_s0 = V_s0_u.to(tl.float32)
            V_s1 = V_s1_u.to(tl.float32)
        else:
            kc0_u, kc1_u, kc2_u, kc3_u = unpack_int2_quartet(K_packed)
            K_s0 = _lloyd_max_dequant_4(kc0_u).to(tl.float32)
            K_s1 = _lloyd_max_dequant_4(kc1_u).to(tl.float32)
            K_s2 = _lloyd_max_dequant_4(kc2_u).to(tl.float32)
            K_s3 = _lloyd_max_dequant_4(kc3_u).to(tl.float32)
            vc0_u, vc1_u, vc2_u, vc3_u = unpack_int2_quartet(V_packed)
            V_s0 = _lloyd_max_dequant_4(vc0_u).to(tl.float32)
            V_s1 = _lloyd_max_dequant_4(vc1_u).to(tl.float32)
            V_s2 = _lloyd_max_dequant_4(vc2_u).to(tl.float32)
            V_s3 = _lloyd_max_dequant_4(vc3_u).to(tl.float32)

        ks_idx = (
            physical_block_idx * stride_ks_blk
            + slot_in_blk * stride_ks_slot
            + kv_head_idx * stride_ks_head
        )
        ks_raw = tl.load(k_scale_cache_ptr + ks_idx, mask=tile_mask, other=0)
        vs_idx = (
            physical_block_idx * stride_vs_blk
            + slot_in_blk * stride_vs_slot
            + kv_head_idx * stride_vs_head
        )
        vs_raw = tl.load(v_scale_cache_ptr + vs_idx, mask=tile_mask, other=0)

        # INT4 steganographs the 4-bit zero-point in the low 4 bits of
        # the float32 scale's mantissa.  INT2 stores a plain scale.
        if PACKING_FACTOR == 2:
            ks_bits = ks_raw.to(tl.int32, bitcast=True)
            k_zp = (ks_bits & 0xF).to(tl.float32)
            k_token_head_scales = (ks_bits & -16).to(tl.float32, bitcast=True)
            vs_bits = vs_raw.to(tl.int32, bitcast=True)
            v_zp = (vs_bits & 0xF).to(tl.float32)
            v_token_head_scales = (vs_bits & -16).to(tl.float32, bitcast=True)
        else:
            k_token_head_scales = ks_raw
            v_token_head_scales = vs_raw

        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = compute_kv_seq_mask(
            query_abs_pos,
            seq_offset,
            seq_idx,
            mm_prefix_range_ptr,
            SLIDING_WINDOW,
            USE_MM_PREFIX,
            MAX_MM_RANGES,
        )

        # Score: split-dot across PACKING_FACTOR streams; fused
        # softmax_scale * per-(token, head) k_scale in one mul.  INT4
        # subtracts the ``zp * sum(Q)`` correction term; INT2 doesn't.
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        if PACKING_FACTOR == 2:
            raw_dot = tl.dot(Q_s0, K_s0) + tl.dot(Q_s1, K_s1)
            S += (raw_dot - Q_sum[:, None] * k_zp[None, :]) * (
                scale * k_token_head_scales[None, :]
            )
        else:
            raw_dot = (
                tl.dot(Q_s0, K_s0)
                + tl.dot(Q_s1, K_s1)
                + tl.dot(Q_s2, K_s2)
                + tl.dot(Q_s3, K_s3)
            )
            S += raw_dot * (scale * k_token_head_scales[None, :])

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            S = apply_alibi_to_score(
                S, alibi_slope, seq_offset, context_len, query_pos, USE_ALIBI_SQRT
            )

        if USE_QQ_BIAS:
            S += load_qq_bias_tile(
                qq_bias_row_ptrs, seq_offset, context_len, qq_bias_stride_0
            )

        M, L, P, alpha = softmax_step(S, M, L)
        acc_s0 = acc_s0 * alpha[:, None]
        acc_s1 = acc_s1 * alpha[:, None]
        if PACKING_FACTOR == 4:
            acc_s2 = acc_s2 * alpha[:, None]
            acc_s3 = acc_s3 * alpha[:, None]

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            sw_mask = (context_len + qpos_lo - seq_offset) < SLIDING_WINDOW
            V_s0 = tl.where(sw_mask[:, None], V_s0, 0.0)
            V_s1 = tl.where(sw_mask[:, None], V_s1, 0.0)
            if PACKING_FACTOR == 4:
                V_s2 = tl.where(sw_mask[:, None], V_s2, 0.0)
                V_s3 = tl.where(sw_mask[:, None], V_s3, 0.0)

        # Fuse v per-(token, head) scale into P.  INT4 also subtracts
        # the v-zero-point contribution from each stream once.
        P_v = (P * v_token_head_scales[None, :]).to(tl.float32)
        if PACKING_FACTOR == 2:
            Pv_zp_sum = tl.sum(P_v * v_zp[None, :], axis=1)
            acc_s0 += tl.dot(P_v, V_s0) - Pv_zp_sum[:, None]
            acc_s1 += tl.dot(P_v, V_s1) - Pv_zp_sum[:, None]
        else:
            acc_s0 += tl.dot(P_v, V_s0)
            acc_s1 += tl.dot(P_v, V_s1)
            acc_s2 += tl.dot(P_v, V_s2)
            acc_s3 += tl.dot(P_v, V_s3)

    # -----------------------------------------------------------------
    # Epilogue.  2D writes the final output with optional FP8 clamp;
    # 3D writes the per-segment partials (output / max / expsum) for
    # ``reduce_segments`` to finalize.  Each stream writes its own
    # stripe in the output layout.
    # -----------------------------------------------------------------
    out_mask = query_mask_0[:, None] & query_mask_1[:, None]
    if IS_3D:
        segm_base = (
            query_offset_0[:, None].to(tl.int64)
            * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
            + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
            + segm_idx * HEAD_SIZE_PADDED
        )
        tl.store(
            segm_output_ptr + segm_base + offs_s0[None, :],
            acc_s0,
            mask=mask_s0[None, :] & out_mask,
        )
        tl.store(
            segm_output_ptr + segm_base + offs_s1[None, :],
            acc_s1,
            mask=mask_s1[None, :] & out_mask,
        )
        if PACKING_FACTOR == 4:
            tl.store(
                segm_output_ptr + segm_base + offs_s2[None, :],
                acc_s2,
                mask=mask_s2[None, :] & out_mask,
            )
            tl.store(
                segm_output_ptr + segm_base + offs_s3[None, :],
                acc_s3,
                mask=mask_s3[None, :] & out_mask,
            )
        store_segm_reduce_scalars(
            segm_max_ptr,
            segm_expsum_ptr,
            query_offset_0,
            query_offset_1,
            segm_idx,
            M,
            L,
            query_mask_0,
            query_mask_1,
            num_query_heads,
            NUM_SEGMENTS_PER_SEQ,
        )
    else:
        acc_s0 = acc_s0 / L[:, None]
        acc_s1 = acc_s1 / L[:, None]
        if PACKING_FACTOR == 4:
            acc_s2 = acc_s2 / L[:, None]
            acc_s3 = acc_s3 / L[:, None]
        if USE_FP8:
            out_s = tl.load(out_scale)
            acc_s0 = tl.clamp(acc_s0 * out_s, FP8_MIN, FP8_MAX)
            acc_s1 = tl.clamp(acc_s1 * out_s, FP8_MIN, FP8_MAX)
            if PACKING_FACTOR == 4:
                acc_s2 = tl.clamp(acc_s2 * out_s, FP8_MIN, FP8_MAX)
                acc_s3 = tl.clamp(acc_s3 * out_s, FP8_MIN, FP8_MAX)
        out_base = (
            query_offset_0[:, None] * output_stride_0
            + query_offset_1[:, None] * output_stride_1
        )
        tl.store(
            output_ptr + out_base + offs_s0[None, :],
            acc_s0,
            mask=mask_s0[None, :] & out_mask,
        )
        tl.store(
            output_ptr + out_base + offs_s1[None, :],
            acc_s1,
            mask=mask_s1[None, :] & out_mask,
        )
        if PACKING_FACTOR == 4:
            tl.store(
                output_ptr + out_base + offs_s2[None, :],
                acc_s2,
                mask=mask_s2[None, :] & out_mask,
            )
            tl.store(
                output_ptr + out_base + offs_s3[None, :],
                acc_s3,
                mask=mask_s3[None, :] & out_mask,
            )


def _launch_packed_attn(
    *,
    q,
    k_cache,
    v_cache,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    softmax_scale,
    window_size,
    block_table,
    softcap,
    sinks,
    alibi_slopes,
    use_alibi_sqrt,
    qq_bias,
    output_scale,
    mm_prefix_range,
    k_scale_cache,
    v_scale_cache,
    seq_threshold_3D,
    num_par_softmax_segments,
    softmax_segm_output,
    softmax_segm_max,
    softmax_segm_expsum,
    packing_factor: int,
):
    """Launch ``_attn_packed`` for one of the sub-byte modes.

    Handles 2D-vs-3D dispatch, placeholder pointers for the unused side
    of that split, and the trailing ``reduce_segments`` pass.  Writes
    into ``out`` (directly for 2D; via the segm buffers for 3D).
    """
    import vllm.envs as envs
    from vllm.v1.attention.ops.triton_unified_attention import _get_tile_size

    is_batch_invariant = envs.VLLM_BATCH_INVARIANT

    use_mm_prefix = False
    max_mm_ranges = 0
    if mm_prefix_range is not None:
        assert mm_prefix_range.ndim == 3, (
            f"Unsupported mm_prefix_range shape: {mm_prefix_range.shape}"
        )
        use_mm_prefix = True
        max_mm_ranges = mm_prefix_range.shape[1]

    block_size = v_cache.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k_cache.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0
    TILE_SIZE_PREFILL = _get_tile_size(
        head_size, sliding_window_val, q.element_size(), is_prefill=True
    )
    TILE_SIZE_DECODE = _get_tile_size(
        head_size, sliding_window_val, q.element_size(), is_prefill=False
    )

    use_3d = not (
        seq_threshold_3D is None
        or num_par_softmax_segments is None
        or softmax_segm_output is None
        or softmax_segm_max is None
        or softmax_segm_expsum is None
        or max_seqlen_q > 1
        or num_seqs > seq_threshold_3D
        or is_batch_invariant
    )

    # 3D never reads ``output_ptr`` and 2D never reads the segm tensors,
    # but Triton needs a non-null pointer everywhere; reuse ``out`` as
    # the placeholder for the unused side.
    segm_output_ptr = softmax_segm_output if use_3d else out
    segm_max_ptr = softmax_segm_max if use_3d else out
    segm_expsum_ptr = softmax_segm_expsum if use_3d else out
    num_segments = num_par_softmax_segments if use_3d else 1

    grid: tuple[Any, ...]
    if use_3d:
        grid = (total_num_q_blocks, num_kv_heads, num_par_softmax_segments)
        tile_size = TILE_SIZE_DECODE
    else:
        grid = (total_num_q_blocks, num_kv_heads)
        tile_size = TILE_SIZE_PREFILL

    _attn_packed[grid](
        output_ptr=out,
        segm_output_ptr=segm_output_ptr,
        segm_max_ptr=segm_max_ptr,
        segm_expsum_ptr=segm_expsum_ptr,
        query_ptr=q,
        key_cache_ptr=k_cache,
        value_cache_ptr=v_cache,
        sink_ptr=sinks,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=alibi_slopes,
        qq_bias_ptr=qq_bias,
        scale=softmax_scale,
        out_scale=1 / output_scale if output_scale is not None else 1.0,
        softcap=softcap,
        k_scale_cache_ptr=k_scale_cache,
        v_scale_cache_ptr=v_scale_cache,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        qq_bias_stride_0=qq_bias.stride(0) if qq_bias is not None else 0,
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        PACKED_HEAD_PADDED=triton.next_power_of_2(head_size) // packing_factor,
        USE_ALIBI_SLOPES=alibi_slopes is not None,
        USE_ALIBI_SQRT=use_alibi_sqrt,
        USE_QQ_BIAS=qq_bias is not None,
        USE_SOFTCAP=(softcap > 0),
        USE_SINKS=(sinks is not None),
        SLIDING_WINDOW=(1 + window_size[0]),
        USE_MM_PREFIX=use_mm_prefix,
        MAX_MM_RANGES=max_mm_ranges,
        mm_prefix_range_ptr=mm_prefix_range,
        stride_k_cache_0=k_cache.stride(0),
        stride_k_cache_1=k_cache.stride(1),
        stride_k_cache_2=k_cache.stride(2),
        stride_k_cache_3=k_cache.stride(3),
        stride_v_cache_0=v_cache.stride(0),
        stride_v_cache_1=v_cache.stride(1),
        stride_v_cache_2=v_cache.stride(2),
        stride_v_cache_3=v_cache.stride(3),
        stride_ks_blk=k_scale_cache.stride(0),
        stride_ks_slot=k_scale_cache.stride(1),
        stride_ks_head=k_scale_cache.stride(2),
        stride_vs_blk=v_scale_cache.stride(0),
        stride_vs_slot=v_scale_cache.stride(1),
        stride_vs_head=v_scale_cache.stride(2),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        NUM_SEGMENTS_PER_SEQ=num_segments,
        USE_FP8=output_scale is not None,
        IS_3D=use_3d,
        PACKING_FACTOR=packing_factor,
    )

    if use_3d:
        reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            out_scale_inv=1 / output_scale if output_scale is not None else 1.0,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0),
            TILE_SIZE=TILE_SIZE_DECODE,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            USE_FP8=output_scale is not None,
        )
