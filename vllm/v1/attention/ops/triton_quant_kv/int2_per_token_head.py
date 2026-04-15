# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT2 per-token-head KV cache quantization backend.

Format
------
- Pre-rotation: Walsh-Hadamard transform gaussianizes the KV vectors so
  values follow a predictable N(0, σ) distribution.
- Storage: 4 × Lloyd-Max 2-bit indices packed per byte → ``head_size // 4``
  bytes per (token, head).
- Scale: stores ``norm / d^1.5`` so the attention kernel only multiplies
  by the stored scale (the ``1/d`` factor and the de-normalization are
  absorbed together).

Read path
---------
``S += dot(Q_rotated, K_centroids) × scale_per_head`` via 4-way split dot.
Q is rotated by the same Hadamard before the kernel and rotated back
(inverse Hadamard) on the output.
"""

from __future__ import annotations

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_quant_kv import register
from vllm.v1.attention.ops.triton_quant_kv._attn_loop import (
    apply_alibi_to_score,
    compute_kv_seq_mask,
    load_qq_bias_tile,
    softmax_step,
)
from vllm.v1.attention.ops.triton_quant_kv._hadamard import fast_hadamard_transform
from vllm.v1.attention.ops.triton_quant_kv._packed import (
    pack_int2_quartet,
    unpack_int2_quartet,
)
from vllm.v1.attention.ops.triton_quant_kv.base import QuantKVBackend
from vllm.v1.attention.ops.triton_unified_attention import (
    apply_softcap,
    cdiv_fn,
    find_seq_idx,
    reduce_segments,
)
from vllm.v1.kv_cache_interface import KVQuantMode

float8_info = torch.finfo(current_platform.fp8_dtype())


# ---------------------------------------------------------------------------
# Lloyd-Max centroid helpers
# ---------------------------------------------------------------------------
@triton.jit
def _lloyd_max_quantize_4(z):
    """Quantize N(0,1) values to 4 Lloyd-Max centroids (INT2).

    Returns index in [0, 3].  Boundaries: [-0.9816, 0, 0.9816].
    """
    return tl.where(
        z < 0.0,
        tl.where(z < -0.9816, 0, 1).to(tl.uint8),
        tl.where(z < 0.9816, 2, 3).to(tl.uint8),
    )


@triton.jit
def _lloyd_max_dequant_4(idx):
    """Look up INT2 Lloyd-Max centroid for N(0,1).  idx in [0..3]."""
    return tl.where(
        idx < 2,
        tl.where(idx == 0, -1.5104, -0.4528),
        tl.where(idx == 2, 0.4528, 1.5104),
    )


# ---------------------------------------------------------------------------
# Reshape kernel: Hadamard-rotated input → 4 centroids/byte + norm scale
# ---------------------------------------------------------------------------
@triton.jit
def _reshape_cache_int2_kernel(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    k_scale_cache_ptr,
    v_scale_cache_ptr,
    slot_mapping_ptr,
    stride_key_tok: tl.int64,
    stride_key_head: tl.int64,
    stride_val_tok: tl.int64,
    stride_val_head: tl.int64,
    stride_kc_blk: tl.int64,
    stride_kc_slot: tl.int64,
    stride_kc_head: tl.int64,
    stride_vc_blk: tl.int64,
    stride_vc_slot: tl.int64,
    stride_vc_head: tl.int64,
    stride_ks_blk: tl.int64,
    stride_ks_slot: tl.int64,
    stride_ks_head: tl.int64,
    stride_vs_blk: tl.int64,
    stride_vs_slot: tl.int64,
    stride_vs_head: tl.int64,
    block_size: tl.constexpr,
    head_size: tl.constexpr,
    head_size_v: tl.constexpr,
    QUARTER_HEAD_PADDED: tl.constexpr,
):
    """INT2 Hadamard + Lloyd-Max 4-centroid quantization.

    Packs 4 × 2-bit indices per byte → head_size/4 bytes per head.
    """
    tok = tl.program_id(0)
    head = tl.program_id(1)

    slot = tl.load(slot_mapping_ptr + tok).to(tl.int64)
    if slot < 0:
        return

    blk = slot // block_size
    slot_in_blk = slot % block_size

    qtr_offs = tl.arange(0, QUARTER_HEAD_PADDED)
    offs_0 = qtr_offs * 4
    offs_1 = qtr_offs * 4 + 1
    offs_2 = qtr_offs * 4 + 2
    offs_3 = qtr_offs * 4 + 3

    # ---- Key ----------------------------------------------------------------
    qtr_k = head_size // 4
    mask_0k = offs_0 < head_size
    mask_1k = offs_1 < head_size
    mask_2k = offs_2 < head_size
    mask_3k = offs_3 < head_size
    key_base = key_ptr + tok * stride_key_tok + head * stride_key_head

    k0 = tl.load(key_base + offs_0, mask=mask_0k, other=0.0).to(tl.float32)
    k1 = tl.load(key_base + offs_1, mask=mask_1k, other=0.0).to(tl.float32)
    k2 = tl.load(key_base + offs_2, mask=mask_2k, other=0.0).to(tl.float32)
    k3 = tl.load(key_base + offs_3, mask=mask_3k, other=0.0).to(tl.float32)

    k_sq = (
        tl.sum(tl.where(mask_0k, k0 * k0, 0.0))
        + tl.sum(tl.where(mask_1k, k1 * k1, 0.0))
        + tl.sum(tl.where(mask_2k, k2 * k2, 0.0))
        + tl.sum(tl.where(mask_3k, k3 * k3, 0.0))
    )
    k_norm = tl.sqrt(k_sq + 1e-12)

    k_inv_sigma = tl.sqrt(float(head_size)) / k_norm
    k0_z = k0 * k_inv_sigma
    k1_z = k1 * k_inv_sigma
    k2_z = k2 * k_inv_sigma
    k3_z = k3 * k_inv_sigma

    q0 = _lloyd_max_quantize_4(k0_z)
    q1 = _lloyd_max_quantize_4(k1_z)
    q2 = _lloyd_max_quantize_4(k2_z)
    q3 = _lloyd_max_quantize_4(k3_z)

    k_packed = pack_int2_quartet(q0, q1, q2, q3)
    tl.store(
        key_cache_ptr
        + blk * stride_kc_blk
        + slot_in_blk * stride_kc_slot
        + head * stride_kc_head
        + qtr_offs,
        k_packed,
        mask=qtr_offs < qtr_k,
    )

    # Store norm/d^1.5 as scale.  See module docstring for the math.
    k_scale = k_norm / float(head_size**1.5)
    tl.store(
        k_scale_cache_ptr
        + blk * stride_ks_blk
        + slot_in_blk * stride_ks_slot
        + head * stride_ks_head,
        k_scale,
    )

    # ---- Value --------------------------------------------------------------
    qtr_v = head_size_v // 4
    mask_0v = offs_0 < head_size_v
    mask_1v = offs_1 < head_size_v
    mask_2v = offs_2 < head_size_v
    mask_3v = offs_3 < head_size_v
    val_base = value_ptr + tok * stride_val_tok + head * stride_val_head

    v0 = tl.load(val_base + offs_0, mask=mask_0v, other=0.0).to(tl.float32)
    v1 = tl.load(val_base + offs_1, mask=mask_1v, other=0.0).to(tl.float32)
    v2 = tl.load(val_base + offs_2, mask=mask_2v, other=0.0).to(tl.float32)
    v3 = tl.load(val_base + offs_3, mask=mask_3v, other=0.0).to(tl.float32)

    v_sq = (
        tl.sum(tl.where(mask_0v, v0 * v0, 0.0))
        + tl.sum(tl.where(mask_1v, v1 * v1, 0.0))
        + tl.sum(tl.where(mask_2v, v2 * v2, 0.0))
        + tl.sum(tl.where(mask_3v, v3 * v3, 0.0))
    )
    v_norm = tl.sqrt(v_sq + 1e-12)
    v_inv_sigma = tl.sqrt(float(head_size_v)) / v_norm
    v0_z = v0 * v_inv_sigma
    v1_z = v1 * v_inv_sigma
    v2_z = v2 * v_inv_sigma
    v3_z = v3 * v_inv_sigma

    vq0 = _lloyd_max_quantize_4(v0_z)
    vq1 = _lloyd_max_quantize_4(v1_z)
    vq2 = _lloyd_max_quantize_4(v2_z)
    vq3 = _lloyd_max_quantize_4(v3_z)

    v_packed = pack_int2_quartet(vq0, vq1, vq2, vq3)
    tl.store(
        value_cache_ptr
        + blk * stride_vc_blk
        + slot_in_blk * stride_vc_slot
        + head * stride_vc_head
        + qtr_offs,
        v_packed,
        mask=qtr_offs < qtr_v,
    )

    v_scale = v_norm / float(head_size_v**1.5)
    tl.store(
        v_scale_cache_ptr
        + blk * stride_vs_blk
        + slot_in_blk * stride_vs_slot
        + head * stride_vs_head,
        v_scale,
    )


# ---------------------------------------------------------------------------
# Attention kernel: 4-way split-dot, Lloyd-Max centroid lookup.
# Fused 2D/3D via IS_3D constexpr.
# ---------------------------------------------------------------------------
@triton.jit
def _attn_int2(
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
    QUARTER_HEAD_PADDED: tl.constexpr,
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
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    if IS_3D:
        segm_idx = tl.program_id(2)

    seq_idx = find_seq_idx(
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
    if IS_3D:
        tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
        if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
            return

    offs_m = tl.arange(0, BLOCK_M)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv

    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # ---- Split-Q prologue: 4-way load --------------------------------------
    qtr_offs = tl.arange(0, QUARTER_HEAD_PADDED)
    offs_q0 = qtr_offs * 4
    offs_q1 = qtr_offs * 4 + 1
    offs_q2 = qtr_offs * 4 + 2
    offs_q3 = qtr_offs * 4 + 3
    mask_q0 = tl.where(offs_q0 < HEAD_SIZE, 1, 0).to(tl.int1)
    mask_q1 = tl.where(offs_q1 < HEAD_SIZE, 1, 0).to(tl.int1)
    mask_q2 = tl.where(offs_q2 < HEAD_SIZE, 1, 0).to(tl.int1)
    mask_q3 = tl.where(offs_q3 < HEAD_SIZE, 1, 0).to(tl.int1)
    qtr_dim_mask = tl.where(qtr_offs < HEAD_SIZE // 4, 1, 0).to(tl.int1)
    q_base = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
    )
    q_mask = query_mask_0[:, None] & query_mask_1[:, None]
    Q_s0 = tl.load(
        query_ptr + q_base + offs_q0[None, :],
        mask=mask_q0[None, :] & q_mask,
        other=0.0,
    ).to(tl.float32)
    Q_s1 = tl.load(
        query_ptr + q_base + offs_q1[None, :],
        mask=mask_q1[None, :] & q_mask,
        other=0.0,
    ).to(tl.float32)
    Q_s2 = tl.load(
        query_ptr + q_base + offs_q2[None, :],
        mask=mask_q2[None, :] & q_mask,
        other=0.0,
    ).to(tl.float32)
    Q_s3 = tl.load(
        query_ptr + q_base + offs_q3[None, :],
        mask=mask_q3[None, :] & q_mask,
        other=0.0,
    ).to(tl.float32)

    block_table_offset = seq_idx * block_table_stride

    if USE_SINKS:
        load_sinks = (not IS_3D) or (segm_idx == 0)
        if load_sinks:
            M = tl.load(
                sink_ptr + query_offset_1,
                mask=query_mask_1,
                other=float("-inf"),
            ).to(dtype=tl.float32)
        else:
            M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc_s0 = tl.zeros([BLOCK_M, QUARTER_HEAD_PADDED], dtype=tl.float32)
    acc_s1 = tl.zeros([BLOCK_M, QUARTER_HEAD_PADDED], dtype=tl.float32)
    acc_s2 = tl.zeros([BLOCK_M, QUARTER_HEAD_PADDED], dtype=tl.float32)
    acc_s3 = tl.zeros([BLOCK_M, QUARTER_HEAD_PADDED], dtype=tl.float32)

    context_len = seq_len - cur_batch_query_len

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    if USE_QQ_BIAS:
        qq_bias_row_ptrs = qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0

    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )
    if USE_MM_PREFIX:
        max_seq_prefix_len = tl.maximum(max_seq_prefix_len, seq_len)
    else:
        max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    tile_start = 0
    tile_end = num_tiles
    if SLIDING_WINDOW > 0 and not USE_MM_PREFIX:
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    if IS_3D:
        loop_lo = max(segm_idx * tiles_per_segment, tile_start)
        loop_hi = min((segm_idx + 1) * tiles_per_segment, tile_end)
    else:
        loop_lo = tile_start
        loop_hi = tile_end

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
            + qtr_offs[:, None] * stride_k_cache_3
            + slot_in_blk[None, :] * stride_k_cache_1
        )
        K_pk = tl.load(
            key_cache_ptr + k_off,
            mask=qtr_dim_mask[:, None] & tile_mask[None, :],
            other=0,
        )
        kc0_u, kc1_u, kc2_u, kc3_u = unpack_int2_quartet(K_pk)
        KC0 = _lloyd_max_dequant_4(kc0_u).to(tl.float32)
        KC1 = _lloyd_max_dequant_4(kc1_u).to(tl.float32)
        KC2 = _lloyd_max_dequant_4(kc2_u).to(tl.float32)
        KC3 = _lloyd_max_dequant_4(kc3_u).to(tl.float32)
        v_off = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + qtr_offs[None, :] * stride_v_cache_3
            + slot_in_blk[:, None] * stride_v_cache_1
        )
        V_pk = tl.load(
            value_cache_ptr + v_off,
            mask=qtr_dim_mask[None, :] & tile_mask[:, None],
            other=0,
        )
        vc0_u, vc1_u, vc2_u, vc3_u = unpack_int2_quartet(V_pk)
        VC0 = _lloyd_max_dequant_4(vc0_u).to(tl.float32)
        VC1 = _lloyd_max_dequant_4(vc1_u).to(tl.float32)
        VC2 = _lloyd_max_dequant_4(vc2_u).to(tl.float32)
        VC3 = _lloyd_max_dequant_4(vc3_u).to(tl.float32)

        ks_idx = (
            physical_block_idx * stride_ks_blk
            + slot_in_blk * stride_ks_slot
            + kv_head_idx * stride_ks_head
        )
        k_token_head_scales = tl.load(
            k_scale_cache_ptr + ks_idx, mask=tile_mask, other=0
        )
        vs_idx = (
            physical_block_idx * stride_vs_blk
            + slot_in_blk * stride_vs_slot
            + kv_head_idx * stride_vs_head
        )
        v_token_head_scales = tl.load(
            v_scale_cache_ptr + vs_idx, mask=tile_mask, other=0
        )

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

        # Score: 4-way split-dot with Lloyd-Max centroids.
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        raw_dot = (
            tl.dot(Q_s0, KC0)
            + tl.dot(Q_s1, KC1)
            + tl.dot(Q_s2, KC2)
            + tl.dot(Q_s3, KC3)
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
        acc_s2 = acc_s2 * alpha[:, None]
        acc_s3 = acc_s3 * alpha[:, None]

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            sw_mask = (context_len + qpos_lo - seq_offset) < SLIDING_WINDOW
            VC0 = tl.where(sw_mask[:, None], VC0, 0.0)
            VC1 = tl.where(sw_mask[:, None], VC1, 0.0)
            VC2 = tl.where(sw_mask[:, None], VC2, 0.0)
            VC3 = tl.where(sw_mask[:, None], VC3, 0.0)
        P_v = (P * v_token_head_scales[None, :]).to(tl.float32)
        acc_s0 += tl.dot(P_v, VC0)
        acc_s1 += tl.dot(P_v, VC1)
        acc_s2 += tl.dot(P_v, VC2)
        acc_s3 += tl.dot(P_v, VC3)

    out_mask = query_mask_0[:, None] & query_mask_1[:, None]
    if IS_3D:
        segm_base = (
            query_offset_0[:, None].to(tl.int64)
            * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
            + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
            + segm_idx * HEAD_SIZE_PADDED
        )
        tl.store(
            segm_output_ptr + segm_base + offs_q0[None, :],
            acc_s0,
            mask=mask_q0[None, :] & out_mask,
        )
        tl.store(
            segm_output_ptr + segm_base + offs_q1[None, :],
            acc_s1,
            mask=mask_q1[None, :] & out_mask,
        )
        tl.store(
            segm_output_ptr + segm_base + offs_q2[None, :],
            acc_s2,
            mask=mask_q2[None, :] & out_mask,
        )
        tl.store(
            segm_output_ptr + segm_base + offs_q3[None, :],
            acc_s3,
            mask=mask_q3[None, :] & out_mask,
        )
        segm_offset = (
            query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
            + query_offset_1 * NUM_SEGMENTS_PER_SEQ
            + segm_idx
        )
        tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
        tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)
    else:
        acc_s0 = acc_s0 / L[:, None]
        acc_s1 = acc_s1 / L[:, None]
        acc_s2 = acc_s2 / L[:, None]
        acc_s3 = acc_s3 / L[:, None]
        if USE_FP8:
            out_s = tl.load(out_scale)
            acc_s0 = tl.clamp(acc_s0 * out_s, FP8_MIN, FP8_MAX)
            acc_s1 = tl.clamp(acc_s1 * out_s, FP8_MIN, FP8_MAX)
            acc_s2 = tl.clamp(acc_s2 * out_s, FP8_MIN, FP8_MAX)
            acc_s3 = tl.clamp(acc_s3 * out_s, FP8_MIN, FP8_MAX)
        out_base = (
            query_offset_0[:, None] * output_stride_0
            + query_offset_1[:, None] * output_stride_1
        )
        tl.store(
            output_ptr + out_base + offs_q0[None, :],
            acc_s0,
            mask=mask_q0[None, :] & out_mask,
        )
        tl.store(
            output_ptr + out_base + offs_q1[None, :],
            acc_s1,
            mask=mask_q1[None, :] & out_mask,
        )
        tl.store(
            output_ptr + out_base + offs_q2[None, :],
            acc_s2,
            mask=mask_q2[None, :] & out_mask,
        )
        tl.store(
            output_ptr + out_base + offs_q3[None, :],
            acc_s3,
            mask=mask_q3[None, :] & out_mask,
        )


# ---------------------------------------------------------------------------
# Backend implementation
# ---------------------------------------------------------------------------
class Int2PerTokenHeadBackend(QuantKVBackend):
    """KV cache backend for ``KVQuantMode.INT2_PER_TOKEN_HEAD``."""

    mode = KVQuantMode.INT2_PER_TOKEN_HEAD
    packing_factor = 4  # 4 × int2 per byte
    needs_scale_caches = True

    def reshape_and_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        *,
        k_scale_cache: torch.Tensor | None = None,
        v_scale_cache: torch.Tensor | None = None,
    ) -> None:
        assert k_scale_cache is not None and v_scale_cache is not None, (
            "INT2 per-token-head requires k_scale_cache / v_scale_cache"
        )
        # Pre-rotation: full Hadamard (no random sign)
        key = fast_hadamard_transform(key.float()).to(key.dtype)
        value = fast_hadamard_transform(value.float()).to(value.dtype)

        num_tokens, num_kv_heads, head_size = key.shape
        head_size_v = value.shape[2]
        assert head_size % 4 == 0 and head_size_v % 4 == 0
        qtr_head_padded = triton.next_power_of_2(max(head_size, head_size_v) // 4)
        if current_platform.is_rocm() or current_platform.is_xpu():
            num_warps = 4
        else:
            num_warps = min(16, max(1, qtr_head_padded // 32))

        _reshape_cache_int2_kernel[(num_tokens, num_kv_heads)](
            key_ptr=key,
            value_ptr=value,
            key_cache_ptr=key_cache,
            value_cache_ptr=value_cache,
            k_scale_cache_ptr=k_scale_cache,
            v_scale_cache_ptr=v_scale_cache,
            slot_mapping_ptr=slot_mapping,
            stride_key_tok=key.stride(0),
            stride_key_head=key.stride(1),
            stride_val_tok=value.stride(0),
            stride_val_head=value.stride(1),
            stride_kc_blk=key_cache.stride(0),
            stride_kc_slot=key_cache.stride(1),
            stride_kc_head=key_cache.stride(2),
            stride_vc_blk=value_cache.stride(0),
            stride_vc_slot=value_cache.stride(1),
            stride_vc_head=value_cache.stride(2),
            stride_ks_blk=k_scale_cache.stride(0),
            stride_ks_slot=k_scale_cache.stride(1),
            stride_ks_head=k_scale_cache.stride(2),
            stride_vs_blk=v_scale_cache.stride(0),
            stride_vs_slot=v_scale_cache.stride(1),
            stride_vs_head=v_scale_cache.stride(2),
            block_size=key_cache.shape[1],
            head_size=head_size,
            head_size_v=head_size_v,
            QUARTER_HEAD_PADDED=qtr_head_padded,
            num_warps=num_warps,
        )

    def unified_attention(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        out: torch.Tensor,
        *,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        seqused_k: torch.Tensor,
        max_seqlen_k: int,
        softmax_scale: float,
        window_size: tuple[int, int],
        block_table: torch.Tensor,
        softcap: float,
        sinks: torch.Tensor | None,
        alibi_slopes: torch.Tensor | None,
        use_alibi_sqrt: bool,
        qq_bias: torch.Tensor | None,
        output_scale: torch.Tensor | None,
        mm_prefix_range: torch.Tensor | None,
        k_scale_cache: torch.Tensor | None = None,
        v_scale_cache: torch.Tensor | None = None,
        seq_threshold_3D: int | None = None,
        num_par_softmax_segments: int | None = None,
        softmax_segm_output: torch.Tensor | None = None,
        softmax_segm_max: torch.Tensor | None = None,
        softmax_segm_expsum: torch.Tensor | None = None,
    ) -> None:
        assert k_scale_cache is not None and v_scale_cache is not None
        import vllm.envs as envs
        from vllm.v1.attention.ops.triton_unified_attention import _get_tile_size

        is_batch_invariant = envs.VLLM_BATCH_INVARIANT

        # ---- Q rotation: full Hadamard, no rescaling -----------------------
        q_orig_dtype = q.dtype
        q = fast_hadamard_transform(q.float()).to(q_orig_dtype)
        head_size = q.shape[2]

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

        BLOCK_M = (
            16
            if num_queries_per_kv <= 16
            else triton.next_power_of_2(num_queries_per_kv)
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

        segm_output_ptr = softmax_segm_output if use_3d else out
        segm_max_ptr = softmax_segm_max if use_3d else out
        segm_expsum_ptr = softmax_segm_expsum if use_3d else out
        num_segments = num_par_softmax_segments if use_3d else 1

        if use_3d:
            grid = (total_num_q_blocks, num_kv_heads, num_par_softmax_segments)
            tile_size = TILE_SIZE_DECODE
        else:
            grid = (total_num_q_blocks, num_kv_heads)
            tile_size = TILE_SIZE_PREFILL

        _attn_int2[grid](
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
            QUARTER_HEAD_PADDED=triton.next_power_of_2(head_size) // 4,
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

        # ---- Inverse Hadamard on output ------------------------------------
        out_f = fast_hadamard_transform(out.float())
        out.copy_(out_f.to(q_orig_dtype))


register(Int2PerTokenHeadBackend())
