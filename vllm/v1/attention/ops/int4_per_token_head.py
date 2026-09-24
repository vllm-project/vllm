# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sub-byte packed (INT4) per-token-head KV cache mode.

INT4 packs two 4-bit values per cache byte, pre-rotates with a single RHT,
and hides a 4-bit zero-point in the scale's low mantissa bits — too
different from the core kernel to share it. Owns the whole mode: nibble
pack/unpack, the reshape (write) kernel, the split-dot attention (read)
kernel, the RHT transform, and the public ``reshape_and_cache_int4`` /
``unified_attention_int4`` entry points.
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
from vllm.v1.attention.ops.triton_unified_attention import reduce_segments

float8_info = torch.finfo(current_platform.fp8_dtype())

# 2 x int4 packed per storage byte.
_INT4_PACKING_FACTOR = 2


# ----------------------------------------------------------------------
# Nibble pack / unpack (shared write+read format)
# ----------------------------------------------------------------------


@triton.jit
def pack_int4_nibbles(lo, hi):
    """Pack two uint8 values (each in [0, 15]) into one byte."""
    return (lo & 0xF) | ((hi & 0xF) << 4)


@triton.jit
def unpack_int4_nibbles(packed):
    """Split one packed byte into the (low, high) nibble pair as uint8."""
    return packed & 0xF, (packed >> 4) & 0xF


# ----------------------------------------------------------------------
# Write path: RHT + pack + per-(token, head) scale
# ----------------------------------------------------------------------


@triton.jit
def _reshape_cache_int4_kernel(
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
    PACKED_HEAD_PADDED: tl.constexpr,
):
    """INT4 asymmetric quantization with zero-point steganography."""
    tok = tl.program_id(0)
    head = tl.program_id(1)

    slot = tl.load(slot_mapping_ptr + tok).to(tl.int64)
    if slot < 0:
        return

    blk = slot // block_size
    slot_in_blk = slot % block_size

    half_offs = tl.arange(0, PACKED_HEAD_PADDED)
    even_offs = half_offs * 2
    odd_offs = half_offs * 2 + 1

    half_k = head_size // 2
    even_k_mask = even_offs < head_size
    odd_k_mask = odd_offs < head_size
    key_base = key_ptr + tok * stride_key_tok + head * stride_key_head

    k_even = tl.load(key_base + even_offs, mask=even_k_mask, other=0.0).to(tl.float32)
    k_odd = tl.load(key_base + odd_offs, mask=odd_k_mask, other=0.0).to(tl.float32)

    k_min = tl.minimum(
        tl.min(tl.where(even_k_mask, k_even, float("inf"))),
        tl.min(tl.where(odd_k_mask, k_odd, float("inf"))),
    )
    k_max = tl.maximum(
        tl.max(tl.where(even_k_mask, k_even, float("-inf"))),
        tl.max(tl.where(odd_k_mask, k_odd, float("-inf"))),
    )
    k_scale = tl.maximum((k_max - k_min) / 15.0, 1e-6)
    k_zp_f = tl.clamp(
        tl.where(
            -k_min / k_scale >= 0,
            (-k_min / k_scale + 0.5).to(tl.int32),
            (-k_min / k_scale - 0.5).to(tl.int32),
        ).to(tl.float32),
        0.0,
        15.0,
    )

    inv_k = 1.0 / k_scale
    k_even_s = k_even * inv_k + k_zp_f
    k_odd_s = k_odd * inv_k + k_zp_f
    k_even_q = tl.clamp(
        tl.where(
            k_even_s >= 0,
            (k_even_s + 0.5).to(tl.int32),
            (k_even_s - 0.5).to(tl.int32),
        ).to(tl.float32),
        0.0,
        15.0,
    )
    k_odd_q = tl.clamp(
        tl.where(
            k_odd_s >= 0,
            (k_odd_s + 0.5).to(tl.int32),
            (k_odd_s - 0.5).to(tl.int32),
        ).to(tl.float32),
        0.0,
        15.0,
    )

    k_zp_int = k_zp_f.to(tl.int32)
    k_scale_bits = k_scale.to(tl.int32, bitcast=True)
    k_scale_packed = ((k_scale_bits & -16) | (k_zp_int & 0xF)).to(
        tl.float32, bitcast=True
    )

    tl.store(
        k_scale_cache_ptr
        + blk * stride_ks_blk
        + slot_in_blk * stride_ks_slot
        + head * stride_ks_head,
        k_scale_packed,
    )

    k_packed = pack_int4_nibbles(k_even_q.to(tl.uint8), k_odd_q.to(tl.uint8))
    tl.store(
        key_cache_ptr
        + blk * stride_kc_blk
        + slot_in_blk * stride_kc_slot
        + head * stride_kc_head
        + half_offs,
        k_packed,
        mask=half_offs < half_k,
    )

    half_v = head_size_v // 2
    even_v_mask = even_offs < head_size_v
    odd_v_mask = odd_offs < head_size_v
    val_base = value_ptr + tok * stride_val_tok + head * stride_val_head

    v_even = tl.load(val_base + even_offs, mask=even_v_mask, other=0.0).to(tl.float32)
    v_odd = tl.load(val_base + odd_offs, mask=odd_v_mask, other=0.0).to(tl.float32)

    v_min = tl.minimum(
        tl.min(tl.where(even_v_mask, v_even, float("inf"))),
        tl.min(tl.where(odd_v_mask, v_odd, float("inf"))),
    )
    v_max = tl.maximum(
        tl.max(tl.where(even_v_mask, v_even, float("-inf"))),
        tl.max(tl.where(odd_v_mask, v_odd, float("-inf"))),
    )
    v_scale = tl.maximum((v_max - v_min) / 15.0, 1e-6)
    v_zp_f = tl.clamp(
        tl.where(
            -v_min / v_scale >= 0,
            (-v_min / v_scale + 0.5).to(tl.int32),
            (-v_min / v_scale - 0.5).to(tl.int32),
        ).to(tl.float32),
        0.0,
        15.0,
    )

    inv_v = 1.0 / v_scale
    v_even_s = v_even * inv_v + v_zp_f
    v_odd_s = v_odd * inv_v + v_zp_f
    v_even_q = tl.clamp(
        tl.where(
            v_even_s >= 0,
            (v_even_s + 0.5).to(tl.int32),
            (v_even_s - 0.5).to(tl.int32),
        ).to(tl.float32),
        0.0,
        15.0,
    )
    v_odd_q = tl.clamp(
        tl.where(
            v_odd_s >= 0,
            (v_odd_s + 0.5).to(tl.int32),
            (v_odd_s - 0.5).to(tl.int32),
        ).to(tl.float32),
        0.0,
        15.0,
    )

    v_zp_int = v_zp_f.to(tl.int32)
    v_scale_bits = v_scale.to(tl.int32, bitcast=True)
    v_scale_packed = ((v_scale_bits & -16) | (v_zp_int & 0xF)).to(
        tl.float32, bitcast=True
    )

    tl.store(
        v_scale_cache_ptr
        + blk * stride_vs_blk
        + slot_in_blk * stride_vs_slot
        + head * stride_vs_head,
        v_scale_packed,
    )

    v_packed = pack_int4_nibbles(v_even_q.to(tl.uint8), v_odd_q.to(tl.uint8))
    tl.store(
        value_cache_ptr
        + blk * stride_vc_blk
        + slot_in_blk * stride_vc_slot
        + head * stride_vc_head
        + half_offs,
        v_packed,
        mask=half_offs < half_v,
    )


def _run_reshape_kernel(
    kernel,
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    packing_factor: int,
) -> None:
    """Launch the packed INT4 reshape kernel."""
    num_tokens, num_kv_heads, head_size = key.shape
    head_size_v = value.shape[2]
    assert head_size % packing_factor == 0 and head_size_v % packing_factor == 0
    packed_padded = triton.next_power_of_2(
        max(head_size, head_size_v) // packing_factor
    )
    if current_platform.is_rocm() or current_platform.is_xpu():
        num_warps = 4
    else:
        num_warps = min(16, max(1, packed_padded // 32))

    kernel[(num_tokens, num_kv_heads)](
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
        PACKED_HEAD_PADDED=packed_padded,
        num_warps=num_warps,
    )


# ----------------------------------------------------------------------
# Read path: split-dot attention over the packed cache
# ----------------------------------------------------------------------


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
    # 2 → INT4 nibble pair (asymmetric + zp).  The packed KV cache stores
    # one byte per ``packed_offs``, holding PACKING_FACTOR values.
    PACKING_FACTOR: tl.constexpr,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    # Shared prologue: sequence lookup, q-block bounds, early returns.
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

    # Split-Q prologue: PACKING_FACTOR interleaved streams of Q.
    # INT4 uses 2 streams (even / odd).  The packed KV cache stores one
    # byte per ``packed_offs``, which holds PACKING_FACTOR values.
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

    # INT4 asymmetric correction needs sum(Q) per row.
    Q_sum = tl.sum(Q_s0, axis=1) + tl.sum(Q_s1, axis=1)

    block_table_offset = seq_idx * block_table_stride

    # Online-softmax state + optional feature loads.
    M = init_softmax_M(
        sink_ptr, query_offset_1, query_mask_1, segm_idx, BLOCK_M, USE_SINKS, IS_3D
    )
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc_s0 = tl.zeros([BLOCK_M, PACKED_HEAD_PADDED], dtype=tl.float32)
    acc_s1 = tl.zeros([BLOCK_M, PACKED_HEAD_PADDED], dtype=tl.float32)

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

    # Tile loop.  Per-tile: load packed KV + scales, dequantize into
    # PACKING_FACTOR streams, compute the split dot, run the shared
    # softmax step, accumulate per stream.
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
        # the zero-point is applied on the score side.
        K_s0_u, K_s1_u = unpack_int4_nibbles(K_packed)
        K_s0 = K_s0_u.to(tl.float32)
        K_s1 = K_s1_u.to(tl.float32)
        V_s0_u, V_s1_u = unpack_int4_nibbles(V_packed)
        V_s0 = V_s0_u.to(tl.float32)
        V_s1 = V_s1_u.to(tl.float32)

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
        # the float32 scale's mantissa.
        ks_bits = ks_raw.to(tl.int32, bitcast=True)
        k_zp = (ks_bits & 0xF).to(tl.float32)
        k_token_head_scales = (ks_bits & -16).to(tl.float32, bitcast=True)
        vs_bits = vs_raw.to(tl.int32, bitcast=True)
        v_zp = (vs_bits & 0xF).to(tl.float32)
        v_token_head_scales = (vs_bits & -16).to(tl.float32, bitcast=True)

        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = compute_kv_seq_mask(
            query_abs_pos,
            seq_offset,
            seq_idx,
            seq_len,
            mm_prefix_range_ptr,
            SLIDING_WINDOW,
            USE_MM_PREFIX,
            MAX_MM_RANGES,
        )

        # Score: split-dot across the 2 INT4 streams; fused
        # softmax_scale * per-(token, head) k_scale in one mul.  INT4
        # subtracts the ``zp * sum(Q)`` correction term.
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        raw_dot = tl.dot(Q_s0, K_s0) + tl.dot(Q_s1, K_s1)
        S += (raw_dot - Q_sum[:, None] * k_zp[None, :]) * (
            scale * k_token_head_scales[None, :]
        )

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

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            sw_mask = (context_len + qpos_lo - seq_offset) < SLIDING_WINDOW
            V_s0 = tl.where(sw_mask[:, None], V_s0, 0.0)
            V_s1 = tl.where(sw_mask[:, None], V_s1, 0.0)

        # Fuse v per-(token, head) scale into P.  INT4 also subtracts
        # the v-zero-point contribution from each stream once.
        P_v = (P * v_token_head_scales[None, :]).to(tl.float32)
        Pv_zp_sum = tl.sum(P_v * v_zp[None, :], axis=1)
        acc_s0 += tl.dot(P_v, V_s0) - Pv_zp_sum[:, None]
        acc_s1 += tl.dot(P_v, V_s1) - Pv_zp_sum[:, None]

    # Epilogue.  2D writes the final output with optional FP8 clamp;
    # 3D writes the per-segment partials (output / max / expsum) for
    # ``reduce_segments`` to finalize.  Each stream writes its own
    # stripe in the output layout.
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
        if USE_FP8:
            out_s = tl.load(out_scale)
            acc_s0 = tl.clamp(acc_s0 * out_s, FP8_MIN, FP8_MAX)
            acc_s1 = tl.clamp(acc_s1 * out_s, FP8_MIN, FP8_MAX)
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


# ----------------------------------------------------------------------
# Public entry points
# ----------------------------------------------------------------------


def reshape_and_cache_int4(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
) -> None:
    """Pre-rotate (RHT), pack to INT4 and write into the paged cache."""
    key = single_rht(key.float()).to(key.dtype)
    value = single_rht(value.float()).to(value.dtype)
    _run_reshape_kernel(
        _reshape_cache_int4_kernel,
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        k_scale_cache=k_scale_cache,
        v_scale_cache=v_scale_cache,
        slot_mapping=slot_mapping,
        packing_factor=_INT4_PACKING_FACTOR,
    )


def unified_attention_int4(
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
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    seq_threshold_3D: int | None = None,
    num_par_softmax_segments: int | None = None,
    softmax_segm_output: torch.Tensor | None = None,
    softmax_segm_max: torch.Tensor | None = None,
    softmax_segm_expsum: torch.Tensor | None = None,
) -> None:
    """Paged attention over the INT4 packed cache, writing into *out*.

    The forward RHT has norm ``sqrt(head_size)``, so ``softmax_scale`` is
    divided by ``head_size`` and the inverse RHT divides the output by
    ``head_size`` as well.
    """
    q_orig_dtype = q.dtype
    q = single_rht(q.float()).to(q_orig_dtype)
    head_size = q.shape[2]
    softmax_scale = softmax_scale / head_size

    _launch_packed_attn(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        seqused_k=seqused_k,
        softmax_scale=softmax_scale,
        window_size=window_size,
        block_table=block_table,
        softcap=softcap,
        sinks=sinks,
        alibi_slopes=alibi_slopes,
        use_alibi_sqrt=use_alibi_sqrt,
        qq_bias=qq_bias,
        output_scale=output_scale,
        mm_prefix_range=mm_prefix_range,
        k_scale_cache=k_scale_cache,
        v_scale_cache=v_scale_cache,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
        packing_factor=_INT4_PACKING_FACTOR,
    )

    out_f = single_rht(out.float(), inverse=True) / head_size
    out.copy_(out_f.to(q_orig_dtype))


# ----------------------------------------------------------------------
# Randomized Hadamard Transform (RHT) — gaussianizes K/V before INT4
# quantization; applied on write and to Q before the read kernels.
# ----------------------------------------------------------------------

# Hadacore (CUDA tensor core kernel) availability check
# Hadacore's CUDA impl is only registered when built for sm_80+, but the
# schema def is unconditional — on ROCm ``hasattr`` is True yet dispatch
# would crash, so we also gate on ``is_cuda()`` and the sm_80 capability.
_HADACORE_AVAILABLE: bool | None = None


def _hadacore_available() -> bool:
    global _HADACORE_AVAILABLE
    if _HADACORE_AVAILABLE is None:
        _HADACORE_AVAILABLE = (
            current_platform.is_cuda()
            and current_platform.has_device_capability(80)
            and hasattr(torch.ops._C, "hadacore_transform")
        )
    return _HADACORE_AVAILABLE


# Cached Hadamard matrices (one per (size, dtype, device) tuple)
_HADAMARD_MATRIX_CACHE: dict[tuple[int, torch.dtype, str], torch.Tensor] = {}


def _get_hadamard_matrix(
    d: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    key = (d, dtype, str(device))
    cached = _HADAMARD_MATRIX_CACHE.get(key)
    if cached is None:
        H = torch.ones(1, 1, dtype=torch.float32, device=device)
        while H.shape[0] < d:
            H = torch.cat(
                [
                    torch.cat([H, H], dim=1),
                    torch.cat([H, -H], dim=1),
                ],
                dim=0,
            )
        cached = H.to(dtype).contiguous()
        _HADAMARD_MATRIX_CACHE[key] = cached
    return cached


# Triton MMA Hadamard kernel (Tier 2)
@triton.jit
def _hadamard_mma_kernel(
    x_ptr,
    h_ptr,
    out_ptr,
    n_rows,
    stride_x_row: tl.int64,
    stride_x_col: tl.int64,
    stride_o_row: tl.int64,
    stride_o_col: tl.int64,
    BLOCK_M: tl.constexpr,
    D: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, D)
    row_mask = rows < n_rows

    x = tl.load(
        x_ptr + rows[:, None] * stride_x_row + cols[None, :] * stride_x_col,
        mask=row_mask[:, None],
        other=0.0,
    )
    H = tl.load(h_ptr + cols[:, None] * D + cols[None, :])

    out = tl.dot(x, H, out_dtype=tl.float32).to(x.dtype)

    tl.store(
        out_ptr + rows[:, None] * stride_o_row + cols[None, :] * stride_o_col,
        out,
        mask=row_mask[:, None],
    )


# H is D×D bf16 = 2·D² bytes of LDS.  AMD CDNA has 64 KiB LDS, so D ≤ 128
# (32 KiB) leaves room for input + accumulator.  Larger D falls back.
_TRITON_HADAMARD_MIN_D = 16
_TRITON_HADAMARD_MAX_D = 128


def _triton_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    orig_shape = x.shape
    orig_dtype = x.dtype

    work_dtype = torch.bfloat16 if orig_dtype == torch.float32 else orig_dtype
    x2d = x.contiguous().to(work_dtype).reshape(-1, d)
    out2d = torch.empty_like(x2d)
    n_rows = x2d.shape[0]
    H_mat = _get_hadamard_matrix(d, work_dtype, x.device)

    BLOCK_M = 16
    grid = (triton.cdiv(n_rows, BLOCK_M),)
    # num_stages=1: the kernel has no loop, so default 3-stage pipelining
    # would triple-buffer H and blow the AMD LDS budget.
    _hadamard_mma_kernel[grid](
        x2d,
        H_mat,
        out2d,
        n_rows,
        x2d.stride(0),
        x2d.stride(1),
        out2d.stride(0),
        out2d.stride(1),
        BLOCK_M=BLOCK_M,
        D=d,
        num_stages=1,
        num_warps=4,
    )
    return out2d.reshape(orig_shape).to(orig_dtype)


# Public API
def fast_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Unnormalized Walsh-Hadamard Transform along the last dimension.

    H_d × x where H_d × H_d = d × I.  Last dim must be a power of 2.

    Three-tier dispatch:
      1. Hadacore CUDA Tensor Core kernel (sm_80+).
      2. Triton MMA matmul kernel (CUDA fallback + ROCm MFMA/WMMA path).
      3. PyTorch butterfly (CPU and any GPU/dtype combo Triton can't take).
    """
    d = x.shape[-1]
    assert d & (d - 1) == 0, f"Requires power-of-2 dim, got {d}"

    # Tier 1 — hadacore on CUDA.
    if _hadacore_available() and 0 < d <= (1 << 15):
        from vllm import _custom_ops as ops

        # hadacore returns x @ (H/√d); rescale to the unnormalized H × x
        # convention the INT4 scale math is calibrated to.
        rescale = d**0.5
        if x.dtype in (torch.float16, torch.bfloat16):
            y = ops.hadacore_transform(x.contiguous().clone(), inplace=True)
            return y * rescale
        # fp32 → bf16 round-trip; precision loss is irrelevant before
        # INT4 quantization.
        orig_dtype = x.dtype
        x_bf16 = x.contiguous().to(torch.bfloat16)
        y_bf16 = ops.hadacore_transform(x_bf16, inplace=True)
        return y_bf16.to(orig_dtype) * rescale

    # Tier 2 — Triton MMA kernel (covers ROCm via MFMA/WMMA codegen, and
    # also CUDA when hadacore is unavailable).
    if (
        x.is_cuda
        and _TRITON_HADAMARD_MIN_D <= d <= _TRITON_HADAMARD_MAX_D
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return _triton_hadamard_transform(x)

    # Tier 3 — PyTorch butterfly (CPU / unsupported dtype / D < 16).
    h = 1
    while h < d:
        xv = x.view(*x.shape[:-1], d // (2 * h), 2, h)
        a = xv[..., 0, :]
        b = xv[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2).reshape(x.shape)
        h <<= 1
    return x


# Randomized Hadamard Transform (used by INT4)
# Deterministic ±1 signs for Randomized Hadamard Transform.
# RHT = H × D × x  (sign flip + Hadamard).  Breaks residual structure
# in KV vectors, improving quantization quality.
_RHT_SIGNS_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}


def _get_rht_signs(d: int, round_idx: int, device: torch.device) -> torch.Tensor:
    """Return a cached deterministic ±1 sign vector of length *d*."""
    key = (d, round_idx, str(device))
    if key not in _RHT_SIGNS_CACHE:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0x9E3779B9 + round_idx * 0x517CC1B7)
        signs = (
            2.0 * torch.bernoulli(torch.full((d,), 0.5, device="cpu"), generator=gen)
            - 1.0
        )
        _RHT_SIGNS_CACHE[key] = signs.to(device)
    return _RHT_SIGNS_CACHE[key]


def single_rht(x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Single Randomized Hadamard Transform: H × D₁ × x.

    Used by INT4 per-token-head quantization to gaussianize data
    before asymmetric quantization.
    """
    d = x.shape[-1]
    d1 = _get_rht_signs(d, 0, x.device)
    if inverse:
        return fast_hadamard_transform(x) * d1
    else:
        return fast_hadamard_transform(x * d1)
