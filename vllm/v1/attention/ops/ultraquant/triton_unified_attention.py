# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified Triton fallback for the UltraQuant 4-bit KV-cache format.

K and V use packed FP4 E2M1 values with UE8M0 group-of-32 scales. QK uses
native scaled FP4×E4M3 MFMA on CDNA4. Production choices are fixed in code;
there are no environment-variable tuning switches.
"""

from __future__ import annotations

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_unified_attention import (
    reduce_segments,
)

# Reuse the sequence-index search and vectorized partition reducer.
from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_unified_attention import (
    _find_seq_idx,
)
from vllm.v1.attention.ops.ultraquant.format import (
    UE8M0_BIAS,
    get_group_size,
    k_scales_offset,
    n_groups,
    slot_size,
    v_codes_offset,
    v_scales_offset,
)
from vllm.v1.attention.ops.ultraquant.triton_store import _kv_cache_flat

_is_hip = current_platform.is_rocm()


@triton.jit
def _ultraquant_fp4_decode_arith(codes):
    """Arithmetic FP4 E2M1 decode, bit-exact to format.FP4_BITS_TO_VALUE."""
    mag = codes & 7
    e = mag >> 1
    m = mag & 1
    exp_field = 126 + e
    mant_field = tl.where(e != 0, m, 0) << 22
    bits = (exp_field << 23) | mant_field
    magval = bits.to(tl.float32, bitcast=True)
    magval = tl.where(mag == 0, 0.0, magval)
    signf = tl.where((codes & 8) != 0, -1.0, 1.0)
    return magval * signf


@triton.jit
def _ultraquant_load_k_packed(
    KV_cache_ptr,
    data_bases,
    k_scales_addrs,
    d_half_offs,
    half_mask,
    tile_mask,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    UNMASKED: tl.constexpr,
):
    """Load packed FP4 K codes and UE8M0 scales for scaled QK MFMA."""
    addrs = data_bases[:, None] + d_half_offs[None, :]
    if UNMASKED:
        K_codes = tl.load(KV_cache_ptr + addrs, mask=half_mask[None, :], other=0)
    else:
        K_codes = tl.load(
            KV_cache_ptr + addrs,
            mask=tile_mask[:, None] & half_mask[None, :],
            other=0,
        )
    K_T_packed = tl.trans(K_codes)
    grp = tl.arange(0, N_GROUPS_C)
    scale_addrs = k_scales_addrs[:, None] + grp[None, :]
    if UNMASKED:
        K_scales = tl.load(KV_cache_ptr + scale_addrs)
    else:
        K_scales = tl.load(KV_cache_ptr + scale_addrs, mask=tile_mask[:, None], other=0)
    _ = HEAD_DIM
    _ = GROUP_SIZE_C
    _ = BLOCK_D
    return K_T_packed, K_scales


@triton.jit
def _ultraquant_load_v_tile(
    KV_cache_ptr,
    val_bases,
    v_scales_addrs,
    d_offs,
    d_mask,
    tile_mask,
    OUT_DTYPE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    UNMASKED: tl.constexpr,
    UE8M0_BIAS_C: tl.constexpr,
):
    """Load and dequant a V tile with arithmetic FP4 decode."""
    half_idx = d_offs // 2
    nibble_shift = (d_offs % 2) * 4
    addrs = val_bases[:, None] + half_idx[None, :]
    if UNMASKED:
        byte_raw = tl.load(KV_cache_ptr + addrs, mask=d_mask[None, :], other=0).to(
            tl.int32
        )
    else:
        byte_raw = tl.load(
            KV_cache_ptr + addrs,
            mask=tile_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
    codes = (byte_raw >> nibble_shift[None, :]) & 0xF
    grp = tl.arange(0, N_GROUPS_C)
    scale_addrs = v_scales_addrs[:, None] + grp[None, :]
    if UNMASKED:
        scale_bytes = tl.load(KV_cache_ptr + scale_addrs).to(tl.int32)
    else:
        scale_bytes = tl.load(
            KV_cache_ptr + scale_addrs, mask=tile_mask[:, None], other=0
        ).to(tl.int32)
    scale_exp = scale_bytes - UE8M0_BIAS_C
    fp4_vals = _ultraquant_fp4_decode_arith(codes)
    scales = tl.where(scale_bytes == 0, 0.0, tl.exp2(tl.cast(scale_exp, tl.float32)))
    V_g = tl.reshape(fp4_vals, [TILE_SIZE, N_GROUPS_C, GROUP_SIZE_C])
    return tl.reshape(V_g * scales[:, :, None], [TILE_SIZE, BLOCK_D]).to(OUT_DTYPE)


@triton.jit
def _ultraquant_pv(
    P,
    KV_cache_ptr,
    val_bases,
    v_scales_addrs,
    d_offs,
    d_mask,
    tile_mask,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    UNMASKED: tl.constexpr,
    UE8M0_BIAS_C: tl.constexpr,
):
    V = _ultraquant_load_v_tile(
        KV_cache_ptr,
        val_bases,
        v_scales_addrs,
        d_offs,
        d_mask,
        tile_mask,
        OUT_DTYPE=tl.bfloat16,
        HEAD_DIM=HEAD_DIM,
        BLOCK_D=BLOCK_D,
        GROUP_SIZE_C=GROUP_SIZE_C,
        N_GROUPS_C=N_GROUPS_C,
        TILE_SIZE=TILE_SIZE,
        UNMASKED=UNMASKED,
        UE8M0_BIAS_C=UE8M0_BIAS_C,
    )
    _ = BLOCK_M
    return tl.dot(P.to(tl.bfloat16), V, out_dtype=tl.float32)


@triton.jit
def _ultraquant_qk(
    Q,
    KV_cache_ptr,
    data_bases,
    k_scales_addrs,
    offs_d_half,
    half_mask,
    tile_mask,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    UNMASKED: tl.constexpr,
):
    K_T_packed, K_scales = _ultraquant_load_k_packed(
        KV_cache_ptr,
        data_bases,
        k_scales_addrs,
        offs_d_half,
        half_mask,
        tile_mask,
        BLOCK_D=BLOCK_D,
        HEAD_DIM=HEAD_DIM,
        GROUP_SIZE_C=GROUP_SIZE_C,
        N_GROUPS_C=N_GROUPS_C,
        TILE_SIZE=TILE_SIZE,
        UNMASKED=UNMASKED,
    )
    return tl.dot_scaled(
        Q,
        None,
        "e4m3",
        K_T_packed,
        K_scales,
        "e2m1",
        out_dtype=tl.float32,
    )


@triton.jit
def _ultraquant_slot_addrs(
    block_base,
    slot_within_block,
    kv_head_idx,
    stride_cache_pos: tl.int64,
    stride_cache_head: tl.int64,
    K_SCALES_OFFSET: tl.constexpr,
    V_CODES_OFFSET: tl.constexpr,
    V_SCALES_OFFSET: tl.constexpr,
):
    data_bases = (
        block_base
        + slot_within_block * stride_cache_pos
        + tl.cast(kv_head_idx, tl.int64) * stride_cache_head
    )
    return (
        data_bases,
        data_bases + K_SCALES_OFFSET,
        data_bases + V_CODES_OFFSET,
        data_bases + V_SCALES_OFFSET,
    )


# ===========================================================================
# Unified 2D attention kernel (prefill / short-context decode)
# ===========================================================================


@triton.jit
def kernel_ultraquant_unified_attention_2d(
    output_ptr,
    query_ptr,  # FP8 E4M3, already Hadamard-rotated
    KV_cache_ptr,  # uint8 view
    block_tables_ptr,
    seq_lens_ptr,
    query_start_len_ptr,
    sinks_ptr,
    scale,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    stride_cache_block: tl.int64,
    stride_cache_pos: tl.int64,
    stride_cache_head: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    num_seqs: tl.int32,
    K_SCALES_OFFSET: tl.constexpr,
    V_CODES_OFFSET: tl.constexpr,
    V_SCALES_OFFSET: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    UE8M0_BIAS_C: tl.constexpr,
    USE_SINKS: tl.constexpr = 0,
    SLIDING_WINDOW: tl.constexpr = 0,
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
    offs_d_half = tl.arange(0, HEAD_SIZE_PADDED // 2)
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
    half_mask = tl.where(offs_d_half * 2 < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=tl.zeros([], tl.float8e4nv),
    )

    block_table_offset = seq_idx * block_table_stride

    if USE_SINKS:
        M = tl.load(
            sinks_ptr + query_offset_1,
            mask=query_mask_1,
            other=float("-inf"),
        ).to(tl.float32)
    else:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - cur_batch_query_len

    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    num_tiles = tl.cdiv(max_seq_prefix_len, TILE_SIZE)

    tile_start = 0
    tile_end = num_tiles
    if SLIDING_WINDOW > 0:
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    query_abs_pos = context_len + query_pos[:, None]
    dummy_tile_mask = tl.full([TILE_SIZE], 1, tl.int1)

    for j in range(tile_start, tile_end - 1):
        seq_offset = j * TILE_SIZE + offs_t
        if TILE_SIZE == BLOCK_SIZE:
            physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j).to(
                tl.int64
            )
            slot_within_block = offs_t.to(tl.int64)
        else:
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            ).to(tl.int64)
            slot_within_block = (seq_offset % BLOCK_SIZE).to(tl.int64)
        block_base = physical_block_idx * stride_cache_block
        data_bases, k_scales_addrs, val_bases, v_scales_addrs = _ultraquant_slot_addrs(
            block_base,
            slot_within_block,
            kv_head_idx,
            stride_cache_pos,
            stride_cache_head,
            K_SCALES_OFFSET=K_SCALES_OFFSET,
            V_CODES_OFFSET=V_CODES_OFFSET,
            V_SCALES_OFFSET=V_SCALES_OFFSET,
        )

        S = scale * _ultraquant_qk(
            Q,
            KV_cache_ptr,
            data_bases,
            k_scales_addrs,
            offs_d_half,
            half_mask,
            dummy_tile_mask,
            BLOCK_D=HEAD_SIZE_PADDED,
            HEAD_DIM=HEAD_SIZE,
            GROUP_SIZE_C=GROUP_SIZE_C,
            N_GROUPS_C=N_GROUPS_C,
            TILE_SIZE=TILE_SIZE,
            UNMASKED=True,
        )
        seq_mask = seq_offset[None, :] <= query_abs_pos
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & (
                (query_abs_pos - seq_offset[None, :]) < SLIDING_WINDOW
            )
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
        acc += _ultraquant_pv(
            P,
            KV_cache_ptr,
            val_bases,
            v_scales_addrs,
            offs_d,
            dim_mask,
            dummy_tile_mask,
            HEAD_DIM=HEAD_SIZE,
            BLOCK_D=HEAD_SIZE_PADDED,
            BLOCK_M=BLOCK_M,
            GROUP_SIZE_C=GROUP_SIZE_C,
            N_GROUPS_C=N_GROUPS_C,
            TILE_SIZE=TILE_SIZE,
            UNMASKED=True,
            UE8M0_BIAS_C=UE8M0_BIAS_C,
        )

    # Tail tile (masked)
    if tile_end > tile_start:
        j = tile_end - 1
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len
        if TILE_SIZE == BLOCK_SIZE:
            physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j).to(
                tl.int64
            )
            slot_within_block = offs_t.to(tl.int64)
        else:
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            ).to(tl.int64)
            slot_within_block = (seq_offset % BLOCK_SIZE).to(tl.int64)
        block_base = physical_block_idx * stride_cache_block
        data_bases, k_scales_addrs, val_bases, v_scales_addrs = _ultraquant_slot_addrs(
            block_base,
            slot_within_block,
            kv_head_idx,
            stride_cache_pos,
            stride_cache_head,
            K_SCALES_OFFSET=K_SCALES_OFFSET,
            V_CODES_OFFSET=V_CODES_OFFSET,
            V_SCALES_OFFSET=V_SCALES_OFFSET,
        )

        S = scale * _ultraquant_qk(
            Q,
            KV_cache_ptr,
            data_bases,
            k_scales_addrs,
            offs_d_half,
            half_mask,
            tile_mask,
            BLOCK_D=HEAD_SIZE_PADDED,
            HEAD_DIM=HEAD_SIZE,
            GROUP_SIZE_C=GROUP_SIZE_C,
            N_GROUPS_C=N_GROUPS_C,
            TILE_SIZE=TILE_SIZE,
            UNMASKED=False,
        )
        seq_mask = seq_offset[None, :] <= query_abs_pos
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & (
                (query_abs_pos - seq_offset[None, :]) < SLIDING_WINDOW
            )
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
        acc += _ultraquant_pv(
            P,
            KV_cache_ptr,
            val_bases,
            v_scales_addrs,
            offs_d,
            dim_mask,
            tile_mask,
            HEAD_DIM=HEAD_SIZE,
            BLOCK_D=HEAD_SIZE_PADDED,
            BLOCK_M=BLOCK_M,
            GROUP_SIZE_C=GROUP_SIZE_C,
            N_GROUPS_C=N_GROUPS_C,
            TILE_SIZE=TILE_SIZE,
            UNMASKED=False,
            UE8M0_BIAS_C=UE8M0_BIAS_C,
        )

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


# ===========================================================================
# Unified 3D (split-KV) attention kernel
# ===========================================================================


@triton.jit
def kernel_ultraquant_unified_attention_3d(
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    query_ptr,  # FP8 E4M3, already Hadamard-rotated
    KV_cache_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    query_start_len_ptr,
    sinks_ptr,
    scale,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    stride_cache_block: tl.int64,
    stride_cache_pos: tl.int64,
    stride_cache_head: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    num_seqs: tl.int32,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    K_SCALES_OFFSET: tl.constexpr,
    V_CODES_OFFSET: tl.constexpr,
    V_SCALES_OFFSET: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    UE8M0_BIAS_C: tl.constexpr,
    USE_SINKS: tl.constexpr = 0,
    SLIDING_WINDOW: tl.constexpr = 0,
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
    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_d_half = tl.arange(0, HEAD_SIZE_PADDED // 2)
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
    half_mask = tl.where(offs_d_half * 2 < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=tl.zeros([], tl.float8e4nv),
    )

    block_table_offset = seq_idx * block_table_stride

    if USE_SINKS and segm_idx == 0:
        M = tl.load(
            sinks_ptr + query_offset_1,
            mask=query_mask_1,
            other=float("-inf"),
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

    tile_lo = segm_idx * tiles_per_segment
    tile_hi = tl.minimum((segm_idx + 1) * tiles_per_segment, num_tiles)

    if SLIDING_WINDOW > 0:
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        swa_tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        swa_tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)
        tile_lo = tl.maximum(tile_lo, swa_tile_start)
        tile_hi = tl.minimum(tile_hi, swa_tile_end)

    query_abs_pos = context_len + query_pos[:, None]
    dummy_tile_mask = tl.full([TILE_SIZE], 1, tl.int1)
    tail = tile_hi - 1

    for j in range(tile_lo, tile_hi - 1):
        seq_offset = j * TILE_SIZE + offs_t
        if TILE_SIZE == BLOCK_SIZE:
            physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j).to(
                tl.int64
            )
            slot_within_block = offs_t.to(tl.int64)
        else:
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            ).to(tl.int64)
            slot_within_block = (seq_offset % BLOCK_SIZE).to(tl.int64)
        block_base = physical_block_idx * stride_cache_block
        data_bases, k_scales_addrs, val_bases, v_scales_addrs = _ultraquant_slot_addrs(
            block_base,
            slot_within_block,
            kv_head_idx,
            stride_cache_pos,
            stride_cache_head,
            K_SCALES_OFFSET=K_SCALES_OFFSET,
            V_CODES_OFFSET=V_CODES_OFFSET,
            V_SCALES_OFFSET=V_SCALES_OFFSET,
        )

        S = scale * _ultraquant_qk(
            Q,
            KV_cache_ptr,
            data_bases,
            k_scales_addrs,
            offs_d_half,
            half_mask,
            dummy_tile_mask,
            BLOCK_D=HEAD_SIZE_PADDED,
            HEAD_DIM=HEAD_SIZE,
            GROUP_SIZE_C=GROUP_SIZE_C,
            N_GROUPS_C=N_GROUPS_C,
            TILE_SIZE=TILE_SIZE,
            UNMASKED=True,
        )
        seq_mask = seq_offset[None, :] <= query_abs_pos
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & (
                (query_abs_pos - seq_offset[None, :]) < SLIDING_WINDOW
            )
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
        acc += _ultraquant_pv(
            P,
            KV_cache_ptr,
            val_bases,
            v_scales_addrs,
            offs_d,
            dim_mask,
            dummy_tile_mask,
            HEAD_DIM=HEAD_SIZE,
            BLOCK_D=HEAD_SIZE_PADDED,
            BLOCK_M=BLOCK_M,
            GROUP_SIZE_C=GROUP_SIZE_C,
            N_GROUPS_C=N_GROUPS_C,
            TILE_SIZE=TILE_SIZE,
            UNMASKED=True,
            UE8M0_BIAS_C=UE8M0_BIAS_C,
        )

    # Tail
    if tile_lo < tile_hi:
        j = tail
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len
        if TILE_SIZE == BLOCK_SIZE:
            physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j).to(
                tl.int64
            )
            slot_within_block = offs_t.to(tl.int64)
        else:
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            ).to(tl.int64)
            slot_within_block = (seq_offset % BLOCK_SIZE).to(tl.int64)
        block_base = physical_block_idx * stride_cache_block
        data_bases, k_scales_addrs, val_bases, v_scales_addrs = _ultraquant_slot_addrs(
            block_base,
            slot_within_block,
            kv_head_idx,
            stride_cache_pos,
            stride_cache_head,
            K_SCALES_OFFSET=K_SCALES_OFFSET,
            V_CODES_OFFSET=V_CODES_OFFSET,
            V_SCALES_OFFSET=V_SCALES_OFFSET,
        )

        S = scale * _ultraquant_qk(
            Q,
            KV_cache_ptr,
            data_bases,
            k_scales_addrs,
            offs_d_half,
            half_mask,
            tile_mask,
            BLOCK_D=HEAD_SIZE_PADDED,
            HEAD_DIM=HEAD_SIZE,
            GROUP_SIZE_C=GROUP_SIZE_C,
            N_GROUPS_C=N_GROUPS_C,
            TILE_SIZE=TILE_SIZE,
            UNMASKED=False,
        )
        seq_mask = seq_offset[None, :] <= query_abs_pos
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & (
                (query_abs_pos - seq_offset[None, :]) < SLIDING_WINDOW
            )
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
        acc += _ultraquant_pv(
            P,
            KV_cache_ptr,
            val_bases,
            v_scales_addrs,
            offs_d,
            dim_mask,
            tile_mask,
            HEAD_DIM=HEAD_SIZE,
            BLOCK_D=HEAD_SIZE_PADDED,
            BLOCK_M=BLOCK_M,
            GROUP_SIZE_C=GROUP_SIZE_C,
            N_GROUPS_C=N_GROUPS_C,
            TILE_SIZE=TILE_SIZE,
            UNMASKED=False,
            UE8M0_BIAS_C=UE8M0_BIAS_C,
        )

    # Write segment partials for stage-2 reduce.
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


# ===========================================================================
# Launcher
# ===========================================================================


_HADAMARD_CACHE: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def _get_pit(dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Sylvester Hadamard projection / sqrt(dim). PiT = PiT.T (symmetric)."""
    key = (dim, device, dtype)
    H = _HADAMARD_CACHE.get(key)
    if H is None:
        assert (dim & (dim - 1)) == 0, f"dim={dim} must be power of 2"
        H = torch.tensor([[1.0]], dtype=torch.float32)
        while H.shape[0] < dim:
            H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
        H = (H / (dim**0.5)).to(device=device, dtype=dtype).contiguous()
        _HADAMARD_CACHE[key] = H
    return H


def ultraquant_unified_attention(
    query: torch.Tensor,  # [num_tokens, Hq, D] fp16/bf16 (raw)
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, padded_slot] uint8
    block_table: torch.Tensor,  # [num_seqs, max_num_blocks] int32
    seq_lens: torch.Tensor,  # [num_seqs] int32
    query_start_loc: torch.Tensor,  # [num_seqs+1] int32
    scale: float,
    PiT: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    tile_size: int | None = None,
    max_query_len: int | None = None,
    max_seq_len: int | None = None,
    num_kv_splits: int | None = None,
    force_2d: bool = False,
    sinks: torch.Tensor | None = None,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Launch unified UltraQuant attention with scaled F8F6F4 MFMA."""
    assert query.dim() == 3, f"query must be [N, Hq, D], got {query.shape}"
    num_tokens, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    padded_slot = kv_cache.shape[3]
    kv_group_size = Hq // Hk
    num_seqs = int(query_start_loc.shape[0] - 1)
    device = query.device

    gs = get_group_size()
    if padded_slot < slot_size(D, gs):
        raise ValueError(
            f"ultraquant_unified: cache slot {padded_slot} < expected "
            f"{slot_size(D, gs)} for D={D} group_size={gs}"
        )
    if Hq % Hk != 0:
        raise ValueError(f"Hq={Hq} must be a multiple of Hk={Hk}")

    if PiT is None:
        PiT = _get_pit(D, device, torch.float32)
    elif PiT.dtype != torch.float32 or not PiT.is_contiguous():
        PiT = PiT.to(torch.float32).contiguous()

    q_rot = (query.float() @ PiT).contiguous()
    q_for_kernel = q_rot.to(torch.float8_e4m3fn).contiguous()

    if sinks is not None:
        sinks_f32 = sinks if sinks.dtype == torch.float32 else sinks.to(torch.float32)
        if not sinks_f32.is_contiguous():
            sinks_f32 = sinks_f32.contiguous()
        assert sinks_f32.numel() == Hq, (
            f"sinks must have shape [Hq={Hq}], got numel={sinks_f32.numel()}"
        )
        use_sinks = True
    else:
        sinks_f32 = q_for_kernel  # harmless dummy; never dereferenced when USE_SINKS=0
        use_sinks = False

    if output is None:
        output = torch.empty_like(query)

    # BLOCK_M heuristic shared with the compressed-KV fallback kernels.
    if max_query_len is not None:
        is_prefill_like = max_query_len > 1
    else:
        is_prefill_like = num_tokens > num_seqs

    if is_prefill_like:
        BLOCK_M = max(128, triton.next_power_of_2(kv_group_size))
    else:
        BLOCK_M = 16 if kv_group_size <= 16 else triton.next_power_of_2(kv_group_size)
    BLOCK_Q = BLOCK_M // kv_group_size

    total_num_q_blocks = num_tokens // BLOCK_Q + num_seqs

    if tile_size is None:
        tile_size = 32 if is_prefill_like else 16

    num_stages_2d = 1 if _is_hip else 2
    num_stages_3d = 3 if _is_hip else 2

    BLOCK_D = triton.next_power_of_2(D)
    N_GROUPS_C = n_groups(D, gs)
    K_SCALES_OFFSET = k_scales_offset(D, gs)
    V_CODES_OFFSET = v_codes_offset(D, gs)
    V_SCALES_OFFSET = v_scales_offset(D, gs)

    scale_for_kernel = float(scale)

    kv_flat = _kv_cache_flat(kv_cache)

    # Dispatch: 2D for prefill / chunked; 3D for pure decode with long KV.
    if max_seq_len is None:
        max_seq_len_hint = int(block_table.shape[1]) * int(block_size)
    else:
        max_seq_len_hint = int(max_seq_len)
    use_3d = (not force_2d) and (not is_prefill_like) and max_seq_len_hint >= 1024

    if not use_3d:
        kernel_ultraquant_unified_attention_2d[(total_num_q_blocks, Hk)](
            output_ptr=output,
            query_ptr=q_for_kernel,
            KV_cache_ptr=kv_flat,
            block_tables_ptr=block_table,
            seq_lens_ptr=seq_lens,
            query_start_len_ptr=query_start_loc,
            sinks_ptr=sinks_f32,
            scale=scale_for_kernel,
            num_query_heads=Hq,
            num_queries_per_kv=kv_group_size,
            block_table_stride=block_table.stride(0),
            query_stride_0=q_for_kernel.stride(0),
            query_stride_1=q_for_kernel.stride(1),
            output_stride_0=output.stride(0),
            output_stride_1=output.stride(1),
            stride_cache_block=kv_cache.stride(0),
            stride_cache_pos=kv_cache.stride(1),
            stride_cache_head=kv_cache.stride(2),
            BLOCK_SIZE=block_size,
            TILE_SIZE=tile_size,
            HEAD_SIZE=D,
            HEAD_SIZE_PADDED=BLOCK_D,
            BLOCK_Q=BLOCK_Q,
            BLOCK_M=BLOCK_M,
            num_seqs=num_seqs,
            K_SCALES_OFFSET=K_SCALES_OFFSET,
            V_CODES_OFFSET=V_CODES_OFFSET,
            V_SCALES_OFFSET=V_SCALES_OFFSET,
            GROUP_SIZE_C=gs,
            N_GROUPS_C=N_GROUPS_C,
            UE8M0_BIAS_C=UE8M0_BIAS,
            USE_SINKS=1 if use_sinks else 0,
            SLIDING_WINDOW=int(sliding_window)
            if sliding_window and sliding_window > 0
            else 0,
            num_warps=4,
            num_stages=num_stages_2d,
        )
        return output

    # 3D split-KV path
    if num_kv_splits is None:
        num_kv_splits = 16
    if num_kv_splits < 1:
        num_kv_splits = 1
    if num_kv_splits & (num_kv_splits - 1) != 0:
        num_kv_splits = 1 << (num_kv_splits.bit_length() - 1)
    max_possible_splits = max(1, (max_seq_len_hint + tile_size - 1) // tile_size)
    num_segments = max(1, min(num_kv_splits, max_possible_splits))

    segm_output = torch.empty(
        (num_tokens, Hq, num_segments, BLOCK_D),
        dtype=torch.float32,
        device=device,
    )
    segm_max = torch.empty(
        (num_tokens, Hq, num_segments),
        dtype=torch.float32,
        device=device,
    )
    segm_expsum = torch.empty(
        (num_tokens, Hq, num_segments),
        dtype=torch.float32,
        device=device,
    )

    kernel_ultraquant_unified_attention_3d[(total_num_q_blocks, Hk, num_segments)](
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        query_ptr=q_for_kernel,
        KV_cache_ptr=kv_flat,
        block_tables_ptr=block_table,
        seq_lens_ptr=seq_lens,
        query_start_len_ptr=query_start_loc,
        sinks_ptr=sinks_f32,
        scale=scale_for_kernel,
        num_query_heads=Hq,
        num_queries_per_kv=kv_group_size,
        block_table_stride=block_table.stride(0),
        query_stride_0=q_for_kernel.stride(0),
        query_stride_1=q_for_kernel.stride(1),
        stride_cache_block=kv_cache.stride(0),
        stride_cache_pos=kv_cache.stride(1),
        stride_cache_head=kv_cache.stride(2),
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE=D,
        HEAD_SIZE_PADDED=BLOCK_D,
        BLOCK_Q=BLOCK_Q,
        BLOCK_M=BLOCK_M,
        num_seqs=num_seqs,
        NUM_SEGMENTS_PER_SEQ=num_segments,
        K_SCALES_OFFSET=K_SCALES_OFFSET,
        V_CODES_OFFSET=V_CODES_OFFSET,
        V_SCALES_OFFSET=V_SCALES_OFFSET,
        GROUP_SIZE_C=gs,
        N_GROUPS_C=N_GROUPS_C,
        UE8M0_BIAS_C=UE8M0_BIAS,
        USE_SINKS=1 if use_sinks else 0,
        SLIDING_WINDOW=int(sliding_window)
        if sliding_window and sliding_window > 0
        else 0,
        num_warps=2,
        num_stages=num_stages_3d,
    )

    # Reduce split-KV partials with the shared vectorized reducer.
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
        HEAD_SIZE_PADDED=BLOCK_D,
        query_start_len_ptr=query_start_loc,
        BLOCK_Q=BLOCK_Q,
        NUM_SEGMENTS_PER_SEQ=num_segments,
        USE_FP8=False,
    )

    return output
