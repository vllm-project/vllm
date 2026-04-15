# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT4 per-token-head KV cache quantization backend.

Format
------
- Storage: 2 × int4 packed per uint8, ``head_size // 2`` bytes per (token, head).
- Range:   asymmetric ``[min, max]`` mapped to ``[0..15]`` with the 4-bit
           zero-point steganographed into the low 4 mantissa bits of the
           float32 scale (zero memory overhead).
- Q transform: single Randomized Hadamard Transform (RHT) before attention,
               inverse RHT on the output.

Read path
---------
``S += (Q·K_uint - zp·sum(Q)) × scale``  via 2-way split dot.
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
from vllm.v1.attention.ops.triton_quant_kv._hadamard import single_rht
from vllm.v1.attention.ops.triton_quant_kv._packed import (
    pack_int4_nibbles,
    unpack_int4_nibbles,
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
# Reshape kernel: pack two 4-bit values per byte, steganograph zp in scale
# ---------------------------------------------------------------------------
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
    HALF_HEAD_PADDED: tl.constexpr,
):
    """Asymmetric INT4 quantization with zero-point steganography."""
    tok = tl.program_id(0)
    head = tl.program_id(1)

    slot = tl.load(slot_mapping_ptr + tok).to(tl.int64)
    if slot < 0:
        return

    blk = slot // block_size
    slot_in_blk = slot % block_size

    half_offs = tl.arange(0, HALF_HEAD_PADDED)
    even_offs = half_offs * 2
    odd_offs = half_offs * 2 + 1

    # ---- Key ----------------------------------------------------------------
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

    # ---- Value (same algorithm) --------------------------------------------
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


# ---------------------------------------------------------------------------
# Attention kernel: split-dot, asymmetric, fused 2D/3D via IS_3D constexpr.
# ---------------------------------------------------------------------------
@triton.jit
def _attn_int4(
    # Output destinations.  In 2D mode we write the final result into
    # ``output_ptr``; in 3D mode we write per-segment partials into the
    # three ``segm_*`` tensors and ``output_ptr`` is unused (callers may
    # pass any non-null pointer).
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
    HALF_HEAD_PADDED: tl.constexpr,
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

    # ---- Split-Q prologue: load Q in two interleaved halves ---------------
    half_offs = tl.arange(0, HALF_HEAD_PADDED)
    even_head_offs = half_offs * 2
    odd_head_offs = half_offs * 2 + 1
    even_head_mask = tl.where(even_head_offs < HEAD_SIZE, 1, 0).to(tl.int1)
    odd_head_mask = tl.where(odd_head_offs < HEAD_SIZE, 1, 0).to(tl.int1)
    half_dim_mask = tl.where(half_offs < HEAD_SIZE // 2, 1, 0).to(tl.int1)
    q_base = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
    )
    q_mask = query_mask_0[:, None] & query_mask_1[:, None]
    Q_even = tl.load(
        query_ptr + q_base + even_head_offs[None, :],
        mask=even_head_mask[None, :] & q_mask,
        other=0.0,
    ).to(tl.float32)
    Q_odd = tl.load(
        query_ptr + q_base + odd_head_offs[None, :],
        mask=odd_head_mask[None, :] & q_mask,
        other=0.0,
    ).to(tl.float32)
    Q_sum = tl.sum(Q_even, axis=1) + tl.sum(Q_odd, axis=1)

    block_table_offset = seq_idx * block_table_stride

    # 2D loads sinks unconditionally; 3D only on segm_idx == 0 because
    # ``reduce_segments`` adds the sink contribution once.
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
    acc_even = tl.zeros([BLOCK_M, HALF_HEAD_PADDED], dtype=tl.float32)
    acc_odd = tl.zeros([BLOCK_M, HALF_HEAD_PADDED], dtype=tl.float32)

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
            + half_offs[:, None] * stride_k_cache_3
            + slot_in_blk[None, :] * stride_k_cache_1
        )
        K_packed = tl.load(
            key_cache_ptr + k_off,
            mask=half_dim_mask[:, None] & tile_mask[None, :],
            other=0,
        )
        K_lo_u, K_hi_u = unpack_int4_nibbles(K_packed)
        K_lo = K_lo_u.to(Q_even.dtype)
        K_hi = K_hi_u.to(Q_odd.dtype)

        v_off = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + half_offs[None, :] * stride_v_cache_3
            + slot_in_blk[:, None] * stride_v_cache_1
        )
        V_packed = tl.load(
            value_cache_ptr + v_off,
            mask=half_dim_mask[None, :] & tile_mask[:, None],
            other=0,
        )
        V_lo_u, V_hi_u = unpack_int4_nibbles(V_packed)
        V_lo = V_lo_u.to(Q_even.dtype)
        V_hi = V_hi_u.to(Q_odd.dtype)

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

        # Steganography: extract zp from low 4 bits of float32 mantissa.
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
            mm_prefix_range_ptr,
            SLIDING_WINDOW,
            USE_MM_PREFIX,
            MAX_MM_RANGES,
        )

        # Score: split-dot with asymmetric correction.  Fused softmax_scale
        # with per-(token, head) k_scale into one mul.
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        raw_dot = tl.dot(Q_even, K_lo) + tl.dot(Q_odd, K_hi)
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
        acc_even = acc_even * alpha[:, None]
        acc_odd = acc_odd * alpha[:, None]

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            sw_mask = (context_len + qpos_lo - seq_offset) < SLIDING_WINDOW
            V_lo = tl.where(sw_mask[:, None], V_lo, 0.0)
            V_hi = tl.where(sw_mask[:, None], V_hi, 0.0)
        # Fuse v per-(token, head) scale into P; subtract the v zero-point
        # contribution once for both halves.
        P_v = (P * v_token_head_scales[None, :]).to(V_lo.dtype)
        Pv_zp_sum = tl.sum(P_v * v_zp[None, :], axis=1)
        acc_even += tl.dot(P_v, V_lo) - Pv_zp_sum[:, None]
        acc_odd += tl.dot(P_v, V_hi) - Pv_zp_sum[:, None]

    out_mask = query_mask_0[:, None] & query_mask_1[:, None]
    if IS_3D:
        # Per-segment partials; finalized by ``reduce_segments``.
        segm_base = (
            query_offset_0[:, None].to(tl.int64)
            * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
            + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
            + segm_idx * HEAD_SIZE_PADDED
        )
        tl.store(
            segm_output_ptr + segm_base + even_head_offs[None, :],
            acc_even,
            mask=even_head_mask[None, :] & out_mask,
        )
        tl.store(
            segm_output_ptr + segm_base + odd_head_offs[None, :],
            acc_odd,
            mask=odd_head_mask[None, :] & out_mask,
        )
        segm_offset = (
            query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
            + query_offset_1 * NUM_SEGMENTS_PER_SEQ
            + segm_idx
        )
        tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
        tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)
    else:
        acc_even = acc_even / L[:, None]
        acc_odd = acc_odd / L[:, None]
        if USE_FP8:
            out_s = tl.load(out_scale)
            acc_even = tl.clamp(acc_even * out_s, FP8_MIN, FP8_MAX)
            acc_odd = tl.clamp(acc_odd * out_s, FP8_MIN, FP8_MAX)
        out_base = (
            query_offset_0[:, None] * output_stride_0
            + query_offset_1[:, None] * output_stride_1
        )
        tl.store(
            output_ptr + out_base + even_head_offs[None, :],
            acc_even,
            mask=even_head_mask[None, :] & out_mask,
        )
        tl.store(
            output_ptr + out_base + odd_head_offs[None, :],
            acc_odd,
            mask=odd_head_mask[None, :] & out_mask,
        )


# ---------------------------------------------------------------------------
# Backend implementation
# ---------------------------------------------------------------------------
class Int4PerTokenHeadBackend(QuantKVBackend):
    """KV cache backend for ``KVQuantMode.INT4_PER_TOKEN_HEAD``."""

    mode = KVQuantMode.INT4_PER_TOKEN_HEAD
    packing_factor = 2  # 2 × int4 per byte
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
            "INT4 per-token-head requires k_scale_cache / v_scale_cache"
        )
        # RHT pre-rotation gaussianizes data → better quantization
        key = single_rht(key.float()).to(key.dtype)
        value = single_rht(value.float()).to(value.dtype)

        num_tokens, num_kv_heads, head_size = key.shape
        head_size_v = value.shape[2]
        assert head_size % 2 == 0 and head_size_v % 2 == 0
        half_head_padded = triton.next_power_of_2(max(head_size, head_size_v) // 2)
        if current_platform.is_rocm() or current_platform.is_xpu():
            num_warps = 4
        else:
            num_warps = min(16, max(1, half_head_padded // 32))

        _reshape_cache_int4_kernel[(num_tokens, num_kv_heads)](
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
            HALF_HEAD_PADDED=half_head_padded,
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

        # ---- Q rotation: single RHT, scale absorbs the d factor ------------
        q_orig_dtype = q.dtype
        q = single_rht(q.float()).to(q_orig_dtype)
        head_size = q.shape[2]
        softmax_scale = softmax_scale / head_size

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

        # Same kernel handles both modes; only the launch grid +
        # IS_3D constexpr + a couple of pointer/integer args differ.
        # 3D never reads ``output_ptr`` and 2D never reads the segm
        # tensors, but Triton needs a non-null pointer everywhere; reuse
        # ``out`` as the placeholder for the unused side.
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

        _attn_int4[grid](
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
            HALF_HEAD_PADDED=triton.next_power_of_2(head_size) // 2,
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

        # ---- Inverse RHT on output -----------------------------------------
        out_f = single_rht(out.float(), inverse=True) / head_size
        out.copy_(out_f.to(q_orig_dtype))


register(Int4PerTokenHeadBackend())
