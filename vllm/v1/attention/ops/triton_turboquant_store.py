# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernels for TurboQuant KV store.

Three kernels:
1. _tq_fused_store_fp8: FP8 key scatter + value uniform quantization.
2. _tq_fused_store_mse_m1: Fused MSE store for small token-head counts.
3. _tq_fused_store_mse_m16: Batched MSE store for larger token-head counts.

The launcher `triton_turboquant_store` selects the appropriate kernel.
"""

import math

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_turboquant_decode import _use_fp8_e4b15

# ═══════════════════════════════════════════════════════════════════════
# Shared: value uniform quantization + pack + scale/zero store
# ═══════════════════════════════════════════════════════════════════════


@triton.jit
def _store_quantized_value(
    Value_ptr,
    KV_cache_ptr,
    base,  # pid * D offset into Value_ptr
    value_data_base,
    value_scale_base,
    value_zero_base,
    d_offs,  # tl.arange(0, BLOCK_D)
    d_mask,  # d_offs < D
    D: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_VAL: tl.constexpr,
    BLOCK_GRP: tl.constexpr,
):
    """Uniform quantization of values to VQB bits, pack, and store with scale/zero."""

    if VQB == 3:
        val_vec = tl.load(Value_ptr + base + d_offs, mask=d_mask, other=0.0).to(
            tl.float32
        )
        val_min = tl.min(tl.where(d_mask, val_vec, float("inf")), axis=0)
        val_max = tl.max(tl.where(d_mask, val_vec, -float("inf")), axis=0)
        v_scale = (val_max - val_min) / 7.0
        v_scale = tl.where(v_scale > 1e-8, v_scale, 1e-8)

        q_vals = tl.minimum(
            tl.maximum(((val_vec - val_min) / v_scale + 0.5).to(tl.int32), 0), 7
        )

        grp_offs = tl.arange(0, BLOCK_GRP)
        grp_mask = grp_offs < (D // 8)
        q_grp = tl.reshape(q_vals, [BLOCK_GRP, 8])
        shifts_3bit = tl.arange(0, 8) * 3
        packed_24 = tl.sum(q_grp << shifts_3bit[None, :], axis=1)
        b0 = (packed_24 & 0xFF).to(tl.uint8)
        b1 = ((packed_24 >> 8) & 0xFF).to(tl.uint8)
        b2 = ((packed_24 >> 16) & 0xFF).to(tl.uint8)
        tl.store(
            KV_cache_ptr + value_data_base + grp_offs * 3,
            b0,
            mask=grp_mask,
        )
        tl.store(
            KV_cache_ptr + value_data_base + grp_offs * 3 + 1,
            b1,
            mask=grp_mask,
        )
        tl.store(
            KV_cache_ptr + value_data_base + grp_offs * 3 + 2,
            b2,
            mask=grp_mask,
        )

        sc_u16 = v_scale.to(tl.float16).to(tl.uint16, bitcast=True)
        zr_u16 = val_min.to(tl.float16).to(tl.uint16, bitcast=True)
        sc_ptr = (KV_cache_ptr + value_scale_base).to(tl.pointer_type(tl.uint16))
        zr_ptr = (KV_cache_ptr + value_zero_base).to(tl.pointer_type(tl.uint16))
        tl.store(sc_ptr, sc_u16)
        tl.store(zr_ptr, zr_u16)

    else:  # VQB == 4
        val_vec = tl.load(Value_ptr + base + d_offs, mask=d_mask, other=0.0).to(
            tl.float32
        )
        val_min = tl.min(tl.where(d_mask, val_vec, float("inf")), axis=0)
        val_max = tl.max(tl.where(d_mask, val_vec, -float("inf")), axis=0)
        v_scale = (val_max - val_min) / 15.0
        v_scale = tl.where(v_scale > 1e-8, v_scale, 1e-8)

        # Quantize all D elements from register (no re-load)
        q_all = tl.minimum(
            tl.maximum(((val_vec - val_min) / v_scale + 0.5).to(tl.int32), 0), 15
        )
        # Reshape to pairs and pack two 4-bit values per byte
        q_pairs = tl.reshape(q_all, [BLOCK_D // 2, 2])
        shifts_4 = tl.arange(0, 2) * 4
        packed_val = tl.sum((q_pairs & 0xF) << shifts_4[None, :], axis=1).to(tl.uint8)
        val_offs = tl.arange(0, BLOCK_D // 2)
        val_mask = val_offs < VAL_DATA_BYTES
        tl.store(
            KV_cache_ptr + value_data_base + val_offs,
            packed_val,
            mask=val_mask,
        )

        sc_u16 = v_scale.to(tl.float16).to(tl.uint16, bitcast=True)
        zr_u16 = val_min.to(tl.float16).to(tl.uint16, bitcast=True)
        sc_ptr = (KV_cache_ptr + value_scale_base).to(tl.pointer_type(tl.uint16))
        zr_ptr = (KV_cache_ptr + value_zero_base).to(tl.pointer_type(tl.uint16))
        tl.store(sc_ptr, sc_u16)
        tl.store(zr_ptr, zr_u16)


# ═══════════════════════════════════════════════════════════════════════
# FP8 key store + value uniform quantization
# ═══════════════════════════════════════════════════════════════════════


@triton.jit
def _tq_fused_store_fp8(
    Key_ptr,  # [NH, D] float16/bfloat16 — raw keys
    Value_ptr,  # [NH, D] float16/bfloat16 — raw values
    KV_cache_ptr,  # [total_bytes] uint8 (flattened view)
    Slot_mapping_ptr,  # [N] int32 — per-token slot indices
    # Cache strides (for computing byte offsets)
    stride_cache_block: tl.constexpr,
    stride_cache_head: tl.constexpr,
    # Dimensions
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    # Value quantization
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    # Packing block sizes
    BLOCK_VAL: tl.constexpr,
    BLOCK_GRP: tl.constexpr = 16,
    FP8_E4B15: tl.constexpr = 0,  # 1 = e4b15 (Ampere/Ada), 0 = e4nv (Hopper+)
):
    """FP8 key cast+scatter + value uniform quantization."""
    pid = tl.program_id(0)
    token_idx = pid // H
    head_idx = pid % H

    slot = tl.load(Slot_mapping_ptr + token_idx)
    if slot < 0:
        return
    blk = (slot // BLOCK_SIZE).to(tl.int64)
    off = (slot % BLOCK_SIZE).to(tl.int64)
    head_idx_i64 = tl.cast(head_idx, tl.int64)
    record_base = blk * stride_cache_block + head_idx_i64 * stride_cache_head
    key_data_base = record_base + off * D
    value_data_plane = record_base + BLOCK_SIZE * D
    value_scale_plane = value_data_plane + BLOCK_SIZE * VAL_DATA_BYTES
    value_zero_plane = value_scale_plane + BLOCK_SIZE * 2
    value_data_base = value_data_plane + off * VAL_DATA_BYTES
    value_scale_base = value_scale_plane + off * 2
    value_zero_base = value_zero_plane + off * 2

    base = pid * D

    # ── FP8 KEY: cast to FP8 in-kernel and store ─────────────────
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    k_vals = tl.load(Key_ptr + base + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    k_fp8 = k_vals.to(tl.float8e4b15) if FP8_E4B15 else k_vals.to(tl.float8e4nv)
    k_bytes = k_fp8.to(tl.uint8, bitcast=True)
    tl.store(KV_cache_ptr + key_data_base + d_offs, k_bytes, mask=d_mask)

    # ── VALUE QUANTIZE + PACK ───────────────────────────────────────
    _store_quantized_value(
        Value_ptr,
        KV_cache_ptr,
        base,
        value_data_base,
        value_scale_base,
        value_zero_base,
        d_offs,
        d_mask,
        D=D,
        VQB=VQB,
        VAL_DATA_BYTES=VAL_DATA_BYTES,
        BLOCK_D=BLOCK_D,
        BLOCK_VAL=BLOCK_VAL,
        BLOCK_GRP=BLOCK_GRP,
    )


# ═══════════════════════════════════════════════════════════════════════
# Fused MSE store: normalize + rotate + bucketize + pack + metadata store
# ═══════════════════════════════════════════════════════════════════════


@triton.jit
def _tq_fused_store_mse_m1(
    Key_ptr,
    Value_ptr,
    PiT_mma_ptr,
    Midpoints_ptr,
    KV_cache_ptr,
    Slot_mapping_ptr,
    stride_cache_block: tl.constexpr,
    stride_cache_head: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    MSE_BITS: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    BLOCK_GRP: tl.constexpr,
    M_PAD: tl.constexpr = 16,
):
    pid = tl.program_id(0)
    token_idx = pid // H
    head_idx = pid % H

    slot = tl.load(Slot_mapping_ptr + token_idx)
    if slot < 0:
        return
    block_idx = (slot // BLOCK_SIZE).to(tl.int64)
    position = (slot % BLOCK_SIZE).to(tl.int64)
    record_base = (
        block_idx * stride_cache_block + tl.cast(head_idx, tl.int64) * stride_cache_head
    )
    key_data_base = record_base + position * MSE_BYTES
    value_data_plane = record_base + BLOCK_SIZE * MSE_BYTES
    key_norm_plane = value_data_plane + BLOCK_SIZE * VAL_DATA_BYTES
    value_scale_plane = key_norm_plane + BLOCK_SIZE * 2
    value_zero_plane = value_scale_plane + BLOCK_SIZE * 2
    value_data_base = value_data_plane + position * VAL_DATA_BYTES
    key_norm_base = key_norm_plane + position * 2
    value_scale_base = value_scale_plane + position * 2
    value_zero_base = value_zero_plane + position * 2

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    input_base = pid * D
    key = tl.load(Key_ptr + input_base + d_offs, mask=d_mask, other=0.0)
    value = tl.load(Value_ptr + input_base + d_offs, mask=d_mask, other=0.0)

    key_fp32 = key.to(tl.float32)
    norm = tl.sqrt(tl.sum(tl.where(d_mask, key_fp32 * key_fp32, 0.0)))
    normalized = key_fp32 / (norm + 1e-8)
    rows = tl.arange(0, M_PAD)
    normalized_2d = tl.where(
        (rows[:, None] == 0) & d_mask[None, :],
        normalized[None, :],
        0.0,
    ).to(tl.float16)
    pit = tl.load(
        PiT_mma_ptr + d_offs[:, None] * D + d_offs[None, :],
        mask=d_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    rotated = tl.sum(tl.dot(normalized_2d, pit), axis=0)

    lo = tl.zeros([BLOCK_D], dtype=tl.int32)
    hi = tl.full([BLOCK_D], N_CENTROIDS - 1, dtype=tl.int32)
    for _ in range(MSE_BITS):
        mid = (lo + hi) >> 1
        midpoint = tl.load(
            Midpoints_ptr + tl.minimum(mid, N_CENTROIDS - 2),
            mask=d_mask,
            other=0.0,
        )
        ge = rotated >= midpoint
        lo = tl.where(ge, mid + 1, lo)
        hi = tl.where(ge, hi, mid)
    key_idx = tl.minimum(lo, N_CENTROIDS - 1)

    if MSE_BITS == 4:
        key_pairs = tl.reshape(key_idx, [BLOCK_D // 2, 2])
        shifts = tl.arange(0, 2) * 4
        key_packed = tl.sum((key_pairs & 0xF) << shifts[None, :], axis=1).to(tl.uint8)
        key_offs = tl.arange(0, BLOCK_D // 2)
        tl.store(
            KV_cache_ptr + key_data_base + key_offs,
            key_packed,
            mask=key_offs < MSE_BYTES,
        )
    else:
        groups = tl.arange(0, BLOCK_GRP)
        group_mask = groups < (D // 8)
        key_groups = tl.reshape(key_idx, [BLOCK_GRP, 8])
        packed = tl.sum((key_groups & 0x7) << (tl.arange(0, 8) * 3)[None, :], axis=1)
        tl.store(
            KV_cache_ptr + key_data_base + groups * 3,
            (packed & 0xFF).to(tl.uint8),
            mask=group_mask,
        )
        tl.store(
            KV_cache_ptr + key_data_base + groups * 3 + 1,
            ((packed >> 8) & 0xFF).to(tl.uint8),
            mask=group_mask,
        )
        tl.store(
            KV_cache_ptr + key_data_base + groups * 3 + 2,
            ((packed >> 16) & 0xFF).to(tl.uint8),
            mask=group_mask,
        )

    norm_u16 = norm.to(tl.float16).to(tl.uint16, bitcast=True)
    norm_ptr = (KV_cache_ptr + key_norm_base).to(tl.pointer_type(tl.uint16))
    tl.store(norm_ptr, norm_u16)

    value_fp32 = value.to(tl.float32)
    value_min = tl.min(tl.where(d_mask, value_fp32, float("inf")), axis=0)
    value_max = tl.max(tl.where(d_mask, value_fp32, -float("inf")), axis=0)
    levels = 7.0 if VQB == 3 else 15.0
    value_scale = tl.maximum((value_max - value_min) / levels, 1e-8)
    value_idx = tl.minimum(
        tl.maximum(((value_fp32 - value_min) / value_scale + 0.5).to(tl.int32), 0),
        7 if VQB == 3 else 15,
    )

    if VQB == 4:
        value_pairs = tl.reshape(value_idx, [BLOCK_D // 2, 2])
        shifts = tl.arange(0, 2) * 4
        value_packed = tl.sum((value_pairs & 0xF) << shifts[None, :], axis=1).to(
            tl.uint8
        )
        value_offs = tl.arange(0, BLOCK_D // 2)
        tl.store(
            KV_cache_ptr + value_data_base + value_offs,
            value_packed,
            mask=value_offs < VAL_DATA_BYTES,
        )
    else:
        groups = tl.arange(0, BLOCK_GRP)
        group_mask = groups < (D // 8)
        value_groups = tl.reshape(value_idx, [BLOCK_GRP, 8])
        packed = tl.sum(value_groups << (tl.arange(0, 8) * 3)[None, :], axis=1)
        tl.store(
            KV_cache_ptr + value_data_base + groups * 3,
            (packed & 0xFF).to(tl.uint8),
            mask=group_mask,
        )
        tl.store(
            KV_cache_ptr + value_data_base + groups * 3 + 1,
            ((packed >> 8) & 0xFF).to(tl.uint8),
            mask=group_mask,
        )
        tl.store(
            KV_cache_ptr + value_data_base + groups * 3 + 2,
            ((packed >> 16) & 0xFF).to(tl.uint8),
            mask=group_mask,
        )

    scale_u16 = value_scale.to(tl.float16).to(tl.uint16, bitcast=True)
    zero_u16 = value_min.to(tl.float16).to(tl.uint16, bitcast=True)
    scale_ptr = (KV_cache_ptr + value_scale_base).to(tl.pointer_type(tl.uint16))
    zero_ptr = (KV_cache_ptr + value_zero_base).to(tl.pointer_type(tl.uint16))
    tl.store(scale_ptr, scale_u16)
    tl.store(zero_ptr, zero_u16)


@triton.jit
def _tq_fused_store_mse_m16(
    Key_ptr,
    Value_ptr,
    PiT_mma_ptr,
    Midpoints_ptr,
    KV_cache_ptr,
    Slot_mapping_ptr,
    stride_cache_block: tl.constexpr,
    stride_cache_head: tl.constexpr,
    NH,
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    MSE_BITS: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    BLOCK_GRP: tl.constexpr,
    BLOCK_M: tl.constexpr = 16,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < NH
    token_idx = rows // H
    head_idx = rows % H
    slots = tl.load(Slot_mapping_ptr + token_idx, mask=row_mask, other=-1)
    store_mask = row_mask & (slots >= 0)

    block_idx = (slots // BLOCK_SIZE).to(tl.int64)
    position = (slots % BLOCK_SIZE).to(tl.int64)
    record_base = (
        block_idx * stride_cache_block + head_idx.to(tl.int64) * stride_cache_head
    )
    key_data_base = record_base + position * MSE_BYTES
    value_data_plane = record_base + BLOCK_SIZE * MSE_BYTES
    key_norm_plane = value_data_plane + BLOCK_SIZE * VAL_DATA_BYTES
    value_scale_plane = key_norm_plane + BLOCK_SIZE * 2
    value_zero_plane = value_scale_plane + BLOCK_SIZE * 2
    value_data_base = value_data_plane + position * VAL_DATA_BYTES
    key_norm_base = key_norm_plane + position * 2
    value_scale_base = value_scale_plane + position * 2
    value_zero_base = value_zero_plane + position * 2

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    input_offsets = rows[:, None] * D + d_offs[None, :]
    input_mask = store_mask[:, None] & d_mask[None, :]
    key = tl.load(Key_ptr + input_offsets, mask=input_mask, other=0.0)
    value = tl.load(Value_ptr + input_offsets, mask=input_mask, other=0.0)

    key_fp32 = key.to(tl.float32)
    norm = tl.sqrt(tl.sum(tl.where(input_mask, key_fp32 * key_fp32, 0.0), axis=1))
    normalized = (key_fp32 / (norm[:, None] + 1e-8)).to(tl.float16)
    pit = tl.load(
        PiT_mma_ptr + d_offs[:, None] * D + d_offs[None, :],
        mask=d_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    rotated = tl.dot(normalized, pit)

    lo = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.int32)
    hi = tl.full([BLOCK_M, BLOCK_D], N_CENTROIDS - 1, dtype=tl.int32)
    for _ in range(MSE_BITS):
        mid = (lo + hi) >> 1
        midpoint = tl.load(
            Midpoints_ptr + tl.minimum(mid, N_CENTROIDS - 2),
            mask=input_mask,
            other=0.0,
        )
        ge = rotated >= midpoint
        lo = tl.where(ge, mid + 1, lo)
        hi = tl.where(ge, hi, mid)
    key_idx = tl.minimum(lo, N_CENTROIDS - 1)

    if MSE_BITS == 4:
        key_pairs = tl.reshape(key_idx, [BLOCK_M, BLOCK_D // 2, 2])
        shifts = tl.arange(0, 2) * 4
        key_packed = tl.sum((key_pairs & 0xF) << shifts[None, None, :], axis=2).to(
            tl.uint8
        )
        key_offs = tl.arange(0, BLOCK_D // 2)
        tl.store(
            KV_cache_ptr + key_data_base[:, None] + key_offs[None, :],
            key_packed,
            mask=store_mask[:, None] & (key_offs[None, :] < MSE_BYTES),
        )
    else:
        groups = tl.arange(0, BLOCK_GRP)
        group_mask = store_mask[:, None] & (groups[None, :] < (D // 8))
        key_groups = tl.reshape(key_idx, [BLOCK_M, BLOCK_GRP, 8])
        packed = tl.sum(
            (key_groups & 0x7) << (tl.arange(0, 8) * 3)[None, None, :],
            axis=2,
        )
        tl.store(
            KV_cache_ptr + key_data_base[:, None] + groups[None, :] * 3,
            (packed & 0xFF).to(tl.uint8),
            mask=group_mask,
        )
        tl.store(
            KV_cache_ptr + key_data_base[:, None] + groups[None, :] * 3 + 1,
            ((packed >> 8) & 0xFF).to(tl.uint8),
            mask=group_mask,
        )
        tl.store(
            KV_cache_ptr + key_data_base[:, None] + groups[None, :] * 3 + 2,
            ((packed >> 16) & 0xFF).to(tl.uint8),
            mask=group_mask,
        )

    norm_u16 = norm.to(tl.float16).to(tl.uint16, bitcast=True)
    norm_ptr = (KV_cache_ptr + key_norm_base).to(tl.pointer_type(tl.uint16))
    tl.store(norm_ptr, norm_u16, mask=store_mask)

    value_fp32 = value.to(tl.float32)
    value_min = tl.min(tl.where(input_mask, value_fp32, float("inf")), axis=1)
    value_max = tl.max(tl.where(input_mask, value_fp32, -float("inf")), axis=1)
    levels = 7.0 if VQB == 3 else 15.0
    value_scale = tl.maximum((value_max - value_min) / levels, 1e-8)
    value_idx = tl.minimum(
        tl.maximum(
            ((value_fp32 - value_min[:, None]) / value_scale[:, None] + 0.5).to(
                tl.int32
            ),
            0,
        ),
        7 if VQB == 3 else 15,
    )

    if VQB == 4:
        value_pairs = tl.reshape(value_idx, [BLOCK_M, BLOCK_D // 2, 2])
        shifts = tl.arange(0, 2) * 4
        value_packed = tl.sum((value_pairs & 0xF) << shifts[None, None, :], axis=2).to(
            tl.uint8
        )
        value_offs = tl.arange(0, BLOCK_D // 2)
        tl.store(
            KV_cache_ptr + value_data_base[:, None] + value_offs[None, :],
            value_packed,
            mask=store_mask[:, None] & (value_offs[None, :] < VAL_DATA_BYTES),
        )
    else:
        groups = tl.arange(0, BLOCK_GRP)
        group_mask = store_mask[:, None] & (groups[None, :] < (D // 8))
        value_groups = tl.reshape(value_idx, [BLOCK_M, BLOCK_GRP, 8])
        packed = tl.sum(value_groups << (tl.arange(0, 8) * 3)[None, None, :], axis=2)
        tl.store(
            KV_cache_ptr + value_data_base[:, None] + groups[None, :] * 3,
            (packed & 0xFF).to(tl.uint8),
            mask=group_mask,
        )
        tl.store(
            KV_cache_ptr + value_data_base[:, None] + groups[None, :] * 3 + 1,
            ((packed >> 8) & 0xFF).to(tl.uint8),
            mask=group_mask,
        )
        tl.store(
            KV_cache_ptr + value_data_base[:, None] + groups[None, :] * 3 + 2,
            ((packed >> 16) & 0xFF).to(tl.uint8),
            mask=group_mask,
        )

    scale_u16 = value_scale.to(tl.float16).to(tl.uint16, bitcast=True)
    zero_u16 = value_min.to(tl.float16).to(tl.uint16, bitcast=True)
    scale_ptr = (KV_cache_ptr + value_scale_base).to(tl.pointer_type(tl.uint16))
    zero_ptr = (KV_cache_ptr + value_zero_base).to(tl.pointer_type(tl.uint16))
    tl.store(scale_ptr, scale_u16, mask=store_mask)
    tl.store(zero_ptr, zero_u16, mask=store_mask)


# ═══════════════════════════════════════════════════════════════════════
# Launcher
# ═══════════════════════════════════════════════════════════════════════


_PIT_MMA_CACHE: dict[tuple[int, int, tuple[int, ...]], torch.Tensor] = {}
_BLOCK_M_THRESHOLD = 256
_BLOCK_M = 16


def _get_pit_mma(PiT: torch.Tensor) -> torch.Tensor:
    if PiT.dtype == torch.float16:
        return PiT.contiguous()
    key = (PiT.data_ptr(), PiT.device.index or 0, tuple(PiT.shape))
    cached = _PIT_MMA_CACHE.get(key)
    if cached is None or cached.device != PiT.device:
        cached = PiT.to(torch.float16).contiguous()
        _PIT_MMA_CACHE[key] = cached
    return cached


def triton_turboquant_store(
    key: torch.Tensor,  # [N, H, D] — raw keys (post-RoPE)
    value: torch.Tensor,  # [N, H, D] — raw values
    kv_cache: torch.Tensor,  # [num_blocks, Hk, 1, page_record_size] uint8
    slot_mapping: torch.Tensor,  # [N] int32
    PiT: torch.Tensor,  # [D, D] float32
    midpoints: torch.Tensor,  # [n_centroids-1] float32
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    key_fp8: bool = False,
):
    """Launch TQ store kernel (FP8 or MSE path)."""
    N, H, D = key.shape
    NH = N * H
    BLOCK_D = triton.next_power_of_2(D)
    mse_bytes = math.ceil(D * mse_bits / 8)
    n_centroids = 2**mse_bits

    val_data_bytes = math.ceil(D * value_quant_bits / 8)
    per_token_bytes = key_packed_size + val_data_bytes + 4
    aligned_token_bytes = per_token_bytes + per_token_bytes % 2
    record_size = kv_cache.shape[-1]
    if kv_cache.ndim != 4 or kv_cache.shape[2] != 1:
        raise ValueError(
            "TurboQuant cache must have shape "
            "[num_blocks, num_kv_heads, 1, page_record_size]"
        )
    if kv_cache.shape[1] != H:
        raise ValueError(
            f"TurboQuant cache has {kv_cache.shape[1]} heads, expected {H}"
        )
    if record_size % aligned_token_bytes:
        raise ValueError(
            "TurboQuant page record is not divisible by its per-token payload: "
            f"{record_size=} {aligned_token_bytes=}"
        )
    block_size = record_size // aligned_token_bytes

    BLOCK_VAL = triton.next_power_of_2(val_data_bytes)

    # Cache strides (element_size=1 for uint8, so stride in bytes = stride())
    stride_block = kv_cache.stride(0)
    stride_head = kv_cache.stride(1)

    block_grp = triton.next_power_of_2(D // 8) if D >= 8 else 1

    # ── FP8 PATH: in-kernel FP8 cast + scatter via fp8 kernel ──
    if key_fp8:
        k_flat = key.reshape(NH, D).contiguous()
        v_flat = value.reshape(NH, D).contiguous()

        fp8_e4b15 = _use_fp8_e4b15(key.device.index or 0)

        grid = (NH,)
        _tq_fused_store_fp8[grid](
            k_flat,
            v_flat,
            kv_cache,
            slot_mapping,
            stride_cache_block=stride_block,
            stride_cache_head=stride_head,
            D=D,
            H=H,
            BLOCK_SIZE=block_size,
            BLOCK_D=BLOCK_D,
            VQB=value_quant_bits,
            VAL_DATA_BYTES=val_data_bytes,
            BLOCK_VAL=BLOCK_VAL,
            BLOCK_GRP=block_grp,
            FP8_E4B15=fp8_e4b15,
            num_warps=4,
            num_stages=1,
        )
        return

    k_flat = key.reshape(NH, D).contiguous()
    v_flat = value.reshape(NH, D).contiguous()
    pit_mma = _get_pit_mma(PiT)
    if NH < _BLOCK_M_THRESHOLD:
        _tq_fused_store_mse_m1[(NH,)](
            k_flat,
            v_flat,
            pit_mma,
            midpoints,
            kv_cache,
            slot_mapping,
            stride_cache_block=stride_block,
            stride_cache_head=stride_head,
            D=D,
            H=H,
            BLOCK_SIZE=block_size,
            BLOCK_D=BLOCK_D,
            MSE_BYTES=mse_bytes,
            VQB=value_quant_bits,
            VAL_DATA_BYTES=val_data_bytes,
            MSE_BITS=mse_bits,
            N_CENTROIDS=n_centroids,
            BLOCK_GRP=block_grp,
            num_warps=4,
            num_stages=1,
        )
    else:
        _tq_fused_store_mse_m16[(triton.cdiv(NH, _BLOCK_M),)](
            k_flat,
            v_flat,
            pit_mma,
            midpoints,
            kv_cache,
            slot_mapping,
            stride_cache_block=stride_block,
            stride_cache_head=stride_head,
            NH=NH,
            D=D,
            H=H,
            BLOCK_SIZE=block_size,
            BLOCK_D=BLOCK_D,
            MSE_BYTES=mse_bytes,
            VQB=value_quant_bits,
            VAL_DATA_BYTES=val_data_bytes,
            MSE_BITS=mse_bits,
            N_CENTROIDS=n_centroids,
            BLOCK_GRP=block_grp,
            BLOCK_M=_BLOCK_M,
            num_warps=4,
            num_stages=3,
        )
