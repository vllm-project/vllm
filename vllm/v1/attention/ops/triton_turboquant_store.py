# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernels for TurboQuant KV store.

Two kernels:
1. _tq_fused_store_fp8: FP8 key scatter + value uniform quantization.
2. _tq_fused_store_mse: Fused binary-search bucketize + MSE index
   packing + value quantization.

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
    slot_base,  # byte offset into KV_cache_ptr for this slot+head
    d_offs,  # tl.arange(0, BLOCK_D)
    d_mask,  # d_offs < D
    D: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_VAL: tl.constexpr,
    BLOCK_GRP: tl.constexpr,
):
    """Uniform quantization of values to VQB bits, pack, and store with scale/zero."""
    val_cache_offset = KPS

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
            KV_cache_ptr + slot_base + val_cache_offset + grp_offs * 3,
            b0,
            mask=grp_mask,
        )
        tl.store(
            KV_cache_ptr + slot_base + val_cache_offset + grp_offs * 3 + 1,
            b1,
            mask=grp_mask,
        )
        tl.store(
            KV_cache_ptr + slot_base + val_cache_offset + grp_offs * 3 + 2,
            b2,
            mask=grp_mask,
        )

        sc_offset = val_cache_offset + VAL_DATA_BYTES
        sc_f16 = v_scale.to(tl.float16)
        sc_u16 = sc_f16.to(tl.uint16, bitcast=True)
        tl.store(KV_cache_ptr + slot_base + sc_offset, (sc_u16 & 0xFF).to(tl.uint8))
        tl.store(
            KV_cache_ptr + slot_base + sc_offset + 1,
            ((sc_u16 >> 8) & 0xFF).to(tl.uint8),
        )
        zr_f16 = val_min.to(tl.float16)
        zr_u16 = zr_f16.to(tl.uint16, bitcast=True)
        tl.store(KV_cache_ptr + slot_base + sc_offset + 2, (zr_u16 & 0xFF).to(tl.uint8))
        tl.store(
            KV_cache_ptr + slot_base + sc_offset + 3,
            ((zr_u16 >> 8) & 0xFF).to(tl.uint8),
        )

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
            KV_cache_ptr + slot_base + val_cache_offset + val_offs,
            packed_val,
            mask=val_mask,
        )

        sc_offset = val_cache_offset + VAL_DATA_BYTES
        sc_f16 = v_scale.to(tl.float16)
        sc_u16 = sc_f16.to(tl.uint16, bitcast=True)
        tl.store(KV_cache_ptr + slot_base + sc_offset, (sc_u16 & 0xFF).to(tl.uint8))
        tl.store(
            KV_cache_ptr + slot_base + sc_offset + 1,
            ((sc_u16 >> 8) & 0xFF).to(tl.uint8),
        )
        zr_f16 = val_min.to(tl.float16)
        zr_u16 = zr_f16.to(tl.uint16, bitcast=True)
        tl.store(KV_cache_ptr + slot_base + sc_offset + 2, (zr_u16 & 0xFF).to(tl.uint8))
        tl.store(
            KV_cache_ptr + slot_base + sc_offset + 3,
            ((zr_u16 >> 8) & 0xFF).to(tl.uint8),
        )


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
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
    # Dimensions
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    # TQ layout
    KPS: tl.constexpr,
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
    slot_base = (
        blk * stride_cache_block
        + off * stride_cache_pos
        + head_idx_i64 * stride_cache_head
    )

    base = pid * D

    # ── FP8 KEY: cast to FP8 in-kernel and store ─────────────────
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    k_vals = tl.load(Key_ptr + base + d_offs, mask=d_mask, other=0.0)
    k_fp8 = k_vals.to(tl.float8e4b15) if FP8_E4B15 else k_vals.to(tl.float8e4nv)
    k_bytes = k_fp8.to(tl.uint8, bitcast=True)
    tl.store(KV_cache_ptr + slot_base + d_offs, k_bytes, mask=d_mask)

    # ── VALUE QUANTIZE + PACK ───────────────────────────────────────
    _store_quantized_value(
        Value_ptr,
        KV_cache_ptr,
        base,
        slot_base,
        d_offs,
        d_mask,
        D=D,
        KPS=KPS,
        VQB=VQB,
        VAL_DATA_BYTES=VAL_DATA_BYTES,
        BLOCK_D=BLOCK_D,
        BLOCK_VAL=BLOCK_VAL,
        BLOCK_GRP=BLOCK_GRP,
    )


# ═══════════════════════════════════════════════════════════════════════
# Fused MSE store: bucketize + MSE index pack + norm store + value pack
# (eliminates 4 PyTorch kernel launches per layer vs pack-only kernel)
# ═══════════════════════════════════════════════════════════════════════


@triton.jit
def _tq_fused_store_mse(
    # Post-rotation inputs
    Y_ptr,  # [NH, D] float32 — rotated normalized keys (x_hat @ PiT)
    Norms_ptr,  # [NH] float32 — key vector norms (||k||)
    Value_ptr,  # [NH, D] float32 — raw values
    # Quantization tables
    Midpoints_ptr,  # [n_centroids-1] float32
    # Cache and indexing
    KV_cache_ptr,  # [total_bytes] uint8 (flattened view)
    Slot_mapping_ptr,  # [N] int32 — per-token slot indices
    # Cache strides
    stride_cache_block: tl.constexpr,
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
    # Dimensions
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    # TQ layout
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    # Value quantization
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    # Packing block sizes
    BLOCK_VAL: tl.constexpr,
    # MSE params
    MSE_BITS: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    BLOCK_GRP: tl.constexpr = 16,
):
    """Fused MSE quantize + pack + store.

    Performs binary-search bucketize, MSE index packing, norm storage,
    and value quantization in one kernel.
    """
    pid = tl.program_id(0)
    token_idx = pid // H
    head_idx = pid % H

    slot = tl.load(Slot_mapping_ptr + token_idx)
    if slot < 0:
        return
    blk = (slot // BLOCK_SIZE).to(tl.int64)
    off = (slot % BLOCK_SIZE).to(tl.int64)
    head_idx_i64 = tl.cast(head_idx, tl.int64)
    slot_base = (
        blk * stride_cache_block
        + off * stride_cache_pos
        + head_idx_i64 * stride_cache_head
    )

    base = pid * D
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # ── 1. BINARY SEARCH BUCKETIZE ───────────────────────────────────
    # Midpoints are sorted (N_CENTROIDS-1 values); binary search finds
    # insertion point in MSE_BITS iterations vs N_CENTROIDS-1 for linear.
    y_vec = tl.load(Y_ptr + base + d_offs, mask=d_mask, other=0.0)
    lo = tl.zeros([BLOCK_D], dtype=tl.int32)
    hi = tl.full([BLOCK_D], N_CENTROIDS - 1, dtype=tl.int32)
    for _ in range(MSE_BITS):
        mid = (lo + hi) >> 1
        # Clamp to valid midpoint index [0, N_CENTROIDS-2] for load safety;
        # the search result (lo) is still correct since converged lanes
        # don't change.
        safe_mid = tl.minimum(mid, N_CENTROIDS - 2)
        mid_val = tl.load(Midpoints_ptr + safe_mid, mask=d_mask, other=0.0)
        lo = tl.where(y_vec >= mid_val, mid + 1, lo)
        hi = tl.where(y_vec >= mid_val, hi, mid)
    idx = tl.minimum(lo, N_CENTROIDS - 1)

    # ── 2. PACK MSE INDICES from register idx ─────────────────────────
    if MSE_BITS == 4:
        idx_pairs = tl.reshape(idx, [BLOCK_D // 2, 2])
        shifts_4 = tl.arange(0, 2) * 4
        packed = tl.sum((idx_pairs & 0xF) << shifts_4[None, :], axis=1).to(tl.uint8)
        mse_offs = tl.arange(0, BLOCK_D // 2)
        mse_mask = mse_offs < MSE_BYTES
        tl.store(KV_cache_ptr + slot_base + mse_offs, packed, mask=mse_mask)

    elif MSE_BITS == 3:
        grp_offs = tl.arange(0, BLOCK_GRP)
        grp_mask = grp_offs < (D // 8)
        idx_grp = tl.reshape(idx, [BLOCK_GRP, 8])
        shifts_3 = tl.arange(0, 8) * 3
        packed_24 = tl.sum((idx_grp & 0x7) << shifts_3[None, :], axis=1)
        b0 = (packed_24 & 0xFF).to(tl.uint8)
        b1 = ((packed_24 >> 8) & 0xFF).to(tl.uint8)
        b2 = ((packed_24 >> 16) & 0xFF).to(tl.uint8)
        tl.store(KV_cache_ptr + slot_base + grp_offs * 3, b0, mask=grp_mask)
        tl.store(KV_cache_ptr + slot_base + grp_offs * 3 + 1, b1, mask=grp_mask)
        tl.store(KV_cache_ptr + slot_base + grp_offs * 3 + 2, b2, mask=grp_mask)

    # ── 3. STORE vec_norm (fp16, 2 bytes) ─────────────────────────────
    norm_offset = MSE_BYTES

    vn_f16 = tl.load(Norms_ptr + pid).to(tl.float16)
    vn_u16 = vn_f16.to(tl.uint16, bitcast=True)
    tl.store(KV_cache_ptr + slot_base + norm_offset, (vn_u16 & 0xFF).to(tl.uint8))
    tl.store(
        KV_cache_ptr + slot_base + norm_offset + 1, ((vn_u16 >> 8) & 0xFF).to(tl.uint8)
    )

    # ── 4. VALUE QUANTIZE + PACK ──────────────────────────────────────
    _store_quantized_value(
        Value_ptr,
        KV_cache_ptr,
        base,
        slot_base,
        d_offs,
        d_mask,
        D=D,
        KPS=KPS,
        VQB=VQB,
        VAL_DATA_BYTES=VAL_DATA_BYTES,
        BLOCK_D=BLOCK_D,
        BLOCK_VAL=BLOCK_VAL,
        BLOCK_GRP=BLOCK_GRP,
    )


# ═══════════════════════════════════════════════════════════════════════
# Launcher
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# MLA fused store kernels (no head dimension, MLA-specific layout)
# ═══════════════════════════════════════════════════════════════════════


@triton.jit
def _tq_mla_fused_store_fp8_kernel(
    kv_c_ptr,  # [N, L] bf16/fp32
    k_pe_ptr,  # [N, R] bf16/fp32
    kv_cache_ptr,  # [total_slots * packed_bytes] uint8
    slot_mapping_ptr,  # [N] int64
    k_scale_ptr,  # [1] fp32
    L: tl.constexpr,
    R: tl.constexpr,
    PACKED_BYTES: tl.constexpr,
    K_PE_BYTES: tl.constexpr,
    K_PE_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """MLA FP8 fused store: quantize kv_c to fp8 + handle k_pe + scatter."""
    token_idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + token_idx)
    if slot < 0:
        return

    slot_i64 = slot.to(tl.int64)
    cache_offset = slot_i64 * PACKED_BYTES

    # Load k_scale (scalar)
    k_scale = tl.load(k_scale_ptr)
    inv_scale = tl.where(k_scale > 0, 1.0 / k_scale, 1.0)

    # ── FP8 kv_c: cast to fp8 and store ──────────────────────────
    l_offs = tl.arange(0, BLOCK_L)
    l_mask = l_offs < L
    kv_c_vals = tl.load(kv_c_ptr + token_idx * L + l_offs, mask=l_mask, other=0.0)
    kv_c_scaled = kv_c_vals.to(tl.float32) * inv_scale
    kv_c_clamped = tl.minimum(tl.maximum(kv_c_scaled, -FP8_MAX), FP8_MAX)
    kv_c_fp8 = kv_c_clamped.to(tl.float8e4nv)
    kv_c_bytes = kv_c_fp8.to(tl.uint8, bitcast=True)
    tl.store(kv_cache_ptr + cache_offset + l_offs, kv_c_bytes, mask=l_mask)

    # ── k_pe ──────────────────────────────────────────────────────
    r_offs = tl.arange(0, BLOCK_R)
    r_mask = r_offs < R
    k_pe_vals = tl.load(k_pe_ptr + token_idx * R + r_offs, mask=r_mask, other=0.0)
    k_pe_offset = cache_offset + L

    if K_PE_FP8:
        # Per-token absmax quantization
        k_pe_f32 = k_pe_vals.to(tl.float32)
        k_pe_absmax = tl.max(tl.abs(tl.where(r_mask, k_pe_f32, 0.0)), axis=0)
        k_pe_scale = k_pe_absmax / FP8_MAX
        k_pe_scale = tl.where(k_pe_scale > 1e-8, k_pe_scale, 1e-8)
        k_pe_inv = 1.0 / k_pe_scale
        kpe_scaled = k_pe_f32 * k_pe_inv
        kpe_clamped = tl.minimum(tl.maximum(kpe_scaled, -FP8_MAX), FP8_MAX)
        k_pe_fp8 = kpe_clamped.to(tl.float8e4nv)
        k_pe_u8 = k_pe_fp8.to(tl.uint8, bitcast=True)
        tl.store(kv_cache_ptr + k_pe_offset + r_offs, k_pe_u8, mask=r_mask)
        # Store scale as fp16 (2 bytes)
        sc_f16 = k_pe_scale.to(tl.float16)
        sc_u16 = sc_f16.to(tl.uint16, bitcast=True)
        tl.store(kv_cache_ptr + k_pe_offset + R, (sc_u16 & 0xFF).to(tl.uint8))
        tl.store(
            kv_cache_ptr + k_pe_offset + R + 1, ((sc_u16 >> 8) & 0xFF).to(tl.uint8)
        )
    else:
        # Store k_pe as bf16 (2*R bytes)
        k_pe_bf16 = k_pe_vals.to(tl.bfloat16)
        k_pe_u16 = k_pe_bf16.to(tl.uint16, bitcast=True)
        tl.store(
            kv_cache_ptr + k_pe_offset + r_offs * 2,
            (k_pe_u16 & 0xFF).to(tl.uint8),
            mask=r_mask,
        )
        tl.store(
            kv_cache_ptr + k_pe_offset + r_offs * 2 + 1,
            ((k_pe_u16 >> 8) & 0xFF).to(tl.uint8),
            mask=r_mask,
        )


@triton.jit
def _tq_mla_fused_store_mse_kernel(
    Y_ptr,  # [N, L] fp32 — rotated normalized keys (x_hat @ PiT)
    Norms_ptr,  # [N] fp32 — key vector norms
    k_pe_ptr,  # [N, R] bf16/fp32
    Midpoints_ptr,  # [n_centroids-1] fp32
    kv_cache_ptr,  # [total_slots * packed_bytes] uint8
    slot_mapping_ptr,  # [N] int64
    L: tl.constexpr,
    R: tl.constexpr,
    PACKED_BYTES: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    K_PE_BYTES: tl.constexpr,
    K_PE_FP8: tl.constexpr,
    MSE_BITS: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """MLA MSE fused store: bucketize + pack + norm store + k_pe store."""
    token_idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + token_idx)
    if slot < 0:
        return

    slot_i64 = slot.to(tl.int64)
    cache_offset = slot_i64 * PACKED_BYTES

    l_offs = tl.arange(0, BLOCK_L)
    l_mask = l_offs < L

    # ── 1. Load rotated keys and binary search bucketize ──────────
    y_vec = tl.load(Y_ptr + token_idx * L + l_offs, mask=l_mask, other=0.0)
    lo = tl.zeros([BLOCK_L], dtype=tl.int32)
    hi = tl.full([BLOCK_L], N_CENTROIDS - 1, dtype=tl.int32)
    for _ in range(MSE_BITS):
        mid = (lo + hi) >> 1
        safe_mid = tl.minimum(mid, N_CENTROIDS - 2)
        mid_val = tl.load(Midpoints_ptr + safe_mid, mask=l_mask, other=0.0)
        lo = tl.where(y_vec >= mid_val, mid + 1, lo)
        hi = tl.where(y_vec >= mid_val, hi, mid)
    idx = tl.minimum(lo, N_CENTROIDS - 1)

    # ── 2. Pack MSE indices ──────────────────────────────────────
    if MSE_BITS == 4:
        idx_pairs = tl.reshape(idx, [BLOCK_L // 2, 2])
        shifts_4 = tl.arange(0, 2) * 4
        packed = tl.sum((idx_pairs & 0xF) << shifts_4[None, :], axis=1).to(tl.uint8)
        mse_offs = tl.arange(0, BLOCK_L // 2)
        mse_mask = mse_offs < MSE_BYTES
        tl.store(kv_cache_ptr + cache_offset + mse_offs, packed, mask=mse_mask)
    else:  # MSE_BITS == 3
        BLOCK_GRP: tl.constexpr = triton.next_power_of_2(L // 8)
        grp_offs = tl.arange(0, BLOCK_GRP)
        grp_mask = grp_offs < (L // 8)
        idx_grp = tl.reshape(idx, [BLOCK_GRP, 8])
        shifts_3 = tl.arange(0, 8) * 3
        packed_24 = tl.sum((idx_grp & 0x7) << shifts_3[None, :], axis=1)
        b0 = (packed_24 & 0xFF).to(tl.uint8)
        b1 = ((packed_24 >> 8) & 0xFF).to(tl.uint8)
        b2 = ((packed_24 >> 16) & 0xFF).to(tl.uint8)
        tl.store(kv_cache_ptr + cache_offset + grp_offs * 3, b0, mask=grp_mask)
        tl.store(kv_cache_ptr + cache_offset + grp_offs * 3 + 1, b1, mask=grp_mask)
        tl.store(kv_cache_ptr + cache_offset + grp_offs * 3 + 2, b2, mask=grp_mask)

    # ── 3. Store vec_norm (fp16, 2 bytes) ─────────────────────────
    norm_offset = cache_offset + MSE_BYTES
    vn_f16 = tl.load(Norms_ptr + token_idx).to(tl.float16)
    vn_u16 = vn_f16.to(tl.uint16, bitcast=True)
    tl.store(kv_cache_ptr + norm_offset, (vn_u16 & 0xFF).to(tl.uint8))
    tl.store(kv_cache_ptr + norm_offset + 1, ((vn_u16 >> 8) & 0xFF).to(tl.uint8))

    # ── 4. k_pe ──────────────────────────────────────────────────
    r_offs = tl.arange(0, BLOCK_R)
    r_mask = r_offs < R
    k_pe_vals = tl.load(k_pe_ptr + token_idx * R + r_offs, mask=r_mask, other=0.0)
    k_pe_offset = cache_offset + MSE_BYTES + 2

    if K_PE_FP8:
        k_pe_f32 = k_pe_vals.to(tl.float32)
        k_pe_absmax = tl.max(tl.abs(tl.where(r_mask, k_pe_f32, 0.0)), axis=0)
        k_pe_scale = k_pe_absmax / FP8_MAX
        k_pe_scale = tl.where(k_pe_scale > 1e-8, k_pe_scale, 1e-8)
        kpe_scaled = k_pe_f32 / k_pe_scale
        kpe_clamped = tl.minimum(tl.maximum(kpe_scaled, -FP8_MAX), FP8_MAX)
        k_pe_fp8 = kpe_clamped.to(tl.float8e4nv)
        k_pe_u8 = k_pe_fp8.to(tl.uint8, bitcast=True)
        tl.store(kv_cache_ptr + k_pe_offset + r_offs, k_pe_u8, mask=r_mask)
        sc_f16 = k_pe_scale.to(tl.float16)
        sc_u16 = sc_f16.to(tl.uint16, bitcast=True)
        tl.store(kv_cache_ptr + k_pe_offset + R, (sc_u16 & 0xFF).to(tl.uint8))
        tl.store(
            kv_cache_ptr + k_pe_offset + R + 1, ((sc_u16 >> 8) & 0xFF).to(tl.uint8)
        )
    else:
        k_pe_bf16 = k_pe_vals.to(tl.bfloat16)
        k_pe_u16 = k_pe_bf16.to(tl.uint16, bitcast=True)
        tl.store(
            kv_cache_ptr + k_pe_offset + r_offs * 2,
            (k_pe_u16 & 0xFF).to(tl.uint8),
            mask=r_mask,
        )
        tl.store(
            kv_cache_ptr + k_pe_offset + r_offs * 2 + 1,
            ((k_pe_u16 >> 8) & 0xFF).to(tl.uint8),
            mask=r_mask,
        )


# ═══════════════════════════════════════════════════════════════════════
# MLA store launchers
# ═══════════════════════════════════════════════════════════════════════


def _mla_fused_store_fp8(
    kv_c: torch.Tensor,  # [N, L] bf16/fp32
    k_pe: torch.Tensor,  # [N, R] bf16/fp32
    kv_cache: torch.Tensor,  # [num_blocks, block_size, packed_bytes] uint8
    slot_mapping: torch.Tensor,  # [N] int64
    k_scale: torch.Tensor,  # scalar fp32
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    k_pe_fp8: bool,
):
    """Launch MLA FP8 fused store kernel."""
    N = kv_c.shape[0]
    L = kv_lora_rank
    R = qk_rope_head_dim
    BLOCK_L = triton.next_power_of_2(L)
    BLOCK_R = triton.next_power_of_2(R)
    k_pe_bytes = R + 2 if k_pe_fp8 else 2 * R
    packed_bytes = L + k_pe_bytes

    grid = (N,)
    _tq_mla_fused_store_fp8_kernel[grid](
        kv_c,
        k_pe,
        kv_cache.view(-1),
        slot_mapping.to(torch.int64),
        k_scale,
        L=L,
        R=R,
        PACKED_BYTES=packed_bytes,
        K_PE_BYTES=k_pe_bytes,
        K_PE_FP8=k_pe_fp8,
        FP8_MAX=448.0,
        BLOCK_L=BLOCK_L,
        BLOCK_R=BLOCK_R,
        num_warps=4,
        num_stages=1,
    )


def _mla_fused_store_mse(
    kv_c: torch.Tensor,  # [N, L] bf16/fp32
    k_pe: torch.Tensor,  # [N, R] bf16/fp32
    kv_cache: torch.Tensor,  # [num_blocks, block_size, packed_bytes] uint8
    slot_mapping: torch.Tensor,  # [N] int64
    PiT: torch.Tensor,  # [L, L] fp32 — Hadamard rotation matrix
    midpoints: torch.Tensor,  # [n_centroids-1] fp32
    mse_bits: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    k_pe_fp8: bool,
):
    """Launch MLA MSE fused store kernel.

    External GEMM handles normalize + rotation (cuBLAS is faster than in-kernel).
    The kernel does: bucketize + MSE index pack + norm store + k_pe store.
    """
    N = kv_c.shape[0]
    L = kv_lora_rank
    R = qk_rope_head_dim
    BLOCK_L = triton.next_power_of_2(L)
    BLOCK_R = triton.next_power_of_2(R)
    mse_bytes = math.ceil(L * mse_bits / 8)
    n_centroids = 2**mse_bits
    k_pe_bytes = R + 2 if k_pe_fp8 else 2 * R
    packed_bytes = mse_bytes + 2 + k_pe_bytes

    # External GEMM: normalize + rotation
    kv_c_f = kv_c.to(torch.float32)
    norms = kv_c_f.norm(dim=1)  # (N,)
    x_hat = kv_c_f / (norms.unsqueeze(1) + 1e-8)
    y = x_hat @ PiT  # (N, L) — rotated

    grid = (N,)
    _tq_mla_fused_store_mse_kernel[grid](
        y,
        norms,
        k_pe,
        midpoints,
        kv_cache.view(-1),
        slot_mapping.to(torch.int64),
        L=L,
        R=R,
        PACKED_BYTES=packed_bytes,
        MSE_BYTES=mse_bytes,
        K_PE_BYTES=k_pe_bytes,
        K_PE_FP8=k_pe_fp8,
        MSE_BITS=mse_bits,
        N_CENTROIDS=n_centroids,
        FP8_MAX=448.0,
        BLOCK_L=BLOCK_L,
        BLOCK_R=BLOCK_R,
        num_warps=4,
        num_stages=1,
    )


def triton_turboquant_store(
    key: torch.Tensor,  # [N, H, D] — raw keys (post-RoPE)
    value: torch.Tensor,  # [N, H, D] — raw values
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, padded_slot] uint8
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
    block_size = kv_cache.shape[1]
    BLOCK_D = triton.next_power_of_2(D)
    mse_bytes = math.ceil(D * mse_bits / 8)
    n_centroids = 2**mse_bits

    val_data_bytes = math.ceil(D * value_quant_bits / 8)

    BLOCK_VAL = triton.next_power_of_2(val_data_bytes)

    # Cache strides (element_size=1 for uint8, so stride in bytes = stride())
    stride_block = kv_cache.stride(0)
    stride_pos = kv_cache.stride(1)
    stride_head = kv_cache.stride(2)

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
            kv_cache.view(-1),
            slot_mapping,
            stride_cache_block=stride_block,
            stride_cache_pos=stride_pos,
            stride_cache_head=stride_head,
            D=D,
            H=H,
            BLOCK_SIZE=block_size,
            BLOCK_D=BLOCK_D,
            KPS=key_packed_size,
            VQB=value_quant_bits,
            VAL_DATA_BYTES=val_data_bytes,
            BLOCK_VAL=BLOCK_VAL,
            BLOCK_GRP=block_grp,
            FP8_E4B15=fp8_e4b15,
            num_warps=4,
            num_stages=1,
        )
        return

    # ── MSE PATH: external GEMM + fused bucketize/pack kernel ──
    # Normalize + rotation GEMM externally (cuBLAS is faster than in-kernel)
    k_flat = key.float().reshape(NH, D)
    norms = k_flat.norm(dim=1, keepdim=True)
    x_hat = k_flat / (norms + 1e-8)
    y = x_hat @ PiT

    v_flat = value.float().reshape(NH, D)

    # Fused kernel: bucketize + MSE index pack + norm store + value pack
    grid = (NH,)
    _tq_fused_store_mse[grid](
        y,
        norms.squeeze(1),
        v_flat,
        midpoints,
        kv_cache.view(-1),
        slot_mapping,
        stride_cache_block=stride_block,
        stride_cache_pos=stride_pos,
        stride_cache_head=stride_head,
        D=D,
        H=H,
        BLOCK_SIZE=block_size,
        BLOCK_D=BLOCK_D,
        MSE_BYTES=mse_bytes,
        KPS=key_packed_size,
        VQB=value_quant_bits,
        VAL_DATA_BYTES=val_data_bytes,
        BLOCK_VAL=BLOCK_VAL,
        MSE_BITS=mse_bits,
        N_CENTROIDS=n_centroids,
        BLOCK_GRP=block_grp,
        num_warps=4,
        num_stages=1,
    )
