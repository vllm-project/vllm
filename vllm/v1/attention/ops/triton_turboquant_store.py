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
# Fully fused MSE store: cast + norm + rotate (fp16) + bucketize
# Replaces _fused_cast_normalize_kv + external GEMM (x_hat @ PiT) +
# _tq_fused_store_mse with a SINGLE kernel launch, eliminating 4
# intermediate tensors (x_hat, norms, v_flat, y) and 1 cuBLAS GEMM.
# ═══════════════════════════════════════════════════════════════════════


@triton.jit
def _tq_fused_cast_norm_rot_store_mse(
    # Raw inputs
    Key_ptr,  # [NH, D] bf16/fp16 — raw keys (post-RoPE)
    Value_ptr,  # [NH, D] bf16/fp16 — raw values
    PiT_half_ptr,  # [D, D] fp16 — rotation matrix (Pi^T in fp16)
    # Quantization tables
    Midpoints_half_ptr,  # [n_centroids-1] fp32
    # Cache and indexing
    KV_cache_ptr,  # [total_bytes] uint8 (flattened view)
    Slot_mapping_ptr,  # [N] int32 — per-token slot indices
    # Cache strides
    stride_cache_block: tl.constexpr,
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
    # PiT_half strides
    stride_pit_r,
    stride_pit_c,
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
    """Fused: bf16→fp32 cast + L2 norm + normalize + fp16 rotation +
    binary-search bucketize + MSE index pack + norm store + value quantize.

    One program per (token, head) pair. All compute in registers —
    eliminates 4 intermediate tensors and 1 cuBLAS GEMM launch.
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

    # ── 1. KEY: bf16 → fp32 + L2 norm + normalize ──────────────────
    k = tl.load(Key_ptr + base + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    k_f32 = k.to(tl.float32)
    norm = tl.sqrt(tl.sum(k_f32 * k_f32, axis=0) + 1e-16).to(tl.float16)
    x_hat = k / norm

    # ── 2. ROTATION: y = x_hat @ PiT_half ──────────────────────────
    # Load PiT_half: [BLOCK_D, BLOCK_D] fp16
    pit_offs = d_offs[:, None] * stride_pit_r + d_offs[None, :] * stride_pit_c
    pit_mask = d_mask[:, None] & d_mask[None, :]
    PiT_half = tl.load(PiT_half_ptr + pit_offs, mask=pit_mask, other=0.0)

    # Matrix-vector multiply via broadcast + reduce:
    # y[j] = sum_i(x_hat[i] * PiT_half[i, j])
    # x_hat_2d = tl.expand_dims(x_hat, axis=1)  # [D, 1]
    y_vec = tl.sum(x_hat[:, None] * PiT_half, axis=0)  # [D] fp16

    # ── 3. BINARY SEARCH BUCKETIZE ─────────────────────────────────
    lo = tl.zeros([BLOCK_D], dtype=tl.int32)
    hi = tl.full([BLOCK_D], N_CENTROIDS - 1, dtype=tl.int32)
    for _ in range(MSE_BITS):
        mid = (lo + hi) >> 1
        safe_mid = tl.minimum(mid, N_CENTROIDS - 2)
        mid_val = tl.load(Midpoints_half_ptr + safe_mid, mask=d_mask, other=0.0)
        lo = tl.where(y_vec >= mid_val, mid + 1, lo)
        hi = tl.where(y_vec >= mid_val, hi, mid)
    idx = tl.minimum(lo, N_CENTROIDS - 1)

    # ── 4. PACK MSE INDICES ────────────────────────────────────────
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

    # ── 5. STORE vec_norm (fp16, 2 bytes) ──────────────────────────
    norm_offset = MSE_BYTES
    vn_f16 = norm
    vn_u16 = vn_f16.to(tl.uint16, bitcast=True)
    tl.store(KV_cache_ptr + slot_base + norm_offset, (vn_u16 & 0xFF).to(tl.uint8))
    tl.store(
        KV_cache_ptr + slot_base + norm_offset + 1, ((vn_u16 >> 8) & 0xFF).to(tl.uint8)
    )

    # ── 6. VALUE QUANTIZE + PACK ───────────────────────────────────
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
# Fused cast + normalize (key) + cast (value) pre-op kernel
# Replaces 3 PyTorch kernel launches (key.float, norm, key/norm) + 1
# (value.float) with a single Triton kernel, reducing launch overhead
# and intermediate-tensor memory bandwidth.
# ═══════════════════════════════════════════════════════════════════════


@triton.jit
def _fused_cast_normalize_kv(
    Key_ptr,  # [NH, D] bfloat16/float16
    Value_ptr,  # [NH, D] bfloat16/float16
    X_hat_out,  # [NH, D] float32 — normalized key (k / ||k||)
    Norms_out,  # [NH, 1] float32 — per-vector L2 norm
    V_flat_out,  # [NH, D] float32 — value cast to fp32
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused: key bf16→fp32 cast + L2 norm + normalize + value bf16→fp32 cast.

    One program per (token, head) pair. Loads K and V for the same
    (token, head), computes the key norm and normalized key in registers,
    and writes all three outputs.
    """
    pid = tl.program_id(0)
    base = pid * D
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # ── Key: load bf16, cast to fp32 ──────────────────────────────
    k = tl.load(Key_ptr + base + d_offs, mask=d_mask, other=0.0).to(tl.float32)

    # ── Compute L2 norm ───────────────────────────────────────────
    k_sq = k * k
    norm = tl.sqrt(tl.sum(k_sq, axis=0) + 1e-16)

    # ── Normalize ─────────────────────────────────────────────────
    x_hat = k / norm
    tl.store(X_hat_out + base + d_offs, x_hat, mask=d_mask)
    tl.store(Norms_out + pid, norm)

    # ── Value: load bf16, cast to fp32, store ────────────────────
    v = tl.load(Value_ptr + base + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    tl.store(V_flat_out + base + d_offs, v, mask=d_mask)


# ═══════════════════════════════════════════════════════════════════════
# Launcher
# ═══════════════════════════════════════════════════════════════════════


def triton_turboquant_store(
    key: torch.Tensor,  # [N, H, D] — raw keys (post-RoPE)
    value: torch.Tensor,  # [N, H, D] — raw values
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, padded_slot] uint8
    slot_mapping: torch.Tensor,  # [N] int32
    PiT: torch.Tensor,  # [D, D] float32 (unused, kept for compat)
    midpoints: torch.Tensor,  # [n_centroids-1] float32
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    key_fp8: bool = False,
    PiT_half: torch.Tensor | None = None,  # [D, D] fp16 — rotation matrix for fused store
    midpoints_half: torch.Tensor | None = None,
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

    # ── MSE PATH: fully fused cast + norm + rotate + bucketize + pack ──
    # Single kernel replaces 3-step process:
    #   1. _fused_cast_normalize_kv (bf16→fp32, norm, normalize)
    #   2. x_hat @ PiT (external cuBLAS GEMM)
    #   3. _tq_fused_store_mse (bucketize + pack)
    # .contiguous() ensures correct pointer arithmetic in Triton regardless
    # of input tensor strides (critical for TP sharding / non-contiguous views).
    k_flat = key.reshape(NH, D).contiguous()
    v_flat = value.reshape(NH, D).contiguous()

    grid = (NH,)
    _tq_fused_cast_norm_rot_store_mse[grid](
        k_flat,
        v_flat,
        PiT_half,
        midpoints_half,
        kv_cache.view(-1),
        slot_mapping,
        stride_cache_block=stride_block,
        stride_cache_pos=stride_pos,
        stride_cache_head=stride_head,
        stride_pit_r=PiT_half.stride(0),
        stride_pit_c=PiT_half.stride(1),
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
