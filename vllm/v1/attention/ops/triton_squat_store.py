# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernel for SQuat KV store (4-bit MSE quantization).

Extends TurboQuant 4bit-nc MSE store with block-wise SQuat correction:
  1. Binary-search bucketize first half of rotated key to 4-bit centroids
  2. Compute quantization error, apply M_update correction to second half
  3. Binary-search bucketize the corrected second half
  4. Pack 4-bit MSE indices + store norm + 4-bit value quantize

Reference: Wang et al., "SQuat: Subspace-orthogonal KV Cache Quantization",
COLM 2025; preprint arXiv:2503.24358.
"""

import math

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_turboquant_store import _store_quantized_value


@triton.jit
def _squat_fused_store_mse(
    Y_ptr,  # [NH, D] float32 — rotated normalized keys
    Norms_ptr,  # [NH] float32 — key vector norms
    Value_ptr,  # [NH, D] float32 — raw values
    Midpoints_ptr,  # [n_centroids-1] float32
    Centroids_ptr,  # [n_centroids] float32 — sorted centroids
    M_update_ptr,  # [H, HALF_D, HALF_D] float32 — correction matrices
    KV_cache_ptr,  # [total_bytes] uint8 (flattened view)
    Slot_mapping_ptr,  # [N] int32
    stride_cache_block: tl.constexpr,
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
    stride_m_head: tl.constexpr,
    stride_m_row: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    HALF_D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    BLOCK_VAL: tl.constexpr,
    MSE_BITS: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    BLOCK_GRP: tl.constexpr = 16,
    HAS_SQUAT: tl.constexpr = 1,
    CORRECTION_SCALE: tl.constexpr = 1.0,
):
    """Fused SQuat correction + MSE quantize + pack + store."""
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

    if HAS_SQUAT:
        # ── SQUAT PATH ──
        y_vec = tl.load(Y_ptr + base + d_offs, mask=d_mask, other=0.0)
        first_offs = tl.arange(0, HALF_D)
        y_first = tl.load(Y_ptr + base + first_offs)

        # Bucketize first half
        lo1 = tl.zeros([HALF_D], dtype=tl.int32)
        hi1 = tl.full([HALF_D], N_CENTROIDS - 1, dtype=tl.int32)
        for _ in range(MSE_BITS):
            mid1 = (lo1 + hi1) >> 1
            safe_mid1 = tl.minimum(mid1, N_CENTROIDS - 2)
            mid_val1 = tl.load(Midpoints_ptr + safe_mid1)
            lo1 = tl.where(y_first >= mid_val1, mid1 + 1, lo1)
            hi1 = tl.where(y_first >= mid_val1, hi1, mid1)
        idx_first = tl.minimum(lo1, N_CENTROIDS - 1)

        # Correction: error @ M_update
        first_quant = tl.load(Centroids_ptr + idx_first)
        error = first_quant - y_first
        m_base = head_idx * stride_m_head
        row_offs = tl.arange(0, HALF_D)
        col_offs = tl.arange(0, HALF_D)
        m_offs = row_offs[:, None] * stride_m_row + col_offs[None, :]
        m_block = tl.load(M_update_ptr + m_base + m_offs)
        error_2d = tl.reshape(error, [1, HALF_D])
        correction_2d = tl.dot(error_2d, m_block)
        correction = tl.reshape(correction_2d, [HALF_D]) * CORRECTION_SCALE

        # Apply correction to second half, then bucketize full vector
        second_offs = tl.arange(0, HALF_D) + HALF_D
        y_second = tl.load(Y_ptr + base + second_offs)
        tl.store(Y_ptr + base + second_offs, y_second + correction)
        y_corrected = tl.load(Y_ptr + base + d_offs, mask=d_mask, other=0.0)

        lo = tl.zeros([BLOCK_D], dtype=tl.int32)
        hi = tl.full([BLOCK_D], N_CENTROIDS - 1, dtype=tl.int32)
        for _ in range(MSE_BITS):
            mid = (lo + hi) >> 1
            safe_mid = tl.minimum(mid, N_CENTROIDS - 2)
            mid_val = tl.load(Midpoints_ptr + safe_mid, mask=d_mask, other=0.0)
            lo = tl.where(y_corrected >= mid_val, mid + 1, lo)
            hi = tl.where(y_corrected >= mid_val, hi, mid)
        idx = tl.minimum(lo, N_CENTROIDS - 1)
    else:
        # ── PLAIN TQ PATH ──
        y_vec = tl.load(Y_ptr + base + d_offs, mask=d_mask, other=0.0)
        lo = tl.zeros([BLOCK_D], dtype=tl.int32)
        hi = tl.full([BLOCK_D], N_CENTROIDS - 1, dtype=tl.int32)
        for _ in range(MSE_BITS):
            mid = (lo + hi) >> 1
            safe_mid = tl.minimum(mid, N_CENTROIDS - 2)
            mid_val = tl.load(Midpoints_ptr + safe_mid, mask=d_mask, other=0.0)
            lo = tl.where(y_vec >= mid_val, mid + 1, lo)
            hi = tl.where(y_vec >= mid_val, hi, mid)
        idx = tl.minimum(lo, N_CENTROIDS - 1)

    # Pack 4-bit MSE indices (two per byte)
    idx_pairs = tl.reshape(idx, [BLOCK_D // 2, 2])
    shifts_4 = tl.arange(0, 2) * 4
    packed = tl.sum((idx_pairs & 0xF) << shifts_4[None, :], axis=1).to(tl.uint8)
    mse_offs = tl.arange(0, BLOCK_D // 2)
    mse_mask = mse_offs < MSE_BYTES
    tl.store(KV_cache_ptr + slot_base + mse_offs, packed, mask=mse_mask)

    # Store norm
    norm_offset = MSE_BYTES
    vn_f16 = tl.load(Norms_ptr + pid).to(tl.float16)
    vn_u16 = vn_f16.to(tl.uint16, bitcast=True)
    tl.store(KV_cache_ptr + slot_base + norm_offset, (vn_u16 & 0xFF).to(tl.uint8))
    tl.store(
        KV_cache_ptr + slot_base + norm_offset + 1, ((vn_u16 >> 8) & 0xFF).to(tl.uint8)
    )

    # Value quantize + pack
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


def squat_triton_store_mse(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    PiT: torch.Tensor,
    midpoints: torch.Tensor,
    centroids_sorted: torch.Tensor,
    M_update: torch.Tensor | None,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    correction_scale: float = 1.0,
):
    """Launch fused SQuat MSE store kernel."""
    N, H, D = key.shape
    NH = N * H
    HALF_D = D // 2
    block_size = kv_cache.shape[1]
    BLOCK_D = triton.next_power_of_2(D)
    mse_bytes = math.ceil(D * mse_bits / 8)
    n_centroids = 2**mse_bits
    val_data_bytes = math.ceil(D * value_quant_bits / 8)
    BLOCK_VAL = triton.next_power_of_2(val_data_bytes)
    block_grp = triton.next_power_of_2(D // 8) if D >= 8 else 1

    stride_block = kv_cache.stride(0)
    stride_pos = kv_cache.stride(1)
    stride_head = kv_cache.stride(2)

    k_flat = key.float().reshape(NH, D)
    norms = k_flat.norm(dim=1, keepdim=True)
    y = (k_flat / (norms + 1e-8)) @ PiT
    v_flat = value.float().reshape(NH, D)

    m_flat = (
        M_update.to(device=key.device, dtype=torch.float32).contiguous()
        if M_update is not None
        else torch.empty(1, device=key.device, dtype=torch.float32)
    )
    has_squat = M_update is not None

    grid = (NH,)
    _squat_fused_store_mse[grid](
        y,
        norms.squeeze(1),
        v_flat,
        midpoints,
        centroids_sorted,
        m_flat,
        kv_cache.view(-1),
        slot_mapping,
        stride_cache_block=stride_block,
        stride_cache_pos=stride_pos,
        stride_cache_head=stride_head,
        stride_m_head=HALF_D * HALF_D,
        stride_m_row=HALF_D,
        D=D,
        H=H,
        HALF_D=HALF_D,
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
        HAS_SQUAT=1 if has_squat else 0,
        CORRECTION_SCALE=correction_scale,
        num_warps=4,
        num_stages=1,
    )
