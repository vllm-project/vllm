# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused TurboQuant MLA KV-cache store (decode hot path).

Hybrid launch (default): one batched ``y = x_hat @ Pi`` GEMM plus a single
Triton kernel for bucketize -> pack -> scale -> k_pe -> scatter.

``turboquant_*_nc`` presets enable FWHT store + k_pe 4-bit automatically
(see ``tq_mla_defaults.py``).  Env vars ``VLLM_TQ_MLA_STORE_FWHT`` /
``VLLM_TQ_KPE_4BIT`` are opt-out regression anchors only (``=0``).

Set ``VLLM_TQ_MLA_FUSED_STORE=0`` to fall back to the legacy PyTorch path.
"""

from __future__ import annotations

import math
import os

import torch

from vllm.triton_utils import tl, triton

_TQ_MLA_STORE_DISABLE_AUTOTUNE = (
    os.environ.get("VLLM_TQ_MLA_STORE_AUTOTUNE", "0") != "1"
)

_TQ_MLA_STORE_AUTOTUNE_CONFIGS = (
    [triton.Config({}, num_warps=4, num_stages=1)]
    if _TQ_MLA_STORE_DISABLE_AUTOTUNE
    else [
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=2),
    ]
)


def tq_mla_fused_store_enabled() -> bool:
    return os.environ.get("VLLM_TQ_MLA_FUSED_STORE", "1") != "0"


def tq_mla_store_fwht_enabled() -> bool:
    return os.environ.get("VLLM_TQ_MLA_STORE_FWHT", "0") == "1"


def tq_mla_store_bf16_direct_enabled() -> bool:
    """Fuse bf16->norm->FWHT into the store kernel (skip Python fp32 preamble).

    Requires ``use_fwht=True`` and 4-bit k_pe; falls back to the legacy launch
    when disabled or unsupported.
    """
    return os.environ.get("VLLM_TQ_MLA_STORE_BF16_DIRECT", "1") == "1"


def tq_mla_kpe_4bit_enabled() -> bool:
    return os.environ.get("VLLM_TQ_KPE_4BIT", "0") == "1"


def kpe_mse_index_bytes(rope_dim: int, bits: int = 4) -> int:
    return math.ceil(rope_dim * bits / 8)


def kpe_packed_bytes(rope_dim: int, *, kpe_4bit: bool, kpe_fp8: bool) -> int:
    if kpe_4bit:
        return kpe_mse_index_bytes(rope_dim, 4) + 2
    if kpe_fp8:
        return rope_dim + 2
    return 2 * rope_dim


@triton.jit
def _fwht_1d(x, N: tl.constexpr, LOG2N: tl.constexpr, INV_SQRT_N: tl.constexpr):
    """Sylvester FWHT matching ``_build_hadamard(N) @ x`` (orthonormal)."""
    idx = tl.arange(0, N)
    v = x.to(tl.float32)
    if LOG2N >= 1:
        partner = idx ^ 1
        v_p = tl.gather(v, partner, 0)
        v = tl.where((idx & 1) == 0, v + v_p, v_p - v)
    if LOG2N >= 2:
        partner = idx ^ 2
        v_p = tl.gather(v, partner, 0)
        v = tl.where((idx & 2) == 0, v + v_p, v_p - v)
    if LOG2N >= 3:
        partner = idx ^ 4
        v_p = tl.gather(v, partner, 0)
        v = tl.where((idx & 4) == 0, v + v_p, v_p - v)
    if LOG2N >= 4:
        partner = idx ^ 8
        v_p = tl.gather(v, partner, 0)
        v = tl.where((idx & 8) == 0, v + v_p, v_p - v)
    if LOG2N >= 5:
        partner = idx ^ 16
        v_p = tl.gather(v, partner, 0)
        v = tl.where((idx & 16) == 0, v + v_p, v_p - v)
    if LOG2N >= 6:
        partner = idx ^ 32
        v_p = tl.gather(v, partner, 0)
        v = tl.where((idx & 32) == 0, v + v_p, v_p - v)
    if LOG2N >= 7:
        partner = idx ^ 64
        v_p = tl.gather(v, partner, 0)
        v = tl.where((idx & 64) == 0, v + v_p, v_p - v)
    if LOG2N >= 8:
        partner = idx ^ 128
        v_p = tl.gather(v, partner, 0)
        v = tl.where((idx & 128) == 0, v + v_p, v_p - v)
    if LOG2N >= 9:
        partner = idx ^ 256
        v_p = tl.gather(v, partner, 0)
        v = tl.where((idx & 256) == 0, v + v_p, v_p - v)
    if LOG2N >= 10:
        partner = idx ^ 512
        v_p = tl.gather(v, partner, 0)
        v = tl.where((idx & 512) == 0, v + v_p, v_p - v)
    return v * INV_SQRT_N


@triton.jit
def _tq_mla_bucketize_1d(
    vals,
    Midpoints_ptr,
    d_mask,
    BLOCK_D: tl.constexpr,
    MSE_BITS: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
):
    lo = tl.zeros([BLOCK_D], dtype=tl.int32)
    hi = tl.full([BLOCK_D], N_CENTROIDS - 1, dtype=tl.int32)
    for _ in tl.static_range(MSE_BITS):
        mid = (lo + hi) >> 1
        safe_mid = tl.minimum(mid, N_CENTROIDS - 2)
        mid_val = tl.load(Midpoints_ptr + safe_mid, mask=d_mask, other=0.0)
        lo = tl.where(vals >= mid_val, mid + 1, lo)
        hi = tl.where(vals >= mid_val, hi, mid)
    return tl.minimum(lo, N_CENTROIDS - 1)


@triton.jit
def _tq_mla_gather_centroids_1d(
    Centroids_ptr,
    idx,
    d_mask,
    BLOCK_D: tl.constexpr,
    MSE_BITS: tl.constexpr,
):
    n_centroids: tl.constexpr = 1 << MSE_BITS
    centroids = tl.load(Centroids_ptr + tl.arange(0, n_centroids))
    y_hat = tl.gather(centroids, idx, 0)
    return tl.where(d_mask, y_hat, 0.0)


@triton.jit
def _tq_mla_pack_4bit(
    idx,
    MSE_BYTES: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    idx_pairs = tl.reshape(idx, [BLOCK_D // 2, 2])
    shifts_4 = tl.arange(0, 2) * 4
    packed = tl.sum((idx_pairs & 0xF) << shifts_4[None, :], axis=1).to(tl.uint8)
    mse_offs = tl.arange(0, BLOCK_D // 2)
    mse_mask = mse_offs < MSE_BYTES
    return packed, mse_offs, mse_mask


@triton.jit
def _tq_mla_store_mse_1d(
    y,
    norm,
    d_mask,
    Midpoints_ptr,
    Centroids_ptr,
    KV_cache_ptr,
    slot_base,
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    BLOCK_D: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    NORM_CORRECTION: tl.constexpr,
):
    idx = _tq_mla_bucketize_1d(
        y, Midpoints_ptr, d_mask, BLOCK_D, MSE_BITS, N_CENTROIDS
    )

    if NORM_CORRECTION:
        y_hat = _tq_mla_gather_centroids_1d(
            Centroids_ptr, idx, d_mask, BLOCK_D, MSE_BITS
        )
        c_norm_sq = tl.sum(y_hat * y_hat, axis=0)
        c_norm = tl.sqrt(tl.maximum(c_norm_sq, 1e-8))
        eff_scale = (norm / c_norm).to(tl.float16)
    else:
        eff_scale = norm.to(tl.float16)

    scale_u16 = eff_scale.to(tl.uint16, bitcast=True)

    if MSE_BITS == 4:
        packed, mse_offs, mse_mask = _tq_mla_pack_4bit(idx, MSE_BYTES, BLOCK_D)
        tl.store(KV_cache_ptr + slot_base + mse_offs, packed, mask=mse_mask)

    tl.store(KV_cache_ptr + slot_base + MSE_BYTES, (scale_u16 & 0xFF).to(tl.uint8))
    tl.store(
        KV_cache_ptr + slot_base + MSE_BYTES + 1,
        ((scale_u16 >> 8) & 0xFF).to(tl.uint8),
    )


@triton.autotune(
    configs=_TQ_MLA_STORE_AUTOTUNE_CONFIGS,
    key=[
        "D",
        "R",
        "MSE_BITS",
        "KPE_MODE",
        "NORM_CORRECTION",
        "USE_FWHT",
    ],
)
@triton.jit
def _tq_mla_fused_store_kernel(
    Y_ptr,
    X_hat_ptr,
    Norm_ptr,
    KPE_ptr,
    KV_cache_ptr,
    Slot_mapping_ptr,
    Midpoints_ptr,
    Centroids_ptr,
    Kpe_midpoints_ptr,
    Kpe_centroids_ptr,
    stride_y,
    stride_xhat,
    stride_norm,
    stride_kpe,
    stride_cache_row,
    D: tl.constexpr,
    R: tl.constexpr,
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KV_C_BYTES: tl.constexpr,
    KPE_MSE_BYTES: tl.constexpr,
    KPE_BYTES: tl.constexpr,
    PACKED_BYTES: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    LOG2N_D: tl.constexpr,
    LOG2N_R: tl.constexpr,
    INV_SQRT_D: tl.constexpr,
    INV_SQRT_R: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    KPE_MODE: tl.constexpr,  # 0=bf16, 1=fp8, 2=4bit
    NORM_CORRECTION: tl.constexpr,
    USE_FWHT: tl.constexpr,
):
    """Fused MLA TQ4 store: rotate -> bucketize -> pack -> k_pe -> scatter."""
    tid = tl.program_id(0)
    slot = tl.load(Slot_mapping_ptr + tid)
    if slot < 0:
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    norm = tl.load(Norm_ptr + tid * stride_norm).to(tl.float32)

    if USE_FWHT:
        x_hat = tl.load(
            X_hat_ptr + tid * stride_xhat + d_offs,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        y = _fwht_1d(x_hat, BLOCK_D, LOG2N_D, INV_SQRT_D)
    else:
        y = tl.load(
            Y_ptr + tid * stride_y + d_offs,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)

    slot_base = slot.to(tl.int64) * stride_cache_row
    _tq_mla_store_mse_1d(
        y,
        norm,
        d_mask,
        Midpoints_ptr,
        Centroids_ptr,
        KV_cache_ptr,
        slot_base,
        MSE_BITS,
        MSE_BYTES,
        BLOCK_D,
        N_CENTROIDS,
        NORM_CORRECTION,
    )

    r_offs = tl.arange(0, BLOCK_R)
    r_mask = r_offs < R
    kpe_base = KPE_ptr + tid * stride_kpe
    kpe_vec = tl.load(kpe_base + r_offs, mask=r_mask, other=0.0).to(tl.float32)

    if KPE_MODE == 2:
        kpe_norm_sq = tl.sum(tl.where(r_mask, kpe_vec * kpe_vec, 0.0), axis=0)
        kpe_norm = tl.sqrt(tl.maximum(kpe_norm_sq, 1e-8))
        kpe_x_hat = tl.where(r_mask, kpe_vec / kpe_norm, 0.0)
        kpe_y = _fwht_1d(kpe_x_hat, BLOCK_R, LOG2N_R, INV_SQRT_R)
        kpe_base_off = slot_base + KV_C_BYTES
        _tq_mla_store_mse_1d(
            kpe_y,
            kpe_norm,
            r_mask,
            Kpe_midpoints_ptr,
            Kpe_centroids_ptr,
            KV_cache_ptr,
            kpe_base_off,
            4,
            KPE_MSE_BYTES,
            BLOCK_R,
            16,
            NORM_CORRECTION,
        )
    elif KPE_MODE == 1:
        max_abs = tl.max(tl.where(r_mask, tl.abs(kpe_vec), 0.0), axis=0)
        max_safe = tl.maximum(max_abs, 1e-8)
        kpe_scale = max_safe / 448.0
        inv_scale = tl.where(kpe_scale > 0.0, 1.0 / kpe_scale, 1.0)
        kpe_scaled = tl.minimum(
            tl.maximum(kpe_vec * inv_scale, -448.0), 448.0
        )
        kpe_fp8 = kpe_scaled.to(tl.float8e4nv)
        kpe_u8 = kpe_fp8.to(tl.uint8, bitcast=True)
        kpe_scale_u16 = kpe_scale.to(tl.float16).to(tl.uint16, bitcast=True)
        tl.store(
            KV_cache_ptr + slot_base + KV_C_BYTES + r_offs,
            kpe_u8,
            mask=r_mask,
        )
        tl.store(
            KV_cache_ptr + slot_base + KV_C_BYTES + R,
            (kpe_scale_u16 & 0xFF).to(tl.uint8),
        )
        tl.store(
            KV_cache_ptr + slot_base + KV_C_BYTES + R + 1,
            ((kpe_scale_u16 >> 8) & 0xFF).to(tl.uint8),
        )
    else:
        kpe_bf16 = kpe_vec.to(tl.bfloat16)
        kpe_u16 = kpe_bf16.to(tl.uint16, bitcast=True)
        tl.store(
            KV_cache_ptr + slot_base + KV_C_BYTES + r_offs * 2,
            (kpe_u16 & 0xFF).to(tl.uint8),
            mask=r_mask,
        )
        tl.store(
            KV_cache_ptr + slot_base + KV_C_BYTES + r_offs * 2 + 1,
            ((kpe_u16 >> 8) & 0xFF).to(tl.uint8),
            mask=r_mask,
        )


@triton.autotune(
    configs=_TQ_MLA_STORE_AUTOTUNE_CONFIGS,
    key=[
        "D",
        "R",
        "MSE_BITS",
        "KPE_MODE",
        "NORM_CORRECTION",
    ],
)
@triton.jit
def _tq_mla_fused_store_kernel_bf16_direct(
    KV_C_ptr,
    KPE_ptr,
    KV_cache_ptr,
    Slot_mapping_ptr,
    Midpoints_ptr,
    Centroids_ptr,
    Kpe_midpoints_ptr,
    Kpe_centroids_ptr,
    stride_kvc,
    stride_kpe,
    stride_cache_row,
    D: tl.constexpr,
    R: tl.constexpr,
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KV_C_BYTES: tl.constexpr,
    KPE_MSE_BYTES: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    LOG2N_D: tl.constexpr,
    LOG2N_R: tl.constexpr,
    INV_SQRT_D: tl.constexpr,
    INV_SQRT_R: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    KPE_MODE: tl.constexpr,
    NORM_CORRECTION: tl.constexpr,
):
    """FWHT store with inline bf16 kv_c norm (no Python fp32 x_hat buffer)."""
    tid = tl.program_id(0)
    slot = tl.load(Slot_mapping_ptr + tid)
    if slot < 0:
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    kv_c_f = (
        tl.load(
            KV_C_ptr + tid * stride_kvc + d_offs,
            mask=d_mask,
            other=0.0,
        )
        .to(tl.float32)
    )
    norm_sq = tl.sum(tl.where(d_mask, kv_c_f * kv_c_f, 0.0), axis=0)
    norm = tl.sqrt(tl.maximum(norm_sq, 1e-8))
    x_hat = tl.where(d_mask, kv_c_f / norm, 0.0)
    y = _fwht_1d(x_hat, BLOCK_D, LOG2N_D, INV_SQRT_D)

    slot_base = slot.to(tl.int64) * stride_cache_row
    _tq_mla_store_mse_1d(
        y,
        norm,
        d_mask,
        Midpoints_ptr,
        Centroids_ptr,
        KV_cache_ptr,
        slot_base,
        MSE_BITS,
        MSE_BYTES,
        BLOCK_D,
        N_CENTROIDS,
        NORM_CORRECTION,
    )

    r_offs = tl.arange(0, BLOCK_R)
    r_mask = r_offs < R
    kpe_vec = (
        tl.load(
            KPE_ptr + tid * stride_kpe + r_offs,
            mask=r_mask,
            other=0.0,
        )
        .to(tl.float32)
    )

    if KPE_MODE == 2:
        kpe_norm_sq = tl.sum(tl.where(r_mask, kpe_vec * kpe_vec, 0.0), axis=0)
        kpe_norm = tl.sqrt(tl.maximum(kpe_norm_sq, 1e-8))
        kpe_x_hat = tl.where(r_mask, kpe_vec / kpe_norm, 0.0)
        kpe_y = _fwht_1d(kpe_x_hat, BLOCK_R, LOG2N_R, INV_SQRT_R)
        kpe_base_off = slot_base + KV_C_BYTES
        _tq_mla_store_mse_1d(
            kpe_y,
            kpe_norm,
            r_mask,
            Kpe_midpoints_ptr,
            Kpe_centroids_ptr,
            KV_cache_ptr,
            kpe_base_off,
            4,
            KPE_MSE_BYTES,
            BLOCK_R,
            16,
            NORM_CORRECTION,
        )
    elif KPE_MODE == 1:
        max_abs = tl.max(tl.where(r_mask, tl.abs(kpe_vec), 0.0), axis=0)
        max_safe = tl.maximum(max_abs, 1e-8)
        kpe_scale = max_safe / 448.0
        inv_scale = tl.where(kpe_scale > 0.0, 1.0 / kpe_scale, 1.0)
        kpe_scaled = tl.minimum(
            tl.maximum(kpe_vec * inv_scale, -448.0), 448.0
        )
        kpe_fp8 = kpe_scaled.to(tl.float8e4nv)
        kpe_u8 = kpe_fp8.to(tl.uint8, bitcast=True)
        kpe_scale_u16 = kpe_scale.to(tl.float16).to(tl.uint16, bitcast=True)
        tl.store(
            KV_cache_ptr + slot_base + KV_C_BYTES + r_offs,
            kpe_u8,
            mask=r_mask,
        )
        tl.store(
            KV_cache_ptr + slot_base + KV_C_BYTES + R,
            (kpe_scale_u16 & 0xFF).to(tl.uint8),
        )
        tl.store(
            KV_cache_ptr + slot_base + KV_C_BYTES + R + 1,
            ((kpe_scale_u16 >> 8) & 0xFF).to(tl.uint8),
        )
    else:
        kpe_bf16 = kpe_vec.to(tl.bfloat16)
        kpe_u16 = kpe_bf16.to(tl.uint16, bitcast=True)
        tl.store(
            KV_cache_ptr + slot_base + KV_C_BYTES + r_offs * 2,
            (kpe_u16 & 0xFF).to(tl.uint8),
            mask=r_mask,
        )
        tl.store(
            KV_cache_ptr + slot_base + KV_C_BYTES + r_offs * 2 + 1,
            ((kpe_u16 >> 8) & 0xFF).to(tl.uint8),
            mask=r_mask,
        )


def _log2_const(n: int) -> int:
    assert n > 0 and (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
    return n.bit_length() - 1


def _launch_fused_store_bf16_direct(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    midpoints: torch.Tensor,
    centroids_fp32: torch.Tensor,
    mse_bits: int,
    mse_bytes: int,
    kv_c_bytes: int,
    packed_bytes: int,
    kpe_fp8: bool,
    kpe_4bit: bool,
    kpe_midpoints: torch.Tensor | None,
    kpe_centroids_fp32: torch.Tensor | None,
    kpe_mse_bytes: int,
    norm_correction: bool,
) -> None:
    N = slot_mapping.shape[0]
    D = kv_c.shape[-1]
    R = k_pe.shape[-1]
    block_d = triton.next_power_of_2(D)
    block_r = triton.next_power_of_2(R)

    if kpe_4bit:
        kpe_mode = 2
        if kpe_midpoints is None or kpe_centroids_fp32 is None:
            raise ValueError("kpe_4bit requires kpe_midpoints and kpe_centroids_fp32")
    elif kpe_fp8:
        kpe_mode = 1
    else:
        kpe_mode = 0

    if kpe_mse_bytes == 0 and kpe_4bit:
        kpe_mse_bytes = kpe_mse_index_bytes(R, 4)

    cache_flat = kv_cache.view(-1, packed_bytes)
    slots = slot_mapping[:N].to(torch.int32)
    _tq_mla_fused_store_kernel_bf16_direct[(N,)](
        kv_c[:N],
        k_pe[:N],
        cache_flat,
        slots,
        midpoints,
        centroids_fp32,
        kpe_midpoints if kpe_midpoints is not None else midpoints,
        kpe_centroids_fp32 if kpe_centroids_fp32 is not None else centroids_fp32,
        kv_c.stride(0),
        k_pe.stride(0),
        packed_bytes,
        D,
        R,
        mse_bits,
        mse_bytes,
        kv_c_bytes,
        kpe_mse_bytes,
        block_d,
        block_r,
        _log2_const(block_d),
        _log2_const(block_r),
        1.0 / math.sqrt(block_d),
        1.0 / math.sqrt(block_r),
        1 << mse_bits,
        kpe_mode,
        norm_correction,
    )


def tq_mla_fused_kv_cache_store(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    pi_t: torch.Tensor,
    midpoints: torch.Tensor,
    centroids_fp32: torch.Tensor,
    mse_bits: int,
    mse_bytes: int,
    kv_c_bytes: int,
    packed_bytes: int,
    kpe_fp8: bool,
    kpe_4bit: bool = False,
    use_fwht: bool = False,
    kpe_midpoints: torch.Tensor | None = None,
    kpe_centroids_fp32: torch.Tensor | None = None,
    kpe_mse_bytes: int = 0,
    norm_correction: bool,
) -> None:
    """Launch fused MLA store into ``kv_cache`` (uint8, flat or blocked)."""
    N = slot_mapping.shape[0]
    if N == 0:
        return
    if mse_bits != 4:
        raise ValueError(
            f"fused MLA store supports 4-bit MSE only (got {mse_bits}-bit); "
            "set VLLM_TQ_MLA_FUSED_STORE=0 for PyTorch fallback"
        )

    D = kv_c.shape[-1]
    R = k_pe.shape[-1]
    block_d = triton.next_power_of_2(D)
    block_r = triton.next_power_of_2(R)
    assert block_d == D, f"kv_lora_rank must be power of 2, got {D}"
    assert block_r == R, f"qk_rope_head_dim must be power of 2 for FWHT, got {R}"

    if use_fwht and tq_mla_store_bf16_direct_enabled():
        _launch_fused_store_bf16_direct(
            kv_c,
            k_pe,
            kv_cache,
            slot_mapping,
            midpoints=midpoints,
            centroids_fp32=centroids_fp32,
            mse_bits=mse_bits,
            mse_bytes=mse_bytes,
            kv_c_bytes=kv_c_bytes,
            packed_bytes=packed_bytes,
            kpe_fp8=kpe_fp8,
            kpe_4bit=kpe_4bit,
            kpe_midpoints=kpe_midpoints,
            kpe_centroids_fp32=kpe_centroids_fp32,
            kpe_mse_bytes=kpe_mse_bytes,
            norm_correction=norm_correction,
        )
        return

    kv_c_f = kv_c[:N].to(torch.float32)
    norms = kv_c_f.norm(dim=1).clamp(min=1e-8)
    x_hat = (kv_c_f / norms.unsqueeze(1)).contiguous()
    norms = norms.contiguous()

    if use_fwht:
        y = x_hat  # placeholder; kernel reads x_hat directly
        stride_y = x_hat.stride(0)
    else:
        y = (x_hat @ pi_t.to(torch.float32)).contiguous()
        stride_y = y.stride(0)

    if kpe_4bit:
        kpe_mode = 2
        if kpe_midpoints is None or kpe_centroids_fp32 is None:
            raise ValueError("kpe_4bit requires kpe_midpoints and kpe_centroids_fp32")
    elif kpe_fp8:
        kpe_mode = 1
    else:
        kpe_mode = 0

    kpe_bytes = kpe_packed_bytes(R, kpe_4bit=kpe_4bit, kpe_fp8=kpe_fp8 and not kpe_4bit)
    if kpe_mse_bytes == 0 and kpe_4bit:
        kpe_mse_bytes = kpe_mse_index_bytes(R, 4)

    cache_flat = kv_cache.view(-1, packed_bytes)
    slots = slot_mapping[:N].to(torch.int32)

    grid = (N,)
    _tq_mla_fused_store_kernel[grid](
        y,
        x_hat,
        norms,
        k_pe[:N],
        cache_flat,
        slots,
        midpoints,
        centroids_fp32,
        kpe_midpoints if kpe_midpoints is not None else midpoints,
        kpe_centroids_fp32 if kpe_centroids_fp32 is not None else centroids_fp32,
        stride_y,
        x_hat.stride(0),
        norms.stride(0),
        k_pe.stride(0),
        packed_bytes,
        D,
        R,
        mse_bits,
        mse_bytes,
        kv_c_bytes,
        kpe_mse_bytes,
        kpe_bytes,
        packed_bytes,
        block_d,
        block_r,
        _log2_const(block_d),
        _log2_const(block_r),
        1.0 / math.sqrt(block_d),
        1.0 / math.sqrt(block_r),
        1 << mse_bits,
        kpe_mode,
        norm_correction,
        use_fwht,
    )
