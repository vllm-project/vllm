# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused TurboQuant MSE dequant for MLA decode.

This module provides a single Triton kernel `_tq_mla_dequant_mse` that
collapses the per-layer hot path:
    {bit-unpack → centroid gather → optional norm-correction → vec_norm mul}
into one launch, writing the un-rotated reconstruction (`y_hat * vec_norm`)
plus the k_pe bf16 slice into a dense workspace.

The final inverse-Hadamard rotation `y_normed @ Pi` is left as a single
cuBLAS bf16 GEMM in Python — that one is already a single fast kernel
and would only complicate the Triton kernel.

Reference for the unpack + centroid gather pattern:
    `vllm/v1/attention/ops/triton_turboquant_decode.py::_tq_full_dequant_kv`
(non-MLA backend; we mirror its K-dequant branch).

Numerical equivalence with `_dequant_kv_c_mse` (PyTorch reference in
`triton_mla_tq.py`) is bit-for-bit at fp32 round-trip; bf16 ULP-level
deviations are allowed.
"""

import math
import os

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.turboquant_attn import _build_hadamard
from vllm.v1.attention.ops.triton_turboquant_mla_store import (
    _fwht_1d,
)

_BF16 = torch.bfloat16

_PI_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def _get_pi(R: int, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    key = (R, str(device), dtype)
    pi = _PI_CACHE.get(key)
    if pi is None:
        pi = _build_hadamard(R, str(device)).to(dtype)
        _PI_CACHE[key] = pi
    return pi


def _log2_const(n: int) -> int:
    assert n > 0 and (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
    return n.bit_length() - 1


def tq_mla_qpe_fwht_in_kernel_enabled() -> bool:
    """Apply Pi_R / FWHT to q_pe inside stage1 (skip Python q_pe @ Pi)."""
    return os.environ.get("VLLM_TQ_MLA_QPE_FWHT_IN_KERNEL", "1") == "1"


def _qpe_fwht_in_kernel_flag(kpe_4bit: bool) -> int:
    return 1 if kpe_4bit and tq_mla_qpe_fwht_in_kernel_enabled() else 0


@triton.jit
def _fwht_qpe_tile(
    qpe,
    mask_h,
    BLOCK_H: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    R: tl.constexpr,
    LOG2N_R: tl.constexpr,
    INV_SQRT_R: tl.constexpr,
):
    """FWHT each head row of q_pe (matches ``q_pe @ Pi_R`` for orthonormal Pi)."""
    qpe_rot = tl.zeros([BLOCK_H, BLOCK_DPE], dtype=tl.float32)
    r_offs = tl.arange(0, BLOCK_DPE)
    r_mask = r_offs < R
    for h in tl.static_range(BLOCK_H):
        h_mask = tl.arange(0, BLOCK_H) == h
        row = tl.sum(
            tl.where(h_mask[:, None] & r_mask[None, :], qpe.to(tl.float32), 0.0),
            axis=0,
        )
        row_rot = _fwht_1d(row, BLOCK_DPE, LOG2N_R, INV_SQRT_R)
        qpe_rot = tl.where(
            h_mask[:, None] & r_mask[None, :],
            row_rot[None, :],
            qpe_rot,
        )
    return qpe_rot.to(qpe.dtype)


def _rotate_qpe_for_kpe_4bit(
    q: torch.Tensor, L: int, R: int, kpe_4bit: bool
) -> torch.Tensor:
    """Map q_pe to Hadamard space: dot(q_pe, k) = dot(q_pe @ Pi_R, k_hadamard)."""
    if not kpe_4bit:
        return q
    pi = _get_pi(R, q.device, q.dtype)
    q_pe_rot = q[..., L : L + R] @ pi
    return torch.cat([q[..., :L], q_pe_rot], dim=-1)

# P1: Triton autotune for the fused stage1 decode kernel.
# Original PR hardcoded BLOCK_N=32, BLOCK_H=16, num_warps=4, num_stages=2 (mirrored
# from upstream `_fwd_grouped_kernel_stage1` defaults). For the TurboQuant 4bit /
# FP8 paths the optimal tile depends on register pressure (4bit dequant is heavy)
# and on q_head_num (varies with TP). Let Triton benchmark a small candidate set
# during warmup and pick the best per (L, R, MSE_BITS, KEY_FP8, KPE_FP8,
# q_head_num) key — these are static for a given deployment, so the choice is
# cached for the lifetime of the process and never re-benchmarked inside CUDA
# Graph capture.
#
# Set VLLM_TQ_MLA_DISABLE_AUTOTUNE=1 to fall back to the original single-config
# behaviour (useful for benchmarking before/after, or as a fast rollback).
_TQ_MLA_DECODE_DISABLE_AUTOTUNE = (
    os.environ.get("VLLM_TQ_MLA_DISABLE_AUTOTUNE", "0") == "1"
)

_TQ_MLA_DECODE_AUTOTUNE_CONFIGS = (
    [
        # PR default; identical to pre-P1 behaviour.
        triton.Config(
            {"BLOCK_N": 32, "BLOCK_H": 16}, num_warps=4, num_stages=2,
        ),
    ]
    if _TQ_MLA_DECODE_DISABLE_AUTOTUNE
    else [
        # PR default — always kept so autotune never picks something worse.
        triton.Config({"BLOCK_N": 32, "BLOCK_H": 16}, num_warps=4, num_stages=2),
        # Smaller BLOCK_N: reduces register pressure when 4bit dequant is heavy.
        triton.Config({"BLOCK_N": 16, "BLOCK_H": 16}, num_warps=4, num_stages=2),
        # Deeper pipeline to hide global-load latency behind dequant compute.
        triton.Config({"BLOCK_N": 32, "BLOCK_H": 16}, num_warps=4, num_stages=3),
        # More warps for higher SM occupancy on long sequences.
        triton.Config({"BLOCK_N": 32, "BLOCK_H": 16}, num_warps=8, num_stages=2),
        # Larger BLOCK_N: fewer iterations when memory-bound on long KV.
        triton.Config({"BLOCK_N": 64, "BLOCK_H": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64, "BLOCK_H": 16}, num_warps=8, num_stages=2),
        # Larger BLOCK_H: only useful when q_head_num is big (small TP);
        # otherwise mask-wasted but still correct.
        triton.Config({"BLOCK_N": 32, "BLOCK_H": 32}, num_warps=8, num_stages=2),
    ]
)

_TQ_MLA_SPARSE_DECODE_AUTOTUNE_CONFIGS = (
    [triton.Config({"BLOCK_N": 32, "BLOCK_H": 16}, num_warps=4, num_stages=2)]
    if _TQ_MLA_DECODE_DISABLE_AUTOTUNE
    else [
        # split64 → 32 tokens/split with BLOCK_N=32 (single inner iteration).
        triton.Config({"BLOCK_N": 32, "BLOCK_H": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 32, "BLOCK_H": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 32, "BLOCK_H": 16}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 32, "BLOCK_H": 16}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 64, "BLOCK_H": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 64, "BLOCK_H": 16}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 16, "BLOCK_H": 8}, num_warps=4, num_stages=2),
    ]
)


@triton.jit
def _tq_mla_load_fp16_scale(
    K_Cache,
    slot_base,
    byte_off,
    n_mask,
):
    """Load one fp16 scalar per token from cache[slot + byte_off : +2]."""
    lo = tl.load(K_Cache + slot_base + byte_off, mask=n_mask, other=0).to(tl.uint16)
    hi = tl.load(K_Cache + slot_base + byte_off + 1, mask=n_mask, other=0).to(tl.uint16)
    u16 = lo | (hi << 8)
    return u16.to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _tq_mla_gather_centroids(
    Centroids_ptr,
    idx,
    mask,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MSE_BITS: tl.constexpr,
):
    """Gather bf16 centroids via a 2**MSE_BITS entry table loaded once per tile.

    Uses tl.gather (not c[idx] — dynamic [] indexing fails in Triton). Compared
    to tl.load(Centroids_ptr + idx), the table usually stays in registers/L1.
    """
    n_centroids: tl.constexpr = 1 << MSE_BITS
    centroids = tl.load(Centroids_ptr + tl.arange(0, n_centroids))
    idx_flat = idx.reshape(BLOCK_DMODEL * BLOCK_N)
    y_flat = tl.gather(centroids, idx_flat, 0)
    y_hat = y_flat.reshape(BLOCK_DMODEL, BLOCK_N)
    return tl.where(mask, y_hat, 0.0)


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
def _load_kpe_4bit_tile_bulk(
    K_Cache,
    sb,
    slot_base,
    KV_C_BYTES,
    KPE_MSE_BYTES,
    BLOCK_R,
    BLOCK_N,
    R,
    Kpe_centroids_ptr,
    r_mask,
    n_mask,
):
    """Bulk-load 32B packed k_pe indices per slot, unpack in registers."""
    byte_offs = tl.arange(0, KPE_MSE_BYTES)
    pack_mask = (byte_offs[:, None] < KPE_MSE_BYTES) & n_mask[None, :]
    packed = tl.load(
        K_Cache + sb + KV_C_BYTES + byte_offs[:, None],
        mask=pack_mask,
        other=0,
        cache_modifier=".cg",
    ).to(tl.int32)

    r = tl.arange(0, BLOCK_R)[:, None]
    b = r // 2
    shift = (r % 2) * 4
    b_safe = tl.minimum(b, KPE_MSE_BYTES - 1)
    b_idx = tl.broadcast_to(b_safe, (BLOCK_R, BLOCK_N))
    row = tl.gather(packed, b_idx, axis=0)
    shift_b = tl.broadcast_to(shift, (BLOCK_R, BLOCK_N))
    idx = (row >> shift_b) & 0xF
    r_full_mask = r_mask[:, None] & n_mask[None, :]
    idx = tl.where(r_full_mask, idx, 0)

    y_hat = _tq_mla_gather_centroids(
        Kpe_centroids_ptr,
        idx,
        r_full_mask,
        BLOCK_R,
        BLOCK_N,
        4,
    )
    kpe_scale = _tq_mla_load_fp16_scale(
        K_Cache, slot_base, KV_C_BYTES + KPE_MSE_BYTES, n_mask
    )
    return (y_hat * kpe_scale[None, :]).to(tl.bfloat16)


@triton.jit
def _load_kpe_4bit_tile(
    K_Cache,
    sb,
    slot_base,
    KV_C_BYTES,
    KPE_MSE_BYTES,
    BLOCK_R,
    BLOCK_N,
    R,
    Kpe_centroids_ptr,
    r_offs,
    r_mask,
    n_mask,
):
    bit_off = r_offs[:, None] * 4
    byte_idx = bit_off // 8
    bit_shift = bit_off % 8
    umask = 15
    r_full_mask = r_mask[:, None] & n_mask[None, :]
    raw0 = tl.load(
        K_Cache + sb + KV_C_BYTES + byte_idx,
        mask=r_full_mask,
        other=0,
        cache_modifier=".cg",
    ).to(tl.int32)
    raw1 = tl.load(
        K_Cache + sb + KV_C_BYTES + byte_idx + 1,
        mask=r_full_mask,
        other=0,
        cache_modifier=".cg",
    ).to(tl.int32)
    raw16 = raw0 | (raw1 << 8)
    idx = (raw16 >> bit_shift) & umask
    y_hat = _tq_mla_gather_centroids(
        Kpe_centroids_ptr,
        idx,
        r_full_mask,
        BLOCK_R,
        BLOCK_N,
        4,
    )
    kpe_scale = _tq_mla_load_fp16_scale(
        K_Cache, slot_base, KV_C_BYTES + KPE_MSE_BYTES, n_mask
    )
    return (y_hat * kpe_scale[None, :]).to(tl.bfloat16)


@triton.jit
def _load_kpe_4bit_1d_bulk(
    Cache_ptr,
    slot_base,
    KV_C_BYTES,
    KPE_MSE_BYTES,
    BLOCK_R,
    R,
    Kpe_centroids_ptr,
    r_offs,
    r_mask,
):
    """Bulk-load 32B k_pe indices for one slot, unpack in registers."""
    byte_offs = tl.arange(0, KPE_MSE_BYTES)
    packed = tl.load(
        Cache_ptr + slot_base + KV_C_BYTES + byte_offs,
        mask=byte_offs < KPE_MSE_BYTES,
        other=0,
        cache_modifier=".cg",
    ).to(tl.int32)

    b = r_offs // 2
    shift = (r_offs % 2) * 4
    b_safe = tl.minimum(b, KPE_MSE_BYTES - 1)
    row = tl.gather(packed, b_safe, axis=0)
    idx = (row >> shift) & 0xF
    idx = tl.where(r_mask, idx, 0)

    y_hat = _tq_mla_gather_centroids_1d(
        Kpe_centroids_ptr, idx, r_mask, BLOCK_R, 4
    )
    n_lo = tl.load(
        Cache_ptr + slot_base + KV_C_BYTES + KPE_MSE_BYTES
    ).to(tl.uint16)
    n_hi = tl.load(
        Cache_ptr + slot_base + KV_C_BYTES + KPE_MSE_BYTES + 1
    ).to(tl.uint16)
    kpe_scale = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    return (y_hat * kpe_scale).to(tl.bfloat16)


@triton.jit
def _load_kpe_4bit_1d(
    Cache_ptr,
    slot_base,
    KV_C_BYTES,
    KPE_MSE_BYTES,
    BLOCK_R,
    R,
    Kpe_centroids_ptr,
    r_offs,
    r_mask,
):
    bit_off = r_offs * 4
    byte_idx = bit_off // 8
    bit_shift = bit_off % 8
    umask = 15
    raw0 = tl.load(
        Cache_ptr + slot_base + KV_C_BYTES + byte_idx, mask=r_mask, other=0
    ).to(tl.int32)
    raw1 = tl.load(
        Cache_ptr + slot_base + KV_C_BYTES + byte_idx + 1, mask=r_mask, other=0
    ).to(tl.int32)
    raw16 = raw0 | (raw1 << 8)
    idx = (raw16 >> bit_shift) & umask
    y_hat = _tq_mla_gather_centroids_1d(
        Kpe_centroids_ptr, idx, r_mask, BLOCK_R, 4
    )
    n_lo = tl.load(
        Cache_ptr + slot_base + KV_C_BYTES + KPE_MSE_BYTES
    ).to(tl.uint16)
    n_hi = tl.load(
        Cache_ptr + slot_base + KV_C_BYTES + KPE_MSE_BYTES + 1
    ).to(tl.uint16)
    kpe_scale = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    return (y_hat * kpe_scale).to(tl.bfloat16)


@triton.jit
def _tq_mla_dequant_mse(
    Cache_ptr,  # uint8: (n_active, block_size, packed_bytes)
    Centroids_ptr,  # bf16: (2**bits,) sorted Lloyd-Max codebook
    Kpe_centroids_ptr,  # bf16: (16,) when KPE_4BIT
    Out_ptr,  # bf16: (n_active, block_size, L+R)
    # Strides (in elements; cache stride in bytes since uint8)
    stride_cache_n,
    stride_cache_p,
    stride_out_n,
    stride_out_p,
    # Compile-time constants
    L: tl.constexpr,  # kv_lora_rank, e.g. 512
    R: tl.constexpr,  # qk_rope_head_dim, e.g. 64
    BLOCK_SIZE: tl.constexpr,  # cache block_size (page count)
    BLOCK_D: tl.constexpr,  # next_pow2(L)
    BLOCK_R: tl.constexpr,  # next_pow2(R)
    MSE_BITS: tl.constexpr,  # 3 or 4
    MSE_BYTES: tl.constexpr,  # ceil(L * MSE_BITS / 8)
    KV_C_BYTES: tl.constexpr,  # MSE_BYTES + 2 (vec_norm fp16)
    NORM_CORRECTION: tl.constexpr,  # 0/1
    KPE_FP8: tl.constexpr,  # 0=bf16, 1=fp8 e4m3 + per-token fp16 scale
    KPE_4BIT: tl.constexpr,
    KPE_MSE_BYTES: tl.constexpr,
    LOG2N_R: tl.constexpr,
    INV_SQRT_R: tl.constexpr,
):
    """One program = one (active_block, page).

    Writes to Out_ptr[active_idx, page, :L+R]:
      out[:L] = y_hat_normed * vec_norm   (un-rotated; caller applies @ Pi)
      out[L:] = k_pe (bf16 reinterpret of cache[..., KV_C_BYTES:])
    """
    n_idx = tl.program_id(0)  # active block index
    p_idx = tl.program_id(1)  # page within block

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < L

    # ----- Address of this slot's packed bytes in the cache -----
    slot_base = n_idx * stride_cache_n + p_idx * stride_cache_p

    # ----- Bit-unpack (1 token/program — per-dim load) -----
    bit_off = d_offs * MSE_BITS
    byte_idx = bit_off // 8
    bit_shift = bit_off % 8
    umask = (1 << MSE_BITS) - 1

    raw0 = tl.load(Cache_ptr + slot_base + byte_idx, mask=d_mask, other=0).to(tl.int32)
    raw1 = tl.load(Cache_ptr + slot_base + byte_idx + 1, mask=d_mask, other=0).to(
        tl.int32
    )
    raw16 = raw0 | (raw1 << 8)
    idx = (raw16 >> bit_shift) & umask

    y_hat = _tq_mla_gather_centroids_1d(
        Centroids_ptr, idx, d_mask, BLOCK_D, MSE_BITS
    )

    # ----- Per-token scale (fp16 @ MSE_BYTES) -----
    # NORM_CORRECTION=1: store wrote eff_scale = vec_norm / ||y_hat_raw||.
    # NORM_CORRECTION=0: raw vec_norm.
    n_lo = tl.load(Cache_ptr + slot_base + MSE_BYTES).to(tl.uint16)
    n_hi = tl.load(Cache_ptr + slot_base + MSE_BYTES + 1).to(tl.uint16)
    token_scale = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    out_kvc = (y_hat.to(tl.float32) * token_scale).to(tl.bfloat16)

    # ----- Store y_hat * vec_norm into Out[..., :L] -----
    out_base = n_idx * stride_out_n + p_idx * stride_out_p
    tl.store(Out_ptr + out_base + d_offs, out_kvc, mask=d_mask)

    # ----- Inline copy k_pe -----
    # bf16 layout: cache[KV_C_BYTES : KV_C_BYTES + 2*R] = R bf16 elems.
    # fp8 layout:  cache[KV_C_BYTES : KV_C_BYTES + R]   = R fp8 e4m3 elems,
    #              cache[KV_C_BYTES + R : KV_C_BYTES + R + 2] = fp16 scale.
    r_offs = tl.arange(0, BLOCK_R)
    r_mask = r_offs < R
    if KPE_4BIT:
        kpe_y = _load_kpe_4bit_1d(
            Cache_ptr,
            slot_base,
            KV_C_BYTES,
            KPE_MSE_BYTES,
            BLOCK_R,
            R,
            Kpe_centroids_ptr,
            r_offs,
            r_mask,
        )
        kpe_bf = _fwht_1d(kpe_y, BLOCK_R, LOG2N_R, INV_SQRT_R).to(tl.bfloat16)
    elif KPE_FP8:
        # Reload fp8 byte and bitcast to fp8e4m3 → fp32 → bf16.
        # Then multiply per-token scale.
        fp8_byte = tl.load(
            Cache_ptr + slot_base + KV_C_BYTES + r_offs,
            mask=r_mask,
            other=0,
        ).to(tl.uint8)
        # bitcast uint8 → fp8 e4m3
        kpe_fp8 = fp8_byte.to(tl.float8e4nv, bitcast=True)
        kpe_f32 = kpe_fp8.to(tl.float32)
        # Per-token fp16 scale at cache[KV_C_BYTES + R : KV_C_BYTES + R + 2]
        s_lo = tl.load(Cache_ptr + slot_base + KV_C_BYTES + R).to(tl.uint16)
        s_hi = tl.load(Cache_ptr + slot_base + KV_C_BYTES + R + 1).to(tl.uint16)
        scale_u16 = s_lo | (s_hi << 8)
        scale_f32 = scale_u16.to(tl.float16, bitcast=True).to(tl.float32)
        kpe_bf = (kpe_f32 * scale_f32).to(tl.bfloat16)
    else:
        # bf16 path: reinterpret 2 consecutive uint8 as one bf16.
        kpe_lo = tl.load(
            Cache_ptr + slot_base + KV_C_BYTES + r_offs * 2,
            mask=r_mask,
            other=0,
        ).to(tl.uint16)
        kpe_hi = tl.load(
            Cache_ptr + slot_base + KV_C_BYTES + r_offs * 2 + 1,
            mask=r_mask,
            other=0,
        ).to(tl.uint16)
        kpe_u16 = kpe_lo | (kpe_hi << 8)
        kpe_bf = kpe_u16.to(tl.bfloat16, bitcast=True)
    tl.store(Out_ptr + out_base + L + r_offs, kpe_bf, mask=r_mask)


@triton.jit
def _tq_mla_sparse_topk_gather_dequant_mse(
    Paged_cache_ptr,  # uint8: (num_blocks, page_size, packed_bytes)
    Topk_slots_ptr,  # int32: (batch, topk) global flat cache indices; -1 invalid
    Centroids_ptr,
    Kpe_centroids_ptr,
    Out_ptr,  # bf16: (batch * topk, 1, L+R)
    stride_cache_n,
    stride_cache_p,
    stride_topk_b,
    stride_topk_k,
    stride_out_n,
    stride_out_p,
    L: tl.constexpr,
    R: tl.constexpr,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KV_C_BYTES: tl.constexpr,
    NORM_CORRECTION: tl.constexpr,
    KPE_FP8: tl.constexpr,
    KPE_4BIT: tl.constexpr,
    KPE_MSE_BYTES: tl.constexpr,
    LOG2N_R: tl.constexpr,
    INV_SQRT_R: tl.constexpr,
):
    """One program = one (query_token, topk_slot): paged gather + MSE dequant."""
    b_idx = tl.program_id(0)
    k_idx = tl.program_id(1)

    global_slot = tl.load(
        Topk_slots_ptr + b_idx * stride_topk_b + k_idx * stride_topk_k
    )
    safe_slot = tl.where(global_slot >= 0, global_slot, 0)
    block_idx = safe_slot // PAGE_SIZE
    in_page = safe_slot % PAGE_SIZE
    slot_base = block_idx * stride_cache_n + in_page * stride_cache_p

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < L
    bit_off = d_offs * MSE_BITS
    byte_idx = bit_off // 8
    bit_shift = bit_off % 8
    umask = (1 << MSE_BITS) - 1

    raw0 = tl.load(
        Paged_cache_ptr + slot_base + byte_idx, mask=d_mask, other=0
    ).to(tl.int32)
    raw1 = tl.load(
        Paged_cache_ptr + slot_base + byte_idx + 1, mask=d_mask, other=0
    ).to(tl.int32)
    raw16 = raw0 | (raw1 << 8)
    idx = (raw16 >> bit_shift) & umask

    y_hat = _tq_mla_gather_centroids_1d(
        Centroids_ptr, idx, d_mask, BLOCK_D, MSE_BITS
    )

    n_lo = tl.load(Paged_cache_ptr + slot_base + MSE_BYTES).to(tl.uint16)
    n_hi = tl.load(Paged_cache_ptr + slot_base + MSE_BYTES + 1).to(tl.uint16)
    token_scale = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    out_kvc = (y_hat.to(tl.float32) * token_scale).to(tl.bfloat16)

    out_row = b_idx * TOPK + k_idx
    out_base = out_row * stride_out_n

    tl.store(Out_ptr + out_base + d_offs, out_kvc, mask=d_mask)

    r_offs = tl.arange(0, BLOCK_R)
    r_mask = r_offs < R
    if KPE_4BIT:
        kpe_y = _load_kpe_4bit_1d(
            Paged_cache_ptr,
            slot_base,
            KV_C_BYTES,
            KPE_MSE_BYTES,
            BLOCK_R,
            R,
            Kpe_centroids_ptr,
            r_offs,
            r_mask,
        )
        kpe_bf = _fwht_1d(kpe_y, BLOCK_R, LOG2N_R, INV_SQRT_R).to(tl.bfloat16)
    elif KPE_FP8:
        fp8_byte = tl.load(
            Paged_cache_ptr + slot_base + KV_C_BYTES + r_offs,
            mask=r_mask,
            other=0,
        ).to(tl.uint8)
        kpe_fp8 = fp8_byte.to(tl.float8e4nv, bitcast=True)
        kpe_f32 = kpe_fp8.to(tl.float32)
        s_lo = tl.load(Paged_cache_ptr + slot_base + KV_C_BYTES + R).to(tl.uint16)
        s_hi = tl.load(Paged_cache_ptr + slot_base + KV_C_BYTES + R + 1).to(
            tl.uint16
        )
        scale_u16 = s_lo | (s_hi << 8)
        scale_f32 = scale_u16.to(tl.float16, bitcast=True).to(tl.float32)
        kpe_bf = (kpe_f32 * scale_f32).to(tl.bfloat16)
    else:
        kpe_lo = tl.load(
            Paged_cache_ptr + slot_base + KV_C_BYTES + r_offs * 2,
            mask=r_mask,
            other=0,
        ).to(tl.uint16)
        kpe_hi = tl.load(
            Paged_cache_ptr + slot_base + KV_C_BYTES + r_offs * 2 + 1,
            mask=r_mask,
            other=0,
        ).to(tl.uint16)
        kpe_u16 = kpe_lo | (kpe_hi << 8)
        kpe_bf = kpe_u16.to(tl.bfloat16, bitcast=True)
    tl.store(Out_ptr + out_base + L + r_offs, kpe_bf, mask=r_mask)


def fused_mla_sparse_topk_gather_dequant_mse(
    paged_cache: torch.Tensor,  # (num_blocks, page_size, packed_bytes) uint8
    global_topk: torch.Tensor,  # (batch, topk) int32 global flat indices
    centroids_bf16: torch.Tensor,
    out: torch.Tensor,  # (batch * topk, 1, L+R) bf16
    *,
    page_size: int,
    L: int,
    R: int,
    mse_bits: int,
    mse_bytes: int,
    kv_c_bytes: int,
    norm_correction: bool,
    kpe_fp8: bool = False,
    kpe_4bit: bool = False,
    kpe_centroids_bf16: torch.Tensor | None = None,
) -> None:
    """Gather top-k paged slots and dequant to bf16 in one Triton launch."""
    assert paged_cache.dtype == torch.uint8
    assert global_topk.dtype == torch.int32
    assert out.dtype == _BF16
    assert centroids_bf16.dtype == _BF16
    batch, topk = global_topk.shape
    num_slots = batch * topk
    assert out.shape == (num_slots, 1, L + R), (
        f"out shape mismatch: {out.shape} vs ({num_slots}, 1, {L + R})"
    )
    if num_slots == 0:
        return

    kpe_cents = (
        kpe_centroids_bf16
        if kpe_centroids_bf16 is not None
        else centroids_bf16[:1]
    )
    kpe_mse_bytes = math.ceil(R * 4 / 8) if kpe_4bit else 0
    BLOCK_D = triton.next_power_of_2(L)
    BLOCK_R = triton.next_power_of_2(R)
    grid = (batch, topk)
    _tq_mla_sparse_topk_gather_dequant_mse[grid](
        paged_cache,
        global_topk,
        centroids_bf16,
        kpe_cents,
        out,
        paged_cache.stride(0),
        paged_cache.stride(1),
        global_topk.stride(0),
        global_topk.stride(1),
        out.stride(0),
        out.stride(1),
        L=L,
        R=R,
        TOPK=topk,
        PAGE_SIZE=page_size,
        BLOCK_D=BLOCK_D,
        BLOCK_R=BLOCK_R,
        MSE_BITS=mse_bits,
        MSE_BYTES=mse_bytes,
        KV_C_BYTES=kv_c_bytes,
        NORM_CORRECTION=1 if norm_correction else 0,
        KPE_FP8=1 if kpe_fp8 and not kpe_4bit else 0,
        KPE_4BIT=1 if kpe_4bit else 0,
        KPE_MSE_BYTES=kpe_mse_bytes,
        LOG2N_R=_log2_const(BLOCK_R),
        INV_SQRT_R=1.0 / math.sqrt(BLOCK_R),
        num_warps=int(os.environ.get("VLLM_TQ_GATHER_WARPS", "2")),
        num_stages=int(os.environ.get("VLLM_TQ_GATHER_STAGES", "2")),
    )


def fused_mla_dequant_mse(
    cache: torch.Tensor,  # (n_active, block_size, packed_bytes) uint8
    centroids_bf16: torch.Tensor,  # (2**bits,) bf16
    out: torch.Tensor,  # (n_active, block_size, L+R) bf16 (un-rotated)
    *,
    L: int,
    R: int,
    mse_bits: int,
    mse_bytes: int,
    kv_c_bytes: int,
    norm_correction: bool,
    kpe_fp8: bool = False,
    kpe_4bit: bool = False,
    kpe_centroids_bf16: torch.Tensor | None = None,
) -> None:
    """Launch the fused MSE dequant kernel.

    `out[..., :L]` receives `y_hat_normed * vec_norm` — the caller must
    apply `out[..., :L] @ Pi` (cuBLAS bf16 GEMM) to finish the inverse
    Hadamard rotation. `out[..., L:]` receives `k_pe`.
    """
    assert cache.dtype == torch.uint8
    assert out.dtype == _BF16
    assert centroids_bf16.dtype == _BF16
    n_active, block_size, _packed = cache.shape
    assert out.shape == (n_active, block_size, L + R), (
        f"out shape mismatch: {out.shape} vs ({n_active}, {block_size}, {L + R})"
    )
    if n_active == 0:
        return

    kpe_cents = (
        kpe_centroids_bf16
        if kpe_centroids_bf16 is not None
        else centroids_bf16[:1]
    )
    kpe_mse_bytes = math.ceil(R * 4 / 8) if kpe_4bit else 0

    BLOCK_D = triton.next_power_of_2(L)
    BLOCK_R = triton.next_power_of_2(R)
    grid = (n_active, block_size)
    _tq_mla_dequant_mse[grid](
        cache,
        centroids_bf16,
        kpe_cents,
        out,
        cache.stride(0),
        cache.stride(1),
        out.stride(0),
        out.stride(1),
        L=L,
        R=R,
        BLOCK_SIZE=block_size,
        BLOCK_D=BLOCK_D,
        BLOCK_R=BLOCK_R,
        MSE_BITS=mse_bits,
        MSE_BYTES=mse_bytes,
        KV_C_BYTES=kv_c_bytes,
        NORM_CORRECTION=1 if norm_correction else 0,
        KPE_FP8=1 if kpe_fp8 and not kpe_4bit else 0,
        KPE_4BIT=1 if kpe_4bit else 0,
        KPE_MSE_BYTES=kpe_mse_bytes,
        LOG2N_R=_log2_const(BLOCK_R),
        INV_SQRT_R=1.0 / math.sqrt(BLOCK_R),
        num_warps=int(os.environ.get("VLLM_TQ_GATHER_WARPS", "2")),
        num_stages=int(os.environ.get("VLLM_TQ_GATHER_STAGES", "2")),
    )


# =============================================================================
# Fused decode stage1: dequant + grouped attention in a single kernel.
#
# Mirrors `_fwd_grouped_kernel_stage1` in
# `vllm/v1/attention/ops/triton_decode_attention.py` (L261-443) but consumes
# a packed-uint8 paged TurboQuant cache directly. At each K-load site we
# inline {bit-unpack → centroid gather → vec_norm scale} instead of reading
# bf16, eliminating the per-layer bf16 dequant workspace. The query is
# expected to be pre-rotated (Πq) on the caller side — see
# `tests/kernels/attention/test_mla_turboquant_qside_rotation.py` for the
# math identity.
#
# Cache slot layout (per token in `(num_blocks, block_size, packed_bytes)`):
#     bytes [0 ... MSE_BYTES)              packed kv_c indices (MSE_BITS each)
#     bytes [MSE_BYTES ... KV_C_BYTES)     fp16 per-token scale (MSE_BYTES + 2):
#         norm_correction off: vec_norm
#         norm_correction on: eff_scale = vec_norm / ||centroid(idx)||
#     bytes [KV_C_BYTES ... )              k_pe (bf16) or k_pe (fp8 + fp16 scale)
# =============================================================================


@triton.autotune(
    configs=_TQ_MLA_DECODE_AUTOTUNE_CONFIGS,
    # Bench only when *static* deployment shape changes. batch / seq_len / etc.
    # intentionally excluded — those vary every step and re-benchmarking inside
    # CUDA Graph capture would error out.
    key=["L", "R", "MSE_BITS", "KEY_FP8", "KPE_FP8", "q_head_num"],
)
@triton.jit
def _fwd_grouped_kernel_stage1_tq(
    Q,  # bf16: (batch, q_head_num, Lk) Lk=L+R; q is Π-rotated on the L slice
    K_Cache,  # uint8: (num_blocks, block_size, packed_bytes)
    Centroids_ptr,  # bf16: (2**MSE_BITS,) Lloyd-Max codebook
    Kpe_centroids_ptr,  # bf16: (16,) for k_pe 4-bit
    sm_scale,
    Req_to_tokens,  # int32: (batch, max_kv_pages) -> page index
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_cache_n,  # bytes per page block (block_size * packed_bytes)
    stride_cache_p,  # bytes per token slot (packed_bytes)
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,  # next_pow2(L)
    BLOCK_DPE: tl.constexpr,  # next_pow2(R)
    BLOCK_DV: tl.constexpr,  # = BLOCK_DMODEL (MLA: V is L slice of kv_c)
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    L: tl.constexpr,  # kv_lora_rank
    R: tl.constexpr,  # qk_rope_head_dim
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KV_C_BYTES: tl.constexpr,
    NORM_CORRECTION: tl.constexpr,
    KPE_FP8: tl.constexpr,
    KPE_4BIT: tl.constexpr,
    KPE_MSE_BYTES: tl.constexpr,
    QPE_FWHT_IN_KERNEL: tl.constexpr,
    LOG2N_R: tl.constexpr,
    INV_SQRT_R: tl.constexpr,
    KEY_FP8: tl.constexpr,
    K_SCALE: tl.constexpr,
):
    """One program = (batch, head_block, kv_split). Heads are grouped via BLOCK_H
    just like the upstream stage1; KV is reduced one BLOCK_N tile at a time.

    KEY_FP8=1: kv_c bytes are L fp8 e4m3 elems (no Hadamard, no vec_norm,
    no centroid). The layer-global K_SCALE compile-time constant is multiplied
    into the bf16 K tile (and therefore into V via the MLA K==V identity)
    so output magnitude matches the bf16-workspace baseline that pre-scales
    the cache.
    """
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    # MLA: kv_group_num == q_head_num (single shared kv_c).
    cur_head = cur_head_id * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < q_head_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_d = offs_d < L
    offs_dpe = L + tl.arange(0, BLOCK_DPE)
    mask_dpe = offs_dpe < (L + R)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    # Load the (already Π-rotated) query.
    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(
        Q + offs_q,
        mask=mask_h[:, None] & mask_d[None, :],
        other=0.0,
        cache_modifier=".ca",
    )
    off_qpe = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
    qpe = tl.load(
        Q + off_qpe,
        mask=mask_h[:, None] & mask_dpe[None, :],
        other=0.0,
        cache_modifier=".ca",
    )
    if KPE_4BIT and QPE_FWHT_IN_KERNEL:
        qpe = _fwht_qpe_tile(
            qpe, mask_h, BLOCK_H, BLOCK_DPE, R, LOG2N_R, INV_SQRT_R
        )

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    # Bit-unpack constants.
    bit_off = offs_d * MSE_BITS  # (BLOCK_DMODEL,)
    byte_idx = bit_off // 8
    bit_shift = bit_off % 8
    umask = (1 << MSE_BITS) - 1

    if split_kv_end > split_kv_start:
        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < split_kv_end

            kv_page = tl.load(
                Req_to_tokens
                + stride_req_to_tokens_b * cur_batch
                + offs_n // PAGE_SIZE,
                mask=n_mask,
                other=0,
                cache_modifier=".ca",
            )
            kv_in_page = offs_n % PAGE_SIZE
            slot_base = kv_page * stride_cache_n + kv_in_page * stride_cache_p
            # slot_base shape (BLOCK_N,); we need (BLOCK_DMODEL, BLOCK_N).
            sb = slot_base[None, :]
            bi = byte_idx[:, None]
            bs = bit_shift[:, None]
            d_full_mask = mask_d[:, None] & n_mask[None, :]
            token_scale = tl.full([BLOCK_N], 1.0, dtype=tl.float32)

            if KEY_FP8:
                # K3: FP8 keys — load fp8 byte directly, bitcast to fp8e4nv,
                # cast to fp32, multiply by layer-global K_SCALE, cast to bf16.
                # Both qk and v share this scaled K (MLA K==V), matching the
                # bf16-workspace baseline that bakes k_scale into the cache.
                fp8_k = tl.load(
                    K_Cache + sb + offs_d[:, None],
                    mask=d_full_mask,
                    other=0,
                    cache_modifier=".cg",
                ).to(tl.uint8)
                y_hat = fp8_k.to(tl.float8e4nv, bitcast=True).to(tl.float32)
                y_hat = y_hat * K_SCALE
                k = y_hat.to(q.dtype)
                qk = tl.dot(q, k)
            else:
                # Bit-unpack (per-dim loads; bulk pack[idx] unsupported in Triton).
                raw0 = tl.load(
                    K_Cache + sb + bi,
                    mask=d_full_mask,
                    other=0,
                    cache_modifier=".cg",
                ).to(tl.int32)
                raw1 = tl.load(
                    K_Cache + sb + bi + 1,
                    mask=d_full_mask,
                    other=0,
                    cache_modifier=".cg",
                ).to(tl.int32)
                raw16 = raw0 | (raw1 << 8)
                idx = (raw16 >> bs) & umask

                y_hat = _tq_mla_gather_centroids(
                    Centroids_ptr,
                    idx,
                    d_full_mask,
                    BLOCK_DMODEL,
                    BLOCK_N,
                    MSE_BITS,
                )

                # fp16 @ MSE_BYTES: vec_norm, or eff_scale if NC precomputed at store.
                token_scale = _tq_mla_load_fp16_scale(
                    K_Cache, slot_base, MSE_BYTES, n_mask
                )

                k = y_hat
                qk = tl.dot(q, k)
                qk = qk * token_scale[None, :]

            # ----- k_pe path -----
            r_offs = tl.arange(0, BLOCK_DPE)
            r_mask = r_offs < R
            r_full_mask = r_mask[:, None] & n_mask[None, :]
            if KPE_4BIT:
                kpe = _load_kpe_4bit_tile(
                    K_Cache,
                    sb,
                    slot_base,
                    KV_C_BYTES,
                    KPE_MSE_BYTES,
                    BLOCK_DPE,
                    BLOCK_N,
                    R,
                    Kpe_centroids_ptr,
                    r_offs,
                    r_mask,
                    n_mask,
                )
            elif KPE_FP8:
                fp8_byte = tl.load(
                    K_Cache + sb + KV_C_BYTES + r_offs[:, None],
                    mask=r_full_mask,
                    other=0,
                    cache_modifier=".cg",
                ).to(tl.uint8)
                kpe_f32 = fp8_byte.to(tl.float8e4nv, bitcast=True).to(tl.float32)
                s_lo = tl.load(
                    K_Cache + slot_base + KV_C_BYTES + R, mask=n_mask, other=0
                ).to(tl.uint16)
                s_hi = tl.load(
                    K_Cache + slot_base + KV_C_BYTES + R + 1, mask=n_mask, other=0
                ).to(tl.uint16)
                scale = (s_lo | (s_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
                kpe = (kpe_f32 * scale[None, :]).to(qpe.dtype)
            else:
                kpe_lo = tl.load(
                    K_Cache + sb + KV_C_BYTES + r_offs[:, None] * 2,
                    mask=r_full_mask,
                    other=0,
                    cache_modifier=".cg",
                ).to(tl.uint16)
                kpe_hi = tl.load(
                    K_Cache + sb + KV_C_BYTES + r_offs[:, None] * 2 + 1,
                    mask=r_full_mask,
                    other=0,
                    cache_modifier=".cg",
                ).to(tl.uint16)
                kpe_u16 = kpe_lo | (kpe_hi << 8)
                kpe = kpe_u16.to(tl.bfloat16, bitcast=True).to(qpe.dtype)

            qk += tl.dot(qpe, kpe)
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * (
                    (tl.exp(qk / logit_cap) - tl.exp(-qk / logit_cap))
                    / (tl.exp(qk / logit_cap) + tl.exp(-qk / logit_cap))
                )

            qk = tl.where(mask_h[:, None] & n_mask[None, :], qk, float("-inf"))

            # MLA reuses k as v (transposed).
            v = tl.trans(k)

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            # K1-a value-path correction: in the MSE branch K is stored as
            # un-scaled centroid `y_hat` (vec_norm is folded into qk *after*
            # the dot to save a (BLOCK_DMODEL, BLOCK_N) multiply). Since
            # `v = trans(k) = trans(y_hat)` is also missing vec_norm, the
            # accumulator must apply vec_norm here — mathematically:
            #   sum_n p_n * (y_hat_n * vn_n) = sum_n (p_n * vn_n) * y_hat_n
            # so we scale p (shape (BLOCK_H, BLOCK_N), small) instead of v.
            # KEY_FP8 path bakes K_SCALE into y_hat before the cast, so v
            # already carries the correct scale and no correction is needed.
            if KEY_FP8:
                p_for_v = p.to(v.dtype)
            else:
                p_for_v = (p * token_scale[None, :]).to(v.dtype)
            acc += tl.dot(p_for_v, v)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_d[None, :]
        )
        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=mask_h[:, None] & mask_d[None, :],
        )
        offs_lse = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + L
        )
        tl.store(Att_Out + offs_lse, e_max + tl.log(e_sum), mask=mask_h)


def fused_mla_tq_decode_stage1(
    q: torch.Tensor,  # bf16: (batch, q_head_num, L+R), Π-rotated on [:L]
    cache: torch.Tensor,  # uint8: (num_blocks, block_size, packed_bytes)
    centroids_bf16: torch.Tensor,  # bf16: (2**MSE_BITS,) or empty when key_fp8
    att_out: torch.Tensor,  # fp32: (batch, q_head_num, NUM_KV_SPLITS, L+1)
    req_to_tokens: torch.Tensor,  # int32: (batch, max_kv_pages)
    b_seqlen: torch.Tensor,  # int32: (batch,)
    *,
    sm_scale: float,
    page_size: int,
    L: int,
    R: int,
    mse_bits: int,
    mse_bytes: int,
    kv_c_bytes: int,
    norm_correction: bool,
    kpe_fp8: bool,
    kpe_4bit: bool = False,
    kpe_centroids_bf16: torch.Tensor | None = None,
    key_fp8: bool = False,
    k_scale: float = 1.0,
    num_kv_splits: int = 4,
    logit_cap: float = 0.0,
) -> None:
    """Launch the fused TurboQuant MLA decode stage1.

    MSE keys (default): `q` must be the standard MLA decode query with the
    Hadamard rotation applied to its first L (kv_lora_rank) elements; the
    kernel does no rotation internally.

    FP8 keys (`key_fp8=True`): cache layout is `[L bytes fp8 e4m3 | k_pe...]`.
    No Hadamard rotation; q is consumed as-is. The layer-global `k_scale`
    is passed as a kernel constexpr and multiplied into the bf16 K tile
    inside the kernel (so V = trans(K) inherits the same scale, preserving
    MLA K==V semantics). centroids_bf16 is unused in this mode (pass any
    bf16 1-D tensor; e.g. an empty one).
    """
    assert cache.dtype == torch.uint8
    assert q.dtype == torch.bfloat16
    assert centroids_bf16.dtype == torch.bfloat16
    batch, q_head_num, head_dim = q.shape
    assert head_dim == L + R, f"q head_dim {head_dim} != L+R {L + R}"

    if kpe_4bit and not tq_mla_qpe_fwht_in_kernel_enabled():
        q = _rotate_qpe_for_kpe_4bit(q, L, R, kpe_4bit)

    BLOCK_DMODEL = triton.next_power_of_2(L)
    BLOCK_DPE = triton.next_power_of_2(R)
    BLOCK_DV = BLOCK_DMODEL
    kpe_cents = (
        kpe_centroids_bf16
        if kpe_centroids_bf16 is not None
        else centroids_bf16[:1]
    )
    kpe_mse_bytes = math.ceil(R * 4 / 8) if kpe_4bit else 0

    # P1: BLOCK_N / BLOCK_H / num_warps / num_stages are now provided by the
    # autotune Config chosen at first call. Grid's second dim depends on the
    # autotuned BLOCK_H, so it must be a callable (`meta` carries the picked
    # config). Same pattern as vllm/model_executor/layers/fla/ops/chunk_o.py.
    def _grid(meta):
        return (
            batch,
            triton.cdiv(q_head_num, meta["BLOCK_H"]),
            num_kv_splits,
        )

    # FP8 path: K_SCALE is a constexpr multiplied into the K tile (and so
    # into V) inside the kernel. MSE path: K_SCALE unused (DCE'd by Triton).
    _fwd_grouped_kernel_stage1_tq[_grid](
        q,
        cache,
        centroids_bf16,
        kpe_cents,
        sm_scale,
        req_to_tokens,
        b_seqlen,
        att_out,
        req_to_tokens.stride(0),
        q.stride(0),
        q.stride(1),
        cache.stride(0),
        cache.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        q_head_num=q_head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        # BLOCK_N, BLOCK_H, num_warps, num_stages: injected by autotune Config.
        NUM_KV_SPLITS=num_kv_splits,
        PAGE_SIZE=page_size,
        logit_cap=logit_cap,
        L=L,
        R=R,
        MSE_BITS=mse_bits,
        MSE_BYTES=mse_bytes,
        KV_C_BYTES=kv_c_bytes,
        NORM_CORRECTION=1 if norm_correction else 0,
        KPE_FP8=1 if kpe_fp8 and not kpe_4bit else 0,
        KPE_4BIT=1 if kpe_4bit else 0,
        KPE_MSE_BYTES=kpe_mse_bytes,
        QPE_FWHT_IN_KERNEL=_qpe_fwht_in_kernel_flag(kpe_4bit),
        LOG2N_R=_log2_const(BLOCK_DPE),
        INV_SQRT_R=1.0 / math.sqrt(BLOCK_DPE),
        KEY_FP8=1 if key_fp8 else 0,
        K_SCALE=k_scale,
    )


# =============================================================================
# Fused sparse decode stage1: inline dequant over indexer top-k global slots.
#
# Same inner dequant/attention loop as `_fwd_grouped_kernel_stage1_tq`, but
# KV positions come from `Topk_slots` (global flat cache indices from
# `triton_convert_req_index_to_global_index`) instead of a dense page table +
# seq_len walk. Invalid slots (-1) are masked out of the softmax.
# =============================================================================


@triton.autotune(
    configs=_TQ_MLA_SPARSE_DECODE_AUTOTUNE_CONFIGS,
    key=["L", "R", "MSE_BITS", "KEY_FP8", "KPE_FP8", "q_head_num", "TOPK", "NUM_KV_SPLITS"],
)
@triton.jit
def _fwd_grouped_kernel_stage1_tq_sparse(
    Q,
    K_Cache,
    Centroids_ptr,
    Kpe_centroids_ptr,
    sm_scale,
    Topk_slots,  # int32: (batch, TOPK) global flat cache indices; -1 = invalid
    Att_Out,
    stride_topk_b,
    stride_qbs,
    stride_qh,
    stride_cache_n,
    stride_cache_p,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    logit_cap: tl.constexpr,
    L: tl.constexpr,
    R: tl.constexpr,
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KV_C_BYTES: tl.constexpr,
    NORM_CORRECTION: tl.constexpr,
    KPE_FP8: tl.constexpr,
    KPE_4BIT: tl.constexpr,
    KPE_MSE_BYTES: tl.constexpr,
    QPE_FWHT_IN_KERNEL: tl.constexpr,
    LOG2N_R: tl.constexpr,
    INV_SQRT_R: tl.constexpr,
    KEY_FP8: tl.constexpr,
    K_SCALE: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_head = cur_head_id * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < q_head_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_d = offs_d < L
    offs_dpe = L + tl.arange(0, BLOCK_DPE)
    mask_dpe = offs_dpe < (L + R)

    cur_batch_seq_len = TOPK

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(
        Q + offs_q,
        mask=mask_h[:, None] & mask_d[None, :],
        other=0.0,
        cache_modifier=".ca",
    )
    off_qpe = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
    qpe = tl.load(
        Q + off_qpe,
        mask=mask_h[:, None] & mask_dpe[None, :],
        other=0.0,
        cache_modifier=".ca",
    )
    if KPE_4BIT and QPE_FWHT_IN_KERNEL:
        qpe = _fwht_qpe_tile(
            qpe, mask_h, BLOCK_H, BLOCK_DPE, R, LOG2N_R, INV_SQRT_R
        )

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    bit_off = offs_d * MSE_BITS
    byte_idx = bit_off // 8
    bit_shift = bit_off % 8
    umask = (1 << MSE_BITS) - 1

    if split_kv_end > split_kv_start:
        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < split_kv_end

            global_slot = tl.load(
                Topk_slots + cur_batch * stride_topk_b + offs_n,
                mask=n_mask,
                other=-1,
            )
            valid_slot = global_slot >= 0
            n_mask = n_mask & valid_slot
            safe_slot = tl.where(valid_slot, global_slot, 0)
            block_idx = safe_slot // PAGE_SIZE
            in_page = safe_slot % PAGE_SIZE
            slot_base = block_idx * stride_cache_n + in_page * stride_cache_p
            sb = slot_base[None, :]
            bi = byte_idx[:, None]
            bs = bit_shift[:, None]
            d_full_mask = mask_d[:, None] & n_mask[None, :]
            token_scale = tl.full([BLOCK_N], 1.0, dtype=tl.float32)

            if KEY_FP8:
                fp8_k = tl.load(
                    K_Cache + sb + offs_d[:, None],
                    mask=d_full_mask,
                    other=0,
                    cache_modifier=".cg",
                ).to(tl.uint8)
                y_hat = fp8_k.to(tl.float8e4nv, bitcast=True).to(tl.float32)
                y_hat = y_hat * K_SCALE
                k = y_hat.to(q.dtype)
                qk = tl.dot(q, k)
            else:
                raw0 = tl.load(
                    K_Cache + sb + bi,
                    mask=d_full_mask,
                    other=0,
                    cache_modifier=".cg",
                ).to(tl.int32)
                raw1 = tl.load(
                    K_Cache + sb + bi + 1,
                    mask=d_full_mask,
                    other=0,
                    cache_modifier=".cg",
                ).to(tl.int32)
                raw16 = raw0 | (raw1 << 8)
                idx = (raw16 >> bs) & umask

                y_hat = _tq_mla_gather_centroids(
                    Centroids_ptr,
                    idx,
                    d_full_mask,
                    BLOCK_DMODEL,
                    BLOCK_N,
                    MSE_BITS,
                )

                token_scale = _tq_mla_load_fp16_scale(
                    K_Cache, slot_base, MSE_BYTES, n_mask
                )

                # Hybrid sparse: caller pre-rotates q with Pi; keep y_hat unrotated
                # here (equiv. to legacy apply_pi_on_k on K outside the kernel).
                k = y_hat
                qk = tl.dot(q, k)
                qk = qk * token_scale[None, :]

            r_offs = tl.arange(0, BLOCK_DPE)
            r_mask = r_offs < R
            r_full_mask = r_mask[:, None] & n_mask[None, :]
            if KPE_4BIT:
                kpe = _load_kpe_4bit_tile(
                    K_Cache,
                    sb,
                    slot_base,
                    KV_C_BYTES,
                    KPE_MSE_BYTES,
                    BLOCK_DPE,
                    BLOCK_N,
                    R,
                    Kpe_centroids_ptr,
                    r_offs,
                    r_mask,
                    n_mask,
                )
            elif KPE_FP8:
                fp8_byte = tl.load(
                    K_Cache + sb + KV_C_BYTES + r_offs[:, None],
                    mask=r_full_mask,
                    other=0,
                    cache_modifier=".cg",
                ).to(tl.uint8)
                kpe_f32 = fp8_byte.to(tl.float8e4nv, bitcast=True).to(tl.float32)
                s_lo = tl.load(
                    K_Cache + slot_base + KV_C_BYTES + R, mask=n_mask, other=0
                ).to(tl.uint16)
                s_hi = tl.load(
                    K_Cache + slot_base + KV_C_BYTES + R + 1, mask=n_mask, other=0
                ).to(tl.uint16)
                scale = (s_lo | (s_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
                kpe = (kpe_f32 * scale[None, :]).to(qpe.dtype)
            else:
                kpe_lo = tl.load(
                    K_Cache + sb + KV_C_BYTES + r_offs[:, None] * 2,
                    mask=r_full_mask,
                    other=0,
                    cache_modifier=".cg",
                ).to(tl.uint16)
                kpe_hi = tl.load(
                    K_Cache + sb + KV_C_BYTES + r_offs[:, None] * 2 + 1,
                    mask=r_full_mask,
                    other=0,
                    cache_modifier=".cg",
                ).to(tl.uint16)
                kpe_u16 = kpe_lo | (kpe_hi << 8)
                kpe = kpe_u16.to(tl.bfloat16, bitcast=True).to(qpe.dtype)

            qk += tl.dot(qpe, kpe)
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * (
                    (tl.exp(qk / logit_cap) - tl.exp(-qk / logit_cap))
                    / (tl.exp(qk / logit_cap) + tl.exp(-qk / logit_cap))
                )

            qk = tl.where(mask_h[:, None] & n_mask[None, :], qk, float("-inf"))

            v = tl.trans(k)

            # Skip softmax update when a row's tile is all masked (-inf only).
            # Otherwise exp(e_max - n_e_max) becomes exp(-inf - (-inf)) = NaN.
            row_max = tl.max(qk, 1)
            has_active = row_max > float("-inf")
            n_e_max = tl.maximum(row_max, e_max)
            safe_n_e_max = tl.where(has_active, n_e_max, 0.0)
            re_scale = tl.where(has_active, tl.exp(e_max - safe_n_e_max), 1.0)
            p = tl.where(
                has_active[:, None],
                tl.exp(qk - safe_n_e_max[:, None]),
                0.0,
            )
            acc *= re_scale[:, None]
            if KEY_FP8:
                p_for_v = p.to(v.dtype)
            else:
                p_for_v = (p * token_scale[None, :]).to(v.dtype)
            acc += tl.dot(p_for_v, v)
            e_sum = tl.where(has_active, e_sum * re_scale + tl.sum(p, 1), e_sum)
            e_max = tl.where(has_active, n_e_max, e_max)

        safe_e_sum = tl.maximum(e_sum, 1e-6)
        valid_split = e_max > float("-inf")
        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_d[None, :]
        )
        tl.store(
            Att_Out + offs_mid_o,
            tl.where(valid_split[:, None], acc / safe_e_sum[:, None], 0.0),
            mask=mask_h[:, None] & mask_d[None, :],
        )
        offs_lse = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + L
        )
        tl.store(
            Att_Out + offs_lse,
            tl.where(valid_split, e_max + tl.log(safe_e_sum), float("-inf")),
            mask=mask_h,
        )


def fused_mla_tq_sparse_decode_stage1(
    q: torch.Tensor,  # bf16: (batch, q_head_num, L+R)
    cache: torch.Tensor,  # uint8: (num_blocks, block_size, packed_bytes)
    centroids_bf16: torch.Tensor,
    att_out: torch.Tensor,  # fp32: (batch, q_head_num, NUM_KV_SPLITS, L+1)
    topk_slots: torch.Tensor,  # int32: (batch, TOPK) global flat cache indices
    b_seqlen: torch.Tensor,  # int32: (batch,) — should be TOPK for stage2 splits
    *,
    sm_scale: float,
    page_size: int,
    topk: int,
    L: int,
    R: int,
    mse_bits: int,
    mse_bytes: int,
    kv_c_bytes: int,
    norm_correction: bool,
    kpe_fp8: bool,
    kpe_4bit: bool = False,
    kpe_centroids_bf16: torch.Tensor | None = None,
    key_fp8: bool = False,
    k_scale: float = 1.0,
    num_kv_splits: int = 4,
    logit_cap: float = 0.0,
) -> None:
    """Launch fused TurboQuant MLA sparse decode stage1 over top-k slots."""
    assert cache.dtype == torch.uint8
    assert q.dtype == torch.bfloat16
    assert centroids_bf16.dtype == torch.bfloat16
    assert topk_slots.dtype == torch.int32
    batch, q_head_num, head_dim = q.shape
    assert head_dim == L + R, f"q head_dim {head_dim} != L+R {L + R}"
    assert topk_slots.shape == (batch, topk)

    if kpe_4bit and not tq_mla_qpe_fwht_in_kernel_enabled():
        q = _rotate_qpe_for_kpe_4bit(q, L, R, kpe_4bit)

    BLOCK_DMODEL = triton.next_power_of_2(L)
    BLOCK_DPE = triton.next_power_of_2(R)
    BLOCK_DV = BLOCK_DMODEL
    kpe_cents = (
        kpe_centroids_bf16
        if kpe_centroids_bf16 is not None
        else centroids_bf16[:1]
    )
    kpe_mse_bytes = math.ceil(R * 4 / 8) if kpe_4bit else 0

    def _grid(meta):
        return (
            batch,
            triton.cdiv(q_head_num, meta["BLOCK_H"]),
            num_kv_splits,
        )

    _fwd_grouped_kernel_stage1_tq_sparse[_grid](
        q,
        cache,
        centroids_bf16,
        kpe_cents,
        sm_scale,
        topk_slots,
        att_out,
        topk_slots.stride(0),
        q.stride(0),
        q.stride(1),
        cache.stride(0),
        cache.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        q_head_num=q_head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        NUM_KV_SPLITS=num_kv_splits,
        PAGE_SIZE=page_size,
        TOPK=topk,
        logit_cap=logit_cap,
        L=L,
        R=R,
        MSE_BITS=mse_bits,
        MSE_BYTES=mse_bytes,
        KV_C_BYTES=kv_c_bytes,
        NORM_CORRECTION=1 if norm_correction else 0,
        KPE_FP8=1 if kpe_fp8 and not kpe_4bit else 0,
        KPE_4BIT=1 if kpe_4bit else 0,
        KPE_MSE_BYTES=kpe_mse_bytes,
        QPE_FWHT_IN_KERNEL=_qpe_fwht_in_kernel_flag(kpe_4bit),
        LOG2N_R=_log2_const(BLOCK_DPE),
        INV_SQRT_R=1.0 / math.sqrt(BLOCK_DPE),
        KEY_FP8=1 if key_fp8 else 0,
        K_SCALE=k_scale,
    )


def tq_mla_sparse_adaptive_enabled() -> bool:
    return os.environ.get("VLLM_TQ_MLA_SPARSE_ADAPTIVE", "1") == "1"


def tq_mla_sparse_split_count(
    topk: int, sm_count: int, batch: int | None = None
) -> int:
    """Resolve sparse decode split count (aligned with vllm-tqa).

    Priority:
      1) fixed override (``VLLM_TQ_MLA_SPARSE_SPLITS``)
      2) adaptive policy (when ``batch`` given): keep ``B * splits`` bounded
      3) static default (~64 for topk=2048, bounded by ``2*SM``)
    """
    fixed = os.environ.get("VLLM_TQ_MLA_SPARSE_SPLITS", "")
    if fixed.isdigit() and int(fixed) > 0:
        return max(1, min(int(fixed), topk, sm_count * 2))

    if batch is not None and tq_mla_sparse_adaptive_enabled():
        # Adaptive: target ~OVERSUB*sm_count total grid programs (B x splits).
        # batch large -> splits small; batch small -> splits up to 64.
        # f32 scratch scales with B*splits, not B*64.
        target = sm_count * 4  # caller passes sm_count*2 => ~8x SM oversubscribe
        ideal = max(1, target // max(1, batch))
        ideal = 1 << (ideal.bit_length() - 1)  # floor to power of 2
        return max(1, min(ideal, 64, topk))

    ideal = max(1, topk // 32)
    ideal = 1 << (ideal - 1).bit_length()
    return min(ideal, sm_count * 2)


def sparse_decode_softmax_reducev_fwd(
    logits: torch.Tensor,
    q: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    v_buffer: torch.Tensor,
    b_seq_len: torch.Tensor,
    num_kv_splits: int,
) -> None:
    """Serial sparse stage2 softmax + reduce-v."""
    from vllm.v1.attention.ops.triton_decode_attention import (
        _decode_softmax_reducev_fwd,
    )

    _decode_softmax_reducev_fwd(
        logits, q, o, lse, v_buffer, b_seq_len, num_kv_splits
    )
