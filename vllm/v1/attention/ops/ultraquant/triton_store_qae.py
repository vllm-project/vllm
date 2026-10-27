# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Query-aware (QAE) Triton store kernel for the ultraquant / MXFP4 KV cache.

Writes the same slot layout as ``triton_store.py`` (FP4 E2M1 codes + one UE8M0
byte per 32 elements), so the decode kernels are unchanged. Differences:

* K is rotated by a block-diagonal Hadamard-32 (the first five butterfly stages)
  instead of a full-head Hadamard, so every 32-group is encoded independently.
  Readers must rotate Q by the same block-diagonal matrix (``PiT``).
* K: per group, the E8M0 exponent is chosen among OCP e0 + {-1, 0, +1}, and up to
  ``FLIPS`` elements move to their other neighbouring E2M1 level, minimising
  d^T S d with S the calibrated 32x32 query second moment of that (head, group).
* V: no rotation; exponent chosen among the same three candidates by MSE.

Two kernels with the same objective: a per-group kernel for decode-sized writes and a
tiled kernel (one program per head/group and 16 tokens, Sigma_g shared) for prefill.
Both update S.d and the cost incrementally (a flip of element i adds dq_i * S[i, :] to
S.d and its gain to the cost), so each candidate needs only one S.d product. The tiled
kernel runs the rotation and S.d on bf16 matrix cores (S.d as a three-term bf16 split,
~fp32 accurate). Decode steps run under CUDA graphs, so the per-group kernel's latency,
not launch overhead, is what matters there (analysis/uq/kernel/bench_store_graph.py).
"""

from __future__ import annotations

from functools import cache

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.ultraquant.format import (
    UE8M0_BIAS,
    get_group_size,
    k_scales_offset,
    n_groups,
    slot_size,
    v_codes_offset,
    v_scales_offset,
)
from vllm.v1.attention.ops.ultraquant.triton_store import (
    _SUPPORTED_DTYPES,
    _kv_cache_flat,
)

QAE_FLIPS = 4
# Below this many (token, head, group) items the per-group kernel is faster than the
# tiled one (MI355X, CUDA-graph GPU time; analysis/uq/kernel/bench_store_graph.py).
_TILED_MIN_ITEMS = 3072
_TILED_BT = 16


@triton.jit
def _e2m1_sorted_idx(x):
    """bucketize(x, midpoints) with right=False, as in the reference store."""
    s = tl.zeros(x.shape, dtype=tl.int32)
    s += tl.where(x > -5.0, 1, 0)
    s += tl.where(x > -3.5, 1, 0)
    s += tl.where(x > -2.5, 1, 0)
    s += tl.where(x > -1.75, 1, 0)
    s += tl.where(x > -1.25, 1, 0)
    s += tl.where(x > -0.75, 1, 0)
    s += tl.where(x > -0.25, 1, 0)
    s += tl.where(x > 0.25, 1, 0)
    s += tl.where(x > 0.75, 1, 0)
    s += tl.where(x > 1.25, 1, 0)
    s += tl.where(x > 1.75, 1, 0)
    s += tl.where(x > 2.5, 1, 0)
    s += tl.where(x > 3.5, 1, 0)
    s += tl.where(x > 5.0, 1, 0)
    return s


@triton.jit
def _e2m1_level(s):
    """Value of sorted E2M1 index s in [0, 14]."""
    m = tl.abs(s - 7)
    mag = tl.where(
        m <= 4,
        m.to(tl.float32) * 0.5,
        tl.where(m == 5, 3.0, tl.where(m == 6, 4.0, 6.0)),
    )
    return tl.where(s < 7, -mag, mag)


@triton.jit
def _pack_store(KV_cache_ptr, code_base, scale_addr, s_idx, byte):
    """Pack 32 sorted indices into 16 bytes of E2M1 bit patterns + one E8M0 byte."""
    bits = tl.where(s_idx < 7, 15 - s_idx, s_idx - 7)
    pairs = tl.reshape(bits, [16, 2])
    shifts = tl.arange(0, 2) * 4
    packed = tl.sum((pairs & 0xF) << shifts[None, :], axis=1).to(tl.uint8)
    tl.store(KV_cache_ptr + code_base + tl.arange(0, 16), packed)
    tl.store(KV_cache_ptr + scale_addr, byte.to(tl.uint8))


@triton.jit
def _ultraquant_store_qae_group_kernel(
    Key_ptr,  # [N*H, D]
    Value_ptr,  # [N*H, D]
    KV_cache_ptr,  # flat uint8
    Slot_mapping_ptr,  # [N] int64
    Sigma_ptr,  # [H, G, 32, 32] fp32
    stride_cache_block: tl.constexpr,
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    K_SCALES_OFFSET: tl.constexpr,
    V_CODES_OFFSET: tl.constexpr,
    V_SCALES_OFFSET: tl.constexpr,
    UE8M0_BIAS_C: tl.constexpr,
    FLIPS: tl.constexpr,
):
    pid = tl.program_id(0)
    g = tl.program_id(1)
    token_idx = pid // H
    head_idx = pid % H
    slot = tl.load(Slot_mapping_ptr + token_idx)
    if slot < 0:
        return
    slot_base = (
        (slot // BLOCK_SIZE) * stride_cache_block
        + (slot % BLOCK_SIZE) * stride_cache_pos
        + head_idx * stride_cache_head
    ).to(tl.int64)

    offs = tl.arange(0, 32)
    elem = pid.to(tl.int64) * HEAD_DIM + g * 32 + offs

    # One program = one (token, head, group). Used for decode-sized writes, where
    # launch latency dominates and the tiled kernel's fixed 64-row work does not
    # pay off.
    # ── K: Hadamard-32 (butterfly stages 0..4 stay inside the group) ──
    x = tl.load(Key_ptr + elem).to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    for _stage in tl.static_range(5):
        _bmask = 1 << _stage
        _pval = tl.gather(x, (offs ^ _bmask).to(tl.int32), 0)
        _sign = tl.where((offs & _bmask) == 0, 1.0, -1.0)
        x = (_sign * x + _pval) * INV_SQRT2

    s_base = ((head_idx * N_GROUPS_C + g) * 32).to(tl.int64) * 32
    S = tl.load(Sigma_ptr + s_base + offs[:, None] * 32 + offs[None, :])  # [32, 32]
    s_diag = tl.sum(tl.where(offs[:, None] == offs[None, :], S, 0.0), axis=1)

    amax = tl.max(tl.abs(x), axis=0)
    e0 = tl.floor(tl.log2(tl.maximum(amax, 1e-30))) - 2.0
    best_s = tl.full([32], 7, tl.int32)
    best_e = e0
    best_cost = float("inf")
    for c in tl.static_range(3):
        e = e0 + (c - 1)
        sc = tl.exp2(e)
        xn = x / sc
        s = _e2m1_sorted_idx(xn)
        lv = _e2m1_level(s)
        alt = tl.minimum(tl.maximum(tl.where(xn > lv, s + 1, s - 1), 0), 14)
        dq = (_e2m1_level(alt) - lv) * sc
        d = lv * sc - x
        Sd = tl.sum(S * d[None, :], axis=1)
        cost = tl.sum(d * Sd, axis=0)
        for _f in tl.static_range(FLIPS):
            gain = 2.0 * dq * Sd + dq * dq * s_diag
            gmin = tl.min(gain, axis=0)
            imin = tl.argmin(gain, axis=0)
            hit = (offs == imin) & (gmin < 0.0)
            if gmin < 0.0:
                cost += gmin
                if _f < FLIPS - 1:
                    # S symmetric: S d gains dq_i * S[i, :]
                    dqi = tl.sum(tl.where(hit, dq, 0.0), axis=0)
                    Sd = Sd + dqi * tl.load(Sigma_ptr + s_base + imin * 32 + offs)
            d = tl.where(hit, d + dq, d)
            tmp = tl.where(hit, alt, s)
            alt = tl.where(hit, s, alt)
            s = tmp
            dq = tl.where(hit, -dq, dq)
        take = cost < best_cost
        best_s = tl.where(take, s, best_s)
        best_e = tl.where(take, e, best_e)
        best_cost = tl.where(take, cost, best_cost)
    k_zero = amax == 0.0
    best_s = tl.where(k_zero, 7, best_s)
    k_byte = tl.where(k_zero, 0, best_e.to(tl.int32) + UE8M0_BIAS_C)
    k_byte = tl.where((k_byte < 0) | (k_byte > 255), 0, k_byte)
    _pack_store(
        KV_cache_ptr,
        slot_base + g * 16,
        slot_base + K_SCALES_OFFSET + g,
        best_s,
        k_byte,
    )

    # ── V: unrotated, best of 3 exponents by MSE ──
    v = tl.load(Value_ptr + elem).to(tl.float32)
    vmax = tl.max(tl.abs(v), axis=0)
    ve0 = tl.floor(tl.log2(tl.maximum(vmax, 1e-30))) - 2.0
    vbest_s = tl.full([32], 7, tl.int32)
    vbest_e = ve0
    vbest_cost = float("inf")
    for c in tl.static_range(3):
        e = ve0 + (c - 1)
        sc = tl.exp2(e)
        s = _e2m1_sorted_idx(v / sc)
        dv = _e2m1_level(s) * sc - v
        cost = tl.sum(dv * dv, axis=0)
        take = cost < vbest_cost
        vbest_s = tl.where(take, s, vbest_s)
        vbest_e = tl.where(take, e, vbest_e)
        vbest_cost = tl.where(take, cost, vbest_cost)
    v_zero = vmax == 0.0
    vbest_s = tl.where(v_zero, 7, vbest_s)
    v_byte = tl.where(v_zero, 0, vbest_e.to(tl.int32) + UE8M0_BIAS_C)
    v_byte = tl.where((v_byte < 0) | (v_byte > 255), 0, v_byte)
    _pack_store(
        KV_cache_ptr,
        slot_base + V_CODES_OFFSET + g * 16,
        slot_base + V_SCALES_OFFSET + g,
        vbest_s,
        v_byte,
    )


@triton.jit
def _pack_store_rows(
    KV_cache_ptr, code_base, scale_addr, s_idx, byte, row_mask, BT: tl.constexpr
):
    """Pack [BT, 32] sorted indices into 16 bytes of E2M1 bit patterns per row,
    plus one E8M0 byte per row."""
    bits = tl.where(s_idx < 7, 15 - s_idx, s_idx - 7)
    pairs = tl.reshape(bits, [BT, 16, 2])
    shifts = tl.arange(0, 2) * 4
    packed = tl.sum((pairs & 0xF) << shifts[None, None, :], axis=2).to(tl.uint8)
    tl.store(
        KV_cache_ptr + code_base[:, None] + tl.arange(0, 16)[None, :],
        packed,
        mask=row_mask[:, None],
    )
    tl.store(KV_cache_ptr + scale_addr, byte.to(tl.uint8), mask=row_mask)


@triton.jit
def _ultraquant_store_qae_kernel(
    Key_ptr,  # [N*H, D]
    Value_ptr,  # [N*H, D]
    KV_cache_ptr,  # flat uint8
    Slot_mapping_ptr,  # [N] int64
    Sigma_ptr,  # [H, G, 32, 32] fp32
    N,
    stride_cache_block: tl.constexpr,
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    K_SCALES_OFFSET: tl.constexpr,
    V_CODES_OFFSET: tl.constexpr,
    V_SCALES_OFFSET: tl.constexpr,
    UE8M0_BIAS_C: tl.constexpr,
    FLIPS: tl.constexpr,
    BT: tl.constexpr,
):
    # One program = one (head, group) and BT tokens: Sigma_g is shared by all of
    # them, so the Hadamard rotation and the initial S.d run as [BT,32]x[32,32] dots.
    hg = tl.program_id(0)
    head_idx = hg // N_GROUPS_C
    g = hg % N_GROUPS_C
    tok = tl.program_id(1) * BT + tl.arange(0, BT)
    offs = tl.arange(0, 32)
    slot = tl.load(Slot_mapping_ptr + tok, mask=tok < N, other=-1)
    valid = slot >= 0
    slot = tl.where(valid, slot, 0)
    slot_base = (
        (slot // BLOCK_SIZE) * stride_cache_block
        + (slot % BLOCK_SIZE) * stride_cache_pos
        + head_idx * stride_cache_head
    ).to(tl.int64)
    elem = (tok.to(tl.int64) * H + head_idx) * HEAD_DIM + g * 32
    ld_mask = valid[:, None]

    # ── K ──
    # Sylvester Hadamard signs (-1)^popcount(i & j): exact in bf16; the bf16 keys times
    # +-1 summed in fp32 equal the fp32 rotation up to summation order.
    ij = offs[:, None] & offs[None, :]
    pc = (
        (ij & 1) + ((ij >> 1) & 1) + ((ij >> 2) & 1) + ((ij >> 3) & 1) + ((ij >> 4) & 1)
    )
    Hs = tl.where((pc & 1) == 0, 1.0, -1.0).to(tl.bfloat16)
    xr = tl.load(Key_ptr + elem[:, None] + offs[None, :], mask=ld_mask, other=0.0).to(
        tl.bfloat16
    )
    x = tl.dot(xr, Hs) * 0.17677669529663687  # 1 / sqrt(32)
    s_base = ((head_idx * N_GROUPS_C + g) * 32).to(tl.int64) * 32
    S = tl.load(Sigma_ptr + s_base + offs[:, None] * 32 + offs[None, :])
    S_hi = S.to(tl.bfloat16)
    S_lo = (S - S_hi.to(tl.float32)).to(tl.bfloat16)
    s_diag = tl.sum(tl.where(offs[:, None] == offs[None, :], S, 0.0), axis=0)

    amax = tl.max(tl.abs(x), axis=1)
    e0 = tl.floor(tl.log2(tl.maximum(amax, 1e-30))) - 2.0
    best_s = tl.full([BT, 32], 7, tl.int32)
    best_e = e0
    best_cost = tl.full([BT], float("inf"), tl.float32)
    for c in tl.static_range(3):
        e = e0 + (c - 1)
        sc = tl.exp2(e)[:, None]
        xn = x / sc
        s = _e2m1_sorted_idx(xn)
        lv = _e2m1_level(s)
        alt = tl.minimum(tl.maximum(tl.where(xn > lv, s + 1, s - 1), 0), 14)
        dq = (_e2m1_level(alt) - lv) * sc
        d = lv * sc - x
        # S symmetric: row i of d S = S d_i.
        # Three-term bf16 split (hi.hi + lo.hi + hi.lo).
        d_hi = d.to(tl.bfloat16)
        d_lo = (d - d_hi.to(tl.float32)).to(tl.bfloat16)
        Sd = tl.dot(d_hi, S_hi)
        Sd = tl.dot(d_lo, S_hi, Sd)
        Sd = tl.dot(d_hi, S_lo, Sd)
        cost = tl.sum(d * Sd, axis=1)
        for _f in tl.static_range(FLIPS):
            gain = 2.0 * dq * Sd + dq * dq * s_diag[None, :]
            gmin = tl.min(gain, axis=1)
            imin = tl.argmin(gain, axis=1)
            ok = gmin < 0.0
            hit = (offs[None, :] == imin[:, None]) & ok[:, None]
            cost = tl.where(ok, cost + gmin, cost)
            if _f < FLIPS - 1:
                dqi = tl.sum(tl.where(hit, dq, 0.0), axis=1)  # 0 where nothing flips
                Srow = tl.load(Sigma_ptr + s_base + imin[:, None] * 32 + offs[None, :])
                Sd = Sd + dqi[:, None] * Srow
            d = tl.where(hit, d + dq, d)
            tmp = tl.where(hit, alt, s)
            alt = tl.where(hit, s, alt)
            s = tmp
            dq = tl.where(hit, -dq, dq)
        take = cost < best_cost
        best_s = tl.where(take[:, None], s, best_s)
        best_e = tl.where(take, e, best_e)
        best_cost = tl.where(take, cost, best_cost)
    k_zero = amax == 0.0
    best_s = tl.where(k_zero[:, None], 7, best_s)
    k_byte = tl.where(k_zero, 0, best_e.to(tl.int32) + UE8M0_BIAS_C)
    k_byte = tl.where((k_byte < 0) | (k_byte > 255), 0, k_byte)
    _pack_store_rows(
        KV_cache_ptr,
        slot_base + g * 16,
        slot_base + K_SCALES_OFFSET + g,
        best_s,
        k_byte,
        valid,
        BT,
    )

    # ── V: unrotated, best of 3 exponents by MSE ──
    v = tl.load(Value_ptr + elem[:, None] + offs[None, :], mask=ld_mask, other=0.0).to(
        tl.float32
    )
    vmax = tl.max(tl.abs(v), axis=1)
    ve0 = tl.floor(tl.log2(tl.maximum(vmax, 1e-30))) - 2.0
    vbest_s = tl.full([BT, 32], 7, tl.int32)
    vbest_e = ve0
    vbest_cost = tl.full([BT], float("inf"), tl.float32)
    for c in tl.static_range(3):
        e = ve0 + (c - 1)
        sc = tl.exp2(e)[:, None]
        s = _e2m1_sorted_idx(v / sc)
        dv = _e2m1_level(s) * sc - v
        cost = tl.sum(dv * dv, axis=1)
        take = cost < vbest_cost
        vbest_s = tl.where(take[:, None], s, vbest_s)
        vbest_e = tl.where(take, e, vbest_e)
        vbest_cost = tl.where(take, cost, vbest_cost)
    v_zero = vmax == 0.0
    vbest_s = tl.where(v_zero[:, None], 7, vbest_s)
    v_byte = tl.where(v_zero, 0, vbest_e.to(tl.int32) + UE8M0_BIAS_C)
    v_byte = tl.where((v_byte < 0) | (v_byte > 255), 0, v_byte)
    _pack_store_rows(
        KV_cache_ptr,
        slot_base + V_CODES_OFFSET + g * 16,
        slot_base + V_SCALES_OFFSET + g,
        vbest_s,
        v_byte,
        valid,
        BT,
    )


@cache
def hadamard32_block_pit(head_dim: int, device: torch.device) -> torch.Tensor:
    """Block-diagonal normalised Hadamard-32 (symmetric), the ``PiT`` readers
    must use with this store."""
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < 32:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    h = h / 32**0.5
    return (
        torch.block_diag(*[h] * (head_dim // 32))
        .to(device=device, dtype=torch.float32)
        .contiguous()
    )


def ultraquant_store_qae(
    key: torch.Tensor,  # [N, H, D]
    value: torch.Tensor,  # [N, H, D]
    kv_cache: torch.Tensor,  # [num_blocks, block_size, H, slot]
    slot_mapping: torch.Tensor,  # [N]
    sigma: torch.Tensor,  # [H, D // 32, 32, 32] fp32, this rank's KV heads
    flips: int = QAE_FLIPS,
) -> None:
    if key.dtype not in _SUPPORTED_DTYPES or value.dtype != key.dtype:
        raise ValueError(
            f"ultraquant_store_qae: unsupported dtypes {key.dtype}/{value.dtype}"
        )
    if slot_mapping.dtype != torch.int64:
        slot_mapping = slot_mapping.to(torch.int64)
    N, H, D = key.shape
    gs = get_group_size()
    G = n_groups(D, gs)
    if kv_cache.shape[3] < slot_size(D) or kv_cache.shape[2] != H:
        raise ValueError(
            f"ultraquant_store_qae: cache shape {tuple(kv_cache.shape)} "
            f"vs key {tuple(key.shape)}"
        )
    if (
        tuple(sigma.shape) != (H, G, 32, 32)
        or sigma.dtype != torch.float32
        or not sigma.is_contiguous()
    ):
        raise ValueError(
            f"ultraquant_store_qae: sigma must be contiguous fp32 [{H}, {G}, 32, 32]"
        )
    common = dict(
        stride_cache_block=kv_cache.stride(0),
        stride_cache_pos=kv_cache.stride(1),
        stride_cache_head=kv_cache.stride(2),
        HEAD_DIM=D,
        H=H,
        BLOCK_SIZE=kv_cache.shape[1],
        N_GROUPS_C=G,
        K_SCALES_OFFSET=k_scales_offset(D, gs),
        V_CODES_OFFSET=v_codes_offset(D, gs),
        V_SCALES_OFFSET=v_scales_offset(D, gs),
        UE8M0_BIAS_C=UE8M0_BIAS,
        FLIPS=flips,
    )
    k2 = key.reshape(N * H, D).contiguous()
    v2 = value.reshape(N * H, D).contiguous()
    cache = _kv_cache_flat(kv_cache)
    if N * H * G < _TILED_MIN_ITEMS:
        _ultraquant_store_qae_group_kernel[(N * H, G)](
            k2, v2, cache, slot_mapping, sigma, **common, num_warps=1
        )
        return
    _ultraquant_store_qae_kernel[(H * G, triton.cdiv(N, _TILED_BT))](
        k2,
        v2,
        cache,
        slot_mapping,
        sigma,
        N,
        **common,
        BT=_TILED_BT,
        num_warps=1,
    )
