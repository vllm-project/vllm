# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton store kernel for the ultraquant KV cache format.

Per-token, per-head, per-group-of-32 the kernel computes:

    s_raw = c * absmax(group)            # c = 0.156 by default
    s     = 2^round(log2(s_raw))         # UE8M0 snap (power of 2)
    code  = quantize_to_fp4(group / s)   # OCP E2M1, 4 bits/elt

and stores ``code`` as packed nibbles + ``s`` as a single UE8M0 byte per
group. The output layout matches ``format.py`` (272 B/slot for D=256).

K is Hadamard-rotated in-register; V is not. There is no per-token L2
norm. The UE8M0 snap makes ``s`` a power of two so scaled F8F6F4 MFMA
can consume it on the read side.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.ultraquant.format import (
    MIDPOINTS_SORTED,
    SORTED_TO_BITS,
    UE8M0_BIAS,
    get_constant_c,
    get_group_size,
    k_scales_offset,
    n_groups,
    slot_size,
    v_codes_offset,
    v_scales_offset,
)


def _kv_cache_flat(kv_cache: torch.Tensor) -> torch.Tensor:
    """1-D uint8 view of the raw KV allocation (handles padded layouts)."""
    if kv_cache.is_contiguous():
        return kv_cache.reshape(-1)
    n = kv_cache.untyped_storage().nbytes() // kv_cache.element_size()
    return torch.empty(0, dtype=kv_cache.dtype, device=kv_cache.device).set_(
        kv_cache.untyped_storage(), 0, (n,)
    )


_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}

# Built lazily on first launch so importing this module does not require triton.
_MIDPOINTS_C = None


def _ensure_const_tables() -> None:
    global _MIDPOINTS_C
    if _MIDPOINTS_C is None:
        _MIDPOINTS_C = tl.constexpr(tuple(float(m) for m in MIDPOINTS_SORTED))


# ── Cached per-device constant tensors ─────────────────────────────────────
_MIDPOINTS_CACHE: dict[torch.device, torch.Tensor] = {}
_SORTED_TO_BITS_CACHE: dict[torch.device, torch.Tensor] = {}
_PIT_CACHE: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def _get_midpoints_tensor(device: torch.device) -> torch.Tensor:
    t = _MIDPOINTS_CACHE.get(device)
    if t is None:
        t = torch.tensor(MIDPOINTS_SORTED, device=device, dtype=torch.float32)
        _MIDPOINTS_CACHE[device] = t
    return t


def _get_sorted_to_bits_tensor(device: torch.device) -> torch.Tensor:
    t = _SORTED_TO_BITS_CACHE.get(device)
    if t is None:
        t = torch.tensor(SORTED_TO_BITS, device=device, dtype=torch.int32)
        _SORTED_TO_BITS_CACHE[device] = t
    return t


def _get_hadamard(
    dim: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Sylvester Hadamard in fp32 (cached, normalised)."""
    key = (dim, device, dtype)
    cached = _PIT_CACHE.get(key)
    if cached is not None:
        return cached
    if dim <= 0 or (dim & (dim - 1)) != 0:
        raise ValueError(f"ultraquant store requires power-of-two dim, got {dim}")
    H = torch.tensor([[1.0]], dtype=torch.float64)
    while H.shape[0] < dim:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    H = (H / (dim**0.5)).to(device=device, dtype=dtype).contiguous()
    _PIT_CACHE[key] = H
    return H


# ═══════════════════════════════════════════════════════════════════════════
# Triton kernel
# ═══════════════════════════════════════════════════════════════════════════


@triton.jit
def _ultraquant_store_kernel(
    Key_ptr,  # [N*H, D] in raw dtype (bf16 or fp16)
    Value_ptr,  # [N*H, D] in raw dtype
    KV_cache_ptr,  # raw uint8 view, flat
    Slot_mapping_ptr,  # [N] int64
    Midpoints_ptr,  # [14] fp32
    SortedToBits_ptr,  # [15] int32
    # Cache strides (in bytes)
    stride_cache_block: tl.constexpr,
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
    # Dimensions
    HEAD_DIM: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOG2_D: tl.constexpr,
    # Layout
    GROUP_SIZE_C: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    K_SCALES_OFFSET: tl.constexpr,
    V_CODES_OFFSET: tl.constexpr,
    V_SCALES_OFFSET: tl.constexpr,
    FP4_C: tl.constexpr,
    UE8M0_BIAS_C: tl.constexpr,
    # When 1, the FP4 midpoints and the sorted->bits remap are materialised as
    # immediates instead of being read from `Midpoints_ptr`/`SortedToBits_ptr`.
    # That removes 28 scalar global loads and two 256-lane divergent gathers
    # from the critical path of a kernel whose runtime is pure latency (flat at
    # ~22 us from batch 1 to batch 64). Numerically identical: the immediates
    # are the same fp32 values, and the remap formula is exact (asserted in
    # `format`, verified for all 15 entries).
    CONST_TABLES: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    token_idx = pid // H
    head_idx = pid % H

    slot = tl.load(Slot_mapping_ptr + token_idx)
    if slot < 0:
        return
    blk = slot // BLOCK_SIZE
    off = slot % BLOCK_SIZE
    slot_base = (
        blk * stride_cache_block + off * stride_cache_pos + head_idx * stride_cache_head
    ).to(tl.int64)
    k_code_base = slot_base
    k_scale_base = slot_base + K_SCALES_OFFSET
    v_code_base = slot_base + V_CODES_OFFSET
    v_scale_base = slot_base + V_SCALES_OFFSET

    base = pid * HEAD_DIM
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    shifts_4 = tl.arange(0, 2) * 4  # 0, 4 for nibble packing
    g_offs = tl.arange(0, N_GROUPS_C)

    # ── K side ─────────────────────────────────────────────────────────
    # 1. Load + Hadamard-rotate K via WHT butterfly.
    k_raw = tl.load(Key_ptr + base + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    k_rot = tl.where(d_mask, k_raw, 0.0)

    INV_SQRT2: tl.constexpr = 0.7071067811865476
    for _stage in tl.static_range(LOG2_D):
        _bmask = 1 << _stage
        _pidx = (d_offs ^ _bmask).to(tl.int32)
        _pval = tl.gather(k_rot, _pidx, 0)
        _sign = tl.where((d_offs & _bmask) == 0, 1.0, -1.0).to(tl.float32)
        k_rot = (_sign * k_rot + _pval) * INV_SQRT2
    k_rot = tl.where(d_mask, k_rot, 0.0)

    # 2. Per-group absmax.
    k_g = tl.reshape(k_rot, [N_GROUPS_C, GROUP_SIZE_C])
    k_absmax = tl.max(tl.abs(k_g), axis=1)  # [N_GROUPS]
    k_is_zero = k_absmax == 0.0

    # 3. UE8M0 snap: s_raw = c * absmax; exp = round(log2(s_raw));
    #    divisor = 2^exp (c folded in). Zero groups write byte=0.
    k_s_raw = k_absmax * tl.cast(FP4_C, tl.float32)
    k_s_safe = tl.where(k_is_zero, 1.0, k_s_raw)
    k_log2 = tl.log2(k_s_safe)
    k_exp = tl.cast(tl.floor(k_log2 + 0.5), tl.int32)  # round-half-up
    k_s_snapped = tl.exp2(tl.cast(k_exp, tl.float32))
    k_s_div = tl.where(k_is_zero, 1.0, k_s_snapped)

    # 4. Normalize + snap to sorted FP4 idx via 14 sequential tl.where (`>` for
    #    bucketize(right=False) parity with the reference).
    k_norm = k_g / k_s_div[:, None]
    k_sorted = tl.zeros([N_GROUPS_C, GROUP_SIZE_C], dtype=tl.int32)
    for i in tl.static_range(14):
        mid = _MIDPOINTS_C.value[i] if CONST_TABLES else tl.load(Midpoints_ptr + i)
        k_sorted += tl.where(k_norm > mid, 1, 0)
    k_sorted = tl.where(k_is_zero[:, None], 7, k_sorted)

    # 5. Remap sorted idx → FP4 E2M1 bit pattern.
    if CONST_TABLES:
        k_bits = tl.where(k_sorted < 7, 15 - k_sorted, k_sorted - 7)
    else:
        k_bits = tl.load(SortedToBits_ptr + k_sorted)

    # 6. Pack pairs of 4-bit codes into uint8 bytes.
    k_bits_flat = tl.reshape(k_bits, [HEAD_DIM])
    k_pairs = tl.reshape(k_bits_flat, [HEAD_DIM // 2, 2])
    k_packed = tl.sum((k_pairs & 0xF) << shifts_4[None, :], axis=1).to(tl.uint8)
    k_code_addrs = k_code_base + tl.arange(0, HEAD_DIM // 2)
    tl.store(KV_cache_ptr + k_code_addrs, k_packed)

    # 7. Store K scales as E8M0 bytes (one per group). Zero-amax groups
    #    write byte=0 (zero sentinel).
    k_byte = tl.where(k_is_zero, 0, k_exp + UE8M0_BIAS_C)
    k_byte = tl.where((k_byte < 0) | (k_byte > 255), 0, k_byte).to(tl.uint8)
    k_scale_addrs = k_scale_base + g_offs
    tl.store(KV_cache_ptr + k_scale_addrs, k_byte)

    # ── V side (mirror K but skip rotation) ────────────────────────────
    v_raw = tl.load(Value_ptr + base + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    v_raw = tl.where(d_mask, v_raw, 0.0)

    v_g = tl.reshape(v_raw, [N_GROUPS_C, GROUP_SIZE_C])
    v_absmax = tl.max(tl.abs(v_g), axis=1)
    v_is_zero = v_absmax == 0.0

    v_s_raw = v_absmax * tl.cast(FP4_C, tl.float32)
    v_s_safe = tl.where(v_is_zero, 1.0, v_s_raw)
    v_log2 = tl.log2(v_s_safe)
    v_exp = tl.cast(tl.floor(v_log2 + 0.5), tl.int32)
    v_s_snapped = tl.exp2(tl.cast(v_exp, tl.float32))
    v_s_div = tl.where(v_is_zero, 1.0, v_s_snapped)

    v_norm = v_g / v_s_div[:, None]
    v_sorted = tl.zeros([N_GROUPS_C, GROUP_SIZE_C], dtype=tl.int32)
    for i in tl.static_range(14):
        mid = _MIDPOINTS_C.value[i] if CONST_TABLES else tl.load(Midpoints_ptr + i)
        v_sorted += tl.where(v_norm > mid, 1, 0)
    v_sorted = tl.where(v_is_zero[:, None], 7, v_sorted)

    if CONST_TABLES:
        v_bits = tl.where(v_sorted < 7, 15 - v_sorted, v_sorted - 7)
    else:
        v_bits = tl.load(SortedToBits_ptr + v_sorted)
    v_bits_flat = tl.reshape(v_bits, [HEAD_DIM])
    v_pairs = tl.reshape(v_bits_flat, [HEAD_DIM // 2, 2])
    v_packed = tl.sum((v_pairs & 0xF) << shifts_4[None, :], axis=1).to(tl.uint8)
    v_code_addrs = v_code_base + tl.arange(0, HEAD_DIM // 2)
    tl.store(KV_cache_ptr + v_code_addrs, v_packed)

    v_byte = tl.where(v_is_zero, 0, v_exp + UE8M0_BIAS_C)
    v_byte = tl.where((v_byte < 0) | (v_byte > 255), 0, v_byte).to(tl.uint8)
    v_scale_addrs = v_scale_base + g_offs
    tl.store(KV_cache_ptr + v_scale_addrs, v_byte)


# ═══════════════════════════════════════════════════════════════════════════
# Launcher
# ═══════════════════════════════════════════════════════════════════════════


def ultraquant_store(
    key: torch.Tensor,  # [N, H, D] bf16 or fp16
    value: torch.Tensor,  # [N, H, D] bf16 or fp16
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, slot_size_aligned] uint8
    slot_mapping: torch.Tensor,  # [N] int64
    *,
    PiT: torch.Tensor | None = None,  # unused; kept for call-site compatibility
    constant_c: float | None = None,
) -> None:
    """Launch the ultraquant store kernel. Writes K/V into `kv_cache`
    in-place at the slots specified by `slot_mapping`."""
    if key.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"ultraquant_store: key.dtype must be one of "
            f"{_SUPPORTED_DTYPES}, got {key.dtype}"
        )
    if value.dtype != key.dtype:
        raise ValueError(
            f"ultraquant_store: key/value dtype mismatch ({key.dtype} vs {value.dtype})"
        )
    if slot_mapping.dtype != torch.int64:
        slot_mapping = slot_mapping.to(torch.int64)

    N, H, D = key.shape
    NH = N * H
    block_size = kv_cache.shape[1]
    num_kv_heads = kv_cache.shape[2]
    padded_slot = kv_cache.shape[3]

    expected_slot = slot_size(D)
    if padded_slot < expected_slot:
        raise ValueError(
            f"ultraquant_store: kv_cache slot {padded_slot} < expected "
            f"{expected_slot} for head_dim={D}"
        )
    if num_kv_heads != H:
        raise ValueError(
            f"ultraquant_store: kv_cache num_kv_heads {num_kv_heads} != key heads {H}"
        )

    midpoints = _get_midpoints_tensor(key.device)
    sorted_to_bits = _get_sorted_to_bits_tensor(key.device)

    k_flat = key.reshape(NH, D).contiguous()
    v_flat = value.reshape(NH, D).contiguous()

    gs = get_group_size()
    BLOCK_D = triton.next_power_of_2(D)
    N_GROUPS_C = n_groups(D, gs)
    K_SCALES_OFFSET = k_scales_offset(D, gs)
    V_CODES_OFFSET = v_codes_offset(D, gs)
    V_SCALES_OFFSET = v_scales_offset(D, gs)

    stride_block = kv_cache.stride(0)
    stride_pos = kv_cache.stride(1)
    stride_head = kv_cache.stride(2)

    c = constant_c if constant_c is not None else get_constant_c()

    _ensure_const_tables()
    grid = (NH,)
    _ultraquant_store_kernel[grid](
        k_flat,
        v_flat,
        _kv_cache_flat(kv_cache),
        slot_mapping,
        midpoints,
        sorted_to_bits,
        stride_cache_block=stride_block,
        stride_cache_pos=stride_pos,
        stride_cache_head=stride_head,
        HEAD_DIM=D,
        H=H,
        BLOCK_SIZE=block_size,
        BLOCK_D=BLOCK_D,
        LOG2_D=int(D).bit_length() - 1,
        GROUP_SIZE_C=gs,
        N_GROUPS_C=N_GROUPS_C,
        K_SCALES_OFFSET=K_SCALES_OFFSET,
        V_CODES_OFFSET=V_CODES_OFFSET,
        V_SCALES_OFFSET=V_SCALES_OFFSET,
        FP4_C=c,
        UE8M0_BIAS_C=UE8M0_BIAS,
        CONST_TABLES=1,
        num_warps=4,
        num_stages=1,
    )
