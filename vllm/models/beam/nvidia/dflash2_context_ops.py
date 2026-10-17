# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["ROWS", "NUM_SLOTS"])
def _context_kv_kernel(
    KV,
    WEIGHT,
    POSITIONS,
    ROPE,
    CACHE_PTRS,
    CACHE,
    SCALE_PTRS,
    SLOTS,
    ROWS,
    NUM_SLOTS,
    LAYERS: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_STRIDE: tl.constexpr,
    HEAD_STRIDE: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    PAGE: tl.constexpr,
    EPS: tl.constexpr,
    VALUE_SCALE: tl.constexpr,
    NEOX: tl.constexpr,
    FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * 4 + tl.arange(0, 4)
    layer_head = tl.program_id(1)
    layer = layer_head // HEADS
    head = layer_head % HEADS
    col = tl.arange(0, BLOCK)
    slot = tl.load(SLOTS + row, (row < ROWS) & (row < NUM_SLOTS), -1).to(tl.int64)
    mask = (row[:, None] < ROWS) & (slot[:, None] >= 0) & (col[None, :] < DIM)
    offsets = (
        row[:, None] * (LAYERS * 2 * HEADS * DIM)
        + (layer * 2 * HEADS + head) * DIM
        + col[None, :]
    )
    key = tl.load(KV + offsets, mask, 0).to(tl.float32)
    if BLOCK == 128:
        squares = (key * key).reshape((4, 4, 32))
        total = tl.sum(tl.sum(squares, 2), 1)
    else:
        total = tl.sum(key * key, 1)
    weight = tl.load(WEIGHT + layer * DIM + col, col < DIM, 0).to(tl.float32)
    key = key * tl.rsqrt(total / DIM + EPS)[:, None] * weight[None, :]
    key = key.to(KV.dtype.element_ty).to(tl.float32)
    half: tl.constexpr = DIM // 2
    if NEOX:
        partner = (col + half) % DIM
        second = col >= half
        rope_col = col % half
    else:
        partner = col ^ 1
        second = col % 2 != 0
        rope_col = col // 2
    other = tl.gather(key, tl.broadcast_to(partner[None, :], (4, BLOCK)), 1)
    position = tl.load(POSITIONS + row, (row < ROWS) & (slot >= 0), 0).to(tl.int64)
    cosine = tl.load(ROPE + position[:, None] * DIM + rope_col[None, :], mask, 0)
    sine = tl.load(ROPE + position[:, None] * DIM + half + rope_col[None, :], mask, 0)
    key = tl.where(
        second[None, :],
        tl.fma(other, sine, key * cosine),
        tl.fma(key, cosine, -other * sine),
    )
    cache = tl.load(CACHE_PTRS + layer).to(tl.pointer_type(CACHE.dtype.element_ty))
    address = (
        slot[:, None] // PAGE * BLOCK_STRIDE
        + slot[:, None] % PAGE * TOKEN_STRIDE
        + head * HEAD_STRIDE
        + col[None, :]
    )
    value = tl.load(KV + offsets + HEADS * DIM, mask, 0).to(tl.float32)
    value = value * VALUE_SCALE
    if FP8:
        # Preserve BF16 rounding after RoPE and value scaling.
        key = key.to(KV.dtype.element_ty).to(tl.float32)
        value = value.to(KV.dtype.element_ty).to(tl.float32)
        k_scale = tl.load(SCALE_PTRS + layer * 2).to(tl.pointer_type(tl.float32))
        v_scale = tl.load(SCALE_PTRS + layer * 2 + 1).to(tl.pointer_type(tl.float32))
        key = tl.minimum(tl.maximum(key / tl.load(k_scale), -FP8_MAX), FP8_MAX)
        value = tl.minimum(tl.maximum(value / tl.load(v_scale), -FP8_MAX), FP8_MAX)
    tl.store(cache + address, key, mask)
    tl.store(cache + address + DIM, value, mask)


def store_context_kv(
    kv: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    rope: torch.Tensor,
    cache_ptrs: torch.Tensor,
    slots: torch.Tensor,
    cache: torch.Tensor,
    eps: float,
    value_scale: float,
    is_neox: bool,
    scale_ptrs: torch.Tensor | None = None,
) -> None:
    """Normalize, rotate, and scatter all draft layers' context K/V."""
    rows, layers, _, heads, dim = kv.shape
    fp8 = cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    assert cache.dtype == kv.dtype or fp8
    assert not fp8 or scale_ptrs is not None
    if rows:
        _context_kv_kernel[(triton.cdiv(rows, 4), layers * heads)](
            kv,
            weight,
            positions,
            rope,
            cache_ptrs,
            cache,
            scale_ptrs,
            slots,
            rows,
            slots.numel(),
            layers,
            heads,
            dim,
            cache.stride(0),
            cache.stride(1),
            cache.stride(2),
            cache.shape[2],
            eps,
            value_scale,
            is_neox,
            fp8,
            torch.finfo(cache.dtype).max if fp8 else 0.0,
            triton.next_power_of_2(dim),
            num_warps=4,
            enable_fp_fusion=False,
        )
