# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_slots"])
def _rope_and_cache_diffkv(
    query,
    key,
    value,
    positions,
    cos_sin_cache,
    kv_cache,
    slot_mapping,
    query_out,
    num_slots,
    Q_STRIDES: tl.constexpr,
    K_STRIDES: tl.constexpr,
    V_STRIDES: tl.constexpr,
    CACHE_STRIDES: tl.constexpr,
    Q_OUT_STRIDES: tl.constexpr,
    POS_STRIDE: tl.constexpr,
    COS_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_SIZE_K: tl.constexpr,
    HEAD_SIZE_V: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    VALUE_SCALE: tl.constexpr,
    IS_NEOX: tl.constexpr,
    OUTPLACE_Q: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    offsets = tl.arange(0, TILE_SIZE)
    half_rotary: tl.constexpr = ROTARY_DIM // 2
    if IS_NEOX:
        x_offsets = offsets
        y_offsets = offsets + half_rotary
    else:
        x_offsets = 2 * offsets
        y_offsets = x_offsets + 1

    position = tl.load(positions + token * POS_STRIDE).to(tl.int64)
    cos = tl.load(
        cos_sin_cache + position * COS_STRIDE + offsets,
        offsets < half_rotary,
        other=0,
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache + position * COS_STRIDE + half_rotary + offsets,
        offsets < half_rotary,
        other=0,
    ).to(tl.float32)

    if head < NUM_Q_HEADS:
        source = query + token * Q_STRIDES[0] + head * Q_STRIDES[1]
    else:
        source = key + token * K_STRIDES[0] + (head - NUM_Q_HEADS) * K_STRIDES[1]
    x = tl.load(source + x_offsets, offsets < half_rotary, other=0).to(tl.float32)
    y = tl.load(source + y_offsets, offsets < half_rotary, other=0).to(tl.float32)
    rotated_x = (x * cos - y * sin).to(source.dtype.element_ty)
    rotated_y = (y * cos + x * sin).to(source.dtype.element_ty)
    if head >= NUM_Q_HEADS:
        kv_head = head - NUM_Q_HEADS
        tail_mask = (offsets >= ROTARY_DIM) & (offsets < HEAD_SIZE_K)
        tail = tl.load(source + offsets, tail_mask, other=0)
        value_ptr = value + token * V_STRIDES[0] + kv_head * V_STRIDES[1] + offsets
        v = tl.load(value_ptr, offsets < HEAD_SIZE_V, other=0)
        if VALUE_SCALE != 1.0:
            v = (v.to(tl.float32) * VALUE_SCALE).to(value.dtype.element_ty)
            if not OUTPLACE_Q:
                tl.store(value_ptr, v, offsets < HEAD_SIZE_V)

        slot = tl.load(
            slot_mapping + token * SLOT_STRIDE, token < num_slots, other=-1
        ).to(tl.int64)
        cache_ptr = (
            kv_cache
            + (slot // BLOCK_SIZE) * CACHE_STRIDES[0]
            + (slot % BLOCK_SIZE) * CACHE_STRIDES[1]
            + kv_head * CACHE_STRIDES[2]
        )
        valid_slot = slot >= 0
        tl.store(cache_ptr + x_offsets, rotated_x, valid_slot & (offsets < half_rotary))
        tl.store(cache_ptr + y_offsets, rotated_y, valid_slot & (offsets < half_rotary))
        tl.store(cache_ptr + offsets, tail, valid_slot & tail_mask)
        tl.store(
            cache_ptr + HEAD_SIZE_K + offsets, v, valid_slot & (offsets < HEAD_SIZE_V)
        )
    if OUTPLACE_Q:
        if head < NUM_Q_HEADS:
            target = query_out + token * Q_OUT_STRIDES[0] + head * Q_OUT_STRIDES[1]
            tail_mask = (offsets >= ROTARY_DIM) & (offsets < HEAD_SIZE_K)
            tail = tl.load(source + offsets, tail_mask, other=0)
            tl.store(target + x_offsets, rotated_x, offsets < half_rotary)
            tl.store(target + y_offsets, rotated_y, offsets < half_rotary)
            tl.store(target + offsets, tail, tail_mask)
    else:
        tl.store(source + x_offsets, rotated_x, offsets < half_rotary)
        tl.store(source + y_offsets, rotated_y, offsets < half_rotary)


def triton_rope_and_cache_diffkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    value_scale: float = 1.0,
    is_neox: bool = True,
    query_out: torch.Tensor | None = None,
) -> None:
    """Apply partial RoPE and V scaling, and store packed DiffKV.

    Q/K/V have shape [tokens, heads, head_dim]. The strided cache has shape
    [blocks, block_size, kv_heads, key_dim + value_dim]. Negative or missing
    slots skip cache writes while still transforming the corresponding Q/K/V.
    With query_out, write rotated Q there and leave input Q/K/V unchanged.
    """
    assert query.ndim == key.ndim == value.ndim == 3
    assert positions.ndim == slot_mapping.ndim == 1
    assert kv_cache.ndim == 4 and cos_sin_cache.ndim == 2
    assert query.dtype in (torch.float16, torch.bfloat16)
    assert key.dtype == value.dtype == query.dtype
    assert kv_cache.dtype in (torch.float16, torch.bfloat16)
    assert all(t.stride(-1) == 1 for t in (query, key, value, kv_cache, cos_sin_cache))
    assert query.shape[0] == key.shape[0] == value.shape[0] == positions.numel()
    assert slot_mapping.numel() <= query.shape[0]
    assert query.shape[2] == key.shape[2]
    assert key.shape[1] == value.shape[1] == kv_cache.shape[2]
    assert kv_cache.shape[3] == key.shape[2] + value.shape[2]
    assert cos_sin_cache.shape[1] <= key.shape[2] and cos_sin_cache.shape[1] % 2 == 0
    if query_out is not None:
        assert query_out.shape == query.shape and query_out.dtype == query.dtype
        assert query_out.stride(-1) == 1
    if query.shape[0] == 0:
        return
    _rope_and_cache_diffkv[(query.shape[0], query.shape[1] + key.shape[1])](
        query,
        key,
        value,
        positions,
        cos_sin_cache,
        kv_cache,
        slot_mapping,
        query if query_out is None else query_out,
        slot_mapping.numel(),
        query.stride(),
        key.stride(),
        value.stride(),
        kv_cache.stride(),
        query.stride() if query_out is None else query_out.stride(),
        positions.stride(0),
        cos_sin_cache.stride(0),
        slot_mapping.stride(0),
        query.shape[1],
        key.shape[2],
        value.shape[2],
        cos_sin_cache.shape[1],
        kv_cache.shape[1],
        value_scale,
        is_neox,
        query_out is not None,
        triton.next_power_of_2(max(key.shape[2], value.shape[2])),
        num_warps=4,
    )
