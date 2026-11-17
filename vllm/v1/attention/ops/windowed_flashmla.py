# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Windowed native FlashMLA using a temporary pack of standard FP8 KV pages."""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.flashmla import flash_mla_with_kvcache, get_mla_metadata
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)


def windowed_flash_mla_workspace(
    requests: int,
    rows: int,
    window: int,
    max_query_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    capacity = triton.cdiv(window + max_query_len - 1, 64) * 64
    topk = triton.cdiv(window, 64) * 64
    shapes = (
        ((requests * capacity // 64, 64, 1, 656), torch.uint8),
        ((rows, 1, topk), torch.int32),
        ((rows,), torch.int32),
    )
    if is_workspace_manager_initialized():
        buffers = current_workspace_manager().get_simultaneous(*shapes)
    else:
        buffers = [
            torch.empty(shape, device=device, dtype=dtype) for shape, dtype in shapes
        ]
    return buffers[0], buffers[1], buffers[2]


@triton.jit
def _pack_visible_window(
    K,
    TABLE,
    LENGTHS,
    STARTS,
    SCALE,
    PACKED,
    PAGE_STRIDE: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    CAPACITY: tl.constexpr,
    WINDOW: tl.constexpr,
    TILE: tl.constexpr,
):
    request = tl.program_id(0)
    offset = tl.program_id(1) * TILE + tl.arange(0, TILE)
    length = tl.load(LENGTHS + request)
    query_len = tl.load(STARTS + request + 1) - tl.load(STARTS + request)
    begin = tl.maximum(length - WINDOW - query_len + 1, 0)
    position = begin + offset
    valid = (position < length) & (offset < CAPACITY) & (query_len > 0)
    page = tl.load(
        TABLE + request * TABLE_STRIDE + position // PAGE_SIZE, valid, other=0
    ).to(tl.int64)
    source = page * PAGE_STRIDE + position % PAGE_SIZE * TOKEN_STRIDE
    destination = (request * CAPACITY + offset).to(tl.int64) * 656
    latent_dim = tl.arange(0, 512)
    raw = tl.load(K + source[:, None] + latent_dim[None, :], valid[:, None], other=0.0)
    tl.store(
        PACKED + destination[:, None] + latent_dim[None, :],
        raw.to(tl.uint8, bitcast=True),
        offset[:, None] < CAPACITY,
    )
    scale = tl.load(SCALE)
    byte = tl.arange(0, 16)
    bits = scale.to(tl.uint32, bitcast=True)
    tl.store(
        PACKED + destination[:, None] + 512 + byte[None, :],
        ((bits >> ((byte[None, :] % 4) * 8)) & 255).to(tl.uint8),
        offset[:, None] < CAPACITY,
    )
    rotary_dim = tl.arange(0, 64)
    rotary = (
        tl.load(
            K + source[:, None] + 512 + rotary_dim[None, :], valid[:, None], other=0.0
        ).to(tl.float32)
        * scale
    )
    rotary_bits = rotary.to(tl.bfloat16).to(tl.uint16, bitcast=True)
    tl.store(
        PACKED + destination[:, None] + 528 + rotary_dim[None, :] * 2,
        (rotary_bits & 255).to(tl.uint8),
        offset[:, None] < CAPACITY,
    )
    tl.store(
        PACKED + destination[:, None] + 529 + rotary_dim[None, :] * 2,
        (rotary_bits >> 8).to(tl.uint8),
        offset[:, None] < CAPACITY,
    )


@triton.jit
def _window_indices(
    LENGTHS,
    STARTS,
    INDICES,
    ACTIVE,
    REQUESTS: tl.constexpr,
    WINDOW: tl.constexpr,
    CAPACITY: tl.constexpr,
    TOPK: tl.constexpr,
    CAUSAL: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    offset = tl.program_id(1) * TILE + tl.arange(0, TILE)
    low, high = 0, REQUESTS
    while low < high:
        mid = (low + high) // 2
        beyond = row >= tl.load(STARTS + mid + 1)
        low = tl.where(beyond, mid + 1, low)
        high = tl.where(beyond, high, mid)
    valid = low < REQUESTS
    request = tl.minimum(low, REQUESTS - 1)
    start = tl.load(STARTS + request)
    stop = tl.load(STARTS + request + 1)
    length = tl.load(LENGTHS + request)
    valid &= (row >= start) & (row < stop) & (length > 0)
    query_len = stop - start
    union_begin = tl.maximum(length - WINDOW - query_len + 1, 0)
    end = length
    if CAUSAL:
        end = tl.maximum(length - query_len + row - start + 1, 0)
    count = tl.where(valid, tl.minimum(end, WINDOW), 0)
    position = tl.maximum(end - WINDOW, 0) + offset
    physical = request * CAPACITY + position - union_begin
    tl.store(
        INDICES + row * TOPK + offset,
        tl.where(offset < count, physical, -1),
        offset < TOPK,
    )
    if tl.program_id(1) == 0:
        tl.store(ACTIVE + row, count)


def windowed_flash_mla(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    window: int,
    max_query_len: int,
    kv_scale: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend to an exact window without changing the persistent FP8 layout.

    The caller must select the native 512-latent/64-RoPE kernel and validate a
    positive power-of-two KV scale. Other geometries/scales use Triton instead.
    Temporary native rows include the union of the queries' visible windows.
    """
    q = q.contiguous()
    capacity = triton.cdiv(window + max_query_len - 1, 64) * 64
    requests = seq_lens.shape[0]
    topk = triton.cdiv(window, 64) * 64
    packed, indices, active = windowed_flash_mla_workspace(
        requests, q.shape[0], window, max_query_len, q.device
    )
    _pack_visible_window[(requests, triton.cdiv(capacity, 16))](
        kv_cache,
        block_table,
        seq_lens,
        query_start_loc,
        kv_scale,
        packed,
        kv_cache.stride(0),
        kv_cache.stride(1),
        block_table.stride(0),
        kv_cache.shape[1],
        capacity,
        window,
        16,
    )
    _window_indices[(q.shape[0], triton.cdiv(topk, 256))](
        seq_lens,
        query_start_loc,
        indices,
        active,
        requests,
        window,
        capacity,
        topk,
        causal,
        256,
    )
    metadata, _ = get_mla_metadata()
    out, lse = flash_mla_with_kvcache(
        q.unsqueeze(1),
        packed,
        None,
        None,
        512,
        metadata,
        softmax_scale=softmax_scale,
        causal=False,
        is_fp8_kvcache=True,
        indices=indices,
        topk_length=active,
    )
    out = torch.where((active > 0)[:, None, None], out.squeeze(1), 0)
    return out, lse.squeeze(-1)
