# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact sparse-prefill KV gathering and reusable host planning metadata."""

from typing import NamedTuple

import torch

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["REGION_COUNT"])
def _gather_prefill_kv(
    Cache,
    Regions,
    Table,
    Owners,
    Positions,
    CuKV,
    Key,
    Value,
    KScale,
    VScale,
    FP8_CACHE: tl.constexpr,
    CACHE_PAGE: tl.constexpr,
    CACHE_TOKEN: tl.constexpr,
    CACHE_DIM: tl.constexpr,
    REGION_ROW: tl.constexpr,
    REGION_COL: tl.constexpr,
    REGION_COUNT,
    TABLE_ROW: tl.constexpr,
    TABLE_COL: tl.constexpr,
    OWNER_STRIDE: tl.constexpr,
    POSITION_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DIM: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(1)
    offset = tl.program_id(0) * TILE + tl.arange(0, TILE)
    start = tl.load(CuKV + row).to(tl.int64)
    end = tl.load(CuKV + row + 1).to(tl.int64)
    base = tl.load(CuKV).to(tl.int64)
    valid = offset < end - start
    position = tl.load(Positions + row * POSITION_STRIDE)
    # Owned selector contract: sorted, left-padded, current region included.
    populated = tl.minimum((position >> 3) + 1, REGION_COUNT)
    column = REGION_COUNT - populated + offset // 8
    region = tl.load(Regions + row * REGION_ROW + column * REGION_COL, valid, other=0)
    token = region * 8 + offset % 8
    owner = tl.load(Owners + row * OWNER_STRIDE)
    physical = tl.load(
        Table + owner * TABLE_ROW + token // BLOCK_SIZE * TABLE_COL,
        valid,
        other=0,
    ).to(tl.int64)
    dim = tl.arange(0, BLOCK_D)
    address = (
        physical[:, None] * CACHE_PAGE
        + (token % BLOCK_SIZE)[:, None] * CACHE_TOKEN
        + dim[None, :] * CACHE_DIM
    )
    mask = valid[:, None] & (dim[None, :] < DIM)
    key = tl.load(Cache + address, mask, other=0.0)
    value = tl.load(Cache + address + DIM * CACHE_DIM, mask, other=0.0)
    if FP8_CACHE:
        key = key.to(tl.float32) * tl.load(KScale)
        value = value.to(tl.float32) * tl.load(VScale)
    output = (start - base + offset)[:, None] * DIM + dim[None, :]
    tl.store(Key + output, key, mask)
    tl.store(Value + output, value, mask)


def make_indptr(positions: list[int], region_count: int) -> torch.Tensor:
    """Compute exact compact lengths from the selector's cardinality contract."""
    if region_count < 1 or any(position < 0 for position in positions):
        raise ValueError("Prefill needs nonnegative positions and a current region")
    result = [0]
    for position in positions:
        count = min(position // 8, region_count - 1) * 8 + position % 8 + 1
        result.append(result[-1] + count)
    return torch.tensor(result, dtype=torch.int32, device="cpu")


def gather_prefill_kv(
    cache,
    regions,
    block_table,
    owners,
    positions,
    cu_kv,
    total_tokens,
    max_length,
    *,
    output_dtype=None,
    k_scale=None,
    v_scale=None,
):
    """Gather selected tokens in exactly the boolean-indexed reference order."""
    fp8_cache = cache.dtype in (torch.uint8, torch.float8_e4m3fn)
    if fp8_cache:
        if output_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("FP8 gather requires a model output dtype")
        if k_scale is None or v_scale is None:
            raise ValueError("FP8 gather requires K/V scales")
        cache = cache.view(torch.float8_e4m3fn)
    elif cache.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Unsupported sparse KV cache dtype")
    rows, count = regions.shape
    dim = cache.shape[-1] // 2
    key = torch.empty(
        (total_tokens, 1, dim), device=cache.device, dtype=output_dtype or cache.dtype
    )
    value = torch.empty_like(key)
    if rows and total_tokens:
        # Integer views make bit preservation explicit, including NaN payloads.
        _gather_prefill_kv[(triton.cdiv(max_length, 16), rows)](
            cache if fp8_cache else cache.view(torch.int16),
            regions,
            block_table,
            owners,
            positions,
            cu_kv,
            key if fp8_cache else key.view(torch.int16),
            value if fp8_cache else value.view(torch.int16),
            k_scale,
            v_scale,
            fp8_cache,
            cache.stride(0),
            cache.stride(2),
            cache.stride(3),
            *regions.stride(),
            count,
            *block_table.stride(),
            owners.stride(0),
            positions.stride(0),
            cache.shape[2],
            dim,
            16,
            triton.next_power_of_2(dim),
        )
    return key, value


class PrefillChunk(NamedTuple):
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    cu_k_device: torch.Tensor
    total_tokens: int
    max_length: int


def make_prefill_chunks(
    positions: list[int], topk: int, device: torch.device
) -> list[PrefillChunk]:
    """Share CPU plans and one GPU prefix copy across all sparse layers."""
    cu_k = make_indptr(positions, topk + 1)
    device_cu_k = cu_k.pin_memory().to(device=device, non_blocking=True)
    chunks = []
    for start in range(0, len(positions), 32):
        end = min(start + 32, len(positions))
        local_cu_k = cu_k[start : end + 1] - cu_k[start]
        chunks.append(
            PrefillChunk(
                torch.arange(end - start + 1, dtype=torch.int32, device="cpu"),
                local_cu_k,
                device_cu_k[start : end + 1],
                int(local_cu_k[-1]),
                int((local_cu_k[1:] - local_cu_k[:-1]).max()),
            )
        )
    return chunks
