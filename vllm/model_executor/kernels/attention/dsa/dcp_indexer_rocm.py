# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm DCP candidates ordered by descending FP32 score, then ascending token id."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import (
    dcp_local_to_global_index,
)

FP8_DTYPE = current_platform.fp8_dtype()


@triton.jit
def _pack_candidates_kernel(
    logits,
    indices,
    row_starts,
    output,
    rank,
    rows: tl.constexpr,
    k: tl.constexpr,
    width: tl.constexpr,
    logits_stride0: tl.constexpr,
    logits_stride1: tl.constexpr,
    indices_stride0: tl.constexpr,
    indices_stride1: tl.constexpr,
    starts_stride: tl.constexpr,
    world_size: tl.constexpr,
    interleave: tl.constexpr,
    has_offsets: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offsets // k, offsets % k
    mask = row < rows
    local = tl.load(
        indices + row * indices_stride0 + col * indices_stride1,
        mask=mask,
        other=-1,
    ).to(tl.int64)
    column = local
    if has_offsets:
        column += tl.load(row_starts + row * starts_stride, mask=mask, other=0)
    valid = mask & (local >= 0) & (column >= 0) & (column < width)
    score = tl.load(
        logits + row * logits_stride0 + column * logits_stride1,
        mask=valid,
        other=float("nan"),
    )
    valid &= tl.abs(score) < float("inf")
    bits = score.to(tl.int32, bitcast=True).to(tl.int64)
    bits = tl.where((bits & 0x7FFFFFFF) == 0, 0, bits)
    ordered = tl.where(bits < 0, ~bits, bits ^ 0x80000000) - 0x80000000
    global_ids = dcp_local_to_global_index(local, rank, world_size, interleave)
    packed = (ordered << 32) | (0xFFFFFFFF - global_ids)
    tl.store(output + offsets, tl.where(valid, packed, -9223372036854775808), mask)


def pack_dcp_candidates(
    logits: torch.Tensor,
    indices: torch.Tensor,
    rank: int,
    world_size: int,
    interleave: int,
    row_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack each local candidate into one sortable int64 collective element."""
    assert logits.dtype == torch.float32 and indices.dtype == torch.int32
    assert 0 <= rank < world_size and interleave > 0
    invalid = torch.iinfo(torch.int64).min
    if logits.shape[1] == 0:
        return torch.full_like(indices, invalid, dtype=torch.int64)
    output = torch.empty(indices.shape, dtype=torch.int64, device=indices.device)
    if output.numel() == 0:
        return output
    _pack_candidates_kernel[(triton.cdiv(output.numel(), 256),)](
        logits,
        indices,
        row_starts if row_starts is not None else logits,
        output,
        rank,
        *indices.shape,
        logits.shape[1],
        *logits.stride(),
        *indices.stride(),
        row_starts.stride(0) if row_starts is not None else 0,
        world_size,
        interleave,
        row_starts is not None,
        BLOCK=256,
    )
    return output


@triton.jit
def _decode_selected_candidates(
    Keys,
    Output,
    Width: tl.constexpr,
    Count: tl.constexpr,
    KS: tl.constexpr,
    OS: tl.constexpr,
    OCOL: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    keys = tl.load(Keys + row * KS + col, col < Count, other=-9223372036854775808)
    ids = 0xFFFFFFFF - (keys & 0xFFFFFFFF)
    ids = tl.where(keys != -9223372036854775808, ids, -1)
    tl.store(Output + row * OS + col * OCOL, ids, col < Width)


def select_dcp_candidates(gathered: torch.Tensor, out: torch.Tensor) -> None:
    """Select global candidates without floating-point conversion of token ids.

    The selected set follows the packed (score, token id) order, but the ids are
    written in no particular order within a row.
    """
    if not out.numel():
        return
    count = min(out.shape[1], gathered.shape[1])
    if count == 0:
        out.fill_(-1)
        return
    keys = gathered.topk(count, dim=-1, sorted=False).values
    _decode_selected_candidates[(out.shape[0], triton.cdiv(out.shape[1], 256))](
        keys,
        out,
        out.shape[1],
        count,
        keys.stride(0),
        out.stride(0),
        out.stride(1),
        B=256,
    )
