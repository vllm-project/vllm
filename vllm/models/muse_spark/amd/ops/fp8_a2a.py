# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 payloads for the Ulysses all-to-all exchanges (`ulysses_sp`).

A payload row carries `cols` values as FP8 E4M3 with one FP32 scale per `head_dim`
values (one per token and head), packed as `[cols FP8 bytes | cols // head_dim FP32
scales]`, so one all-to-all moves both. `quant_pack` also does the layout change in
front of the exchange (rows of each destination rank contiguous) and
`dequant_unpack` the one after it.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

_HEADS_PER_PROGRAM = 32
_NUM_WARPS = 2


def row_bytes(cols: int, head_dim: int) -> int:
    """Bytes of one packed row of `cols` values."""
    return cols + cols // head_dim * 4


@triton.jit
def _quant_pack_kernel(
    x_ptr,
    x_stride,
    q_ptr,
    s_ptr,
    num_rows,
    rows_per_group,
    cols,
    row_bytes,
    total_heads,
    HEAD: tl.constexpr,
    HPP: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    row = tl.program_id(0)
    heads = tl.program_id(1) * HPP + tl.arange(0, HPP)
    hmask = heads < total_heads
    dims = tl.arange(0, HEAD)
    mask = hmask[:, None] & (row < num_rows)
    x = tl.load(
        x_ptr + row.to(tl.int64) * x_stride + heads[:, None] * HEAD + dims[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-12) / FP8_MAX
    q = tl.clamp(x / scale[:, None], -FP8_MAX, FP8_MAX).to(q_ptr.dtype.element_ty)
    heads_per_group = cols // HEAD
    group = heads // heads_per_group
    local = heads % heads_per_group
    base = (group.to(tl.int64) * rows_per_group + row) * row_bytes
    tl.store(
        q_ptr + base[:, None] + local[:, None] * HEAD + dims[None, :],
        q,
        mask=hmask[:, None],
    )
    tl.store(s_ptr + (base + cols) // 4 + local, scale, mask=hmask)


@triton.jit
def _dequant_unpack_kernel(
    q_ptr,
    s_ptr,
    y_ptr,
    y_stride,
    rows_per_group,
    cols,
    row_bytes,
    total_heads,
    HEAD: tl.constexpr,
    HPP: tl.constexpr,
):
    row = tl.program_id(0)
    heads = tl.program_id(1) * HPP + tl.arange(0, HPP)
    hmask = heads < total_heads
    dims = tl.arange(0, HEAD)
    heads_per_group = cols // HEAD
    group = heads // heads_per_group
    local = heads % heads_per_group
    base = (group.to(tl.int64) * rows_per_group + row) * row_bytes
    q = tl.load(
        q_ptr + base[:, None] + local[:, None] * HEAD + dims[None, :],
        mask=hmask[:, None],
        other=0.0,
    ).to(tl.float32)
    s = tl.load(s_ptr + (base + cols) // 4 + local, mask=hmask, other=0.0)
    y = (q * s[:, None]).to(y_ptr.dtype.element_ty)
    tl.store(
        y_ptr + row.to(tl.int64) * y_stride + heads[:, None] * HEAD + dims[None, :],
        y,
        mask=hmask[:, None],
    )


def quant_pack(
    x: torch.Tensor, groups: int, rows_per_group: int, head_dim: int
) -> torch.Tensor:
    """Quantize `x` `[num_rows, groups * cols]` per (row, head) into a packed
    `[groups, rows_per_group, row_bytes]` uint8 buffer: column group `g` of row
    `r` lands at `[g, r]`. Rows past `num_rows` are zero."""
    num_rows, width = x.shape
    assert x.stride(-1) == 1 and width % groups == 0
    cols = width // groups
    assert cols % head_dim == 0 and head_dim % 4 == 0 and num_rows <= rows_per_group
    nbytes = row_bytes(cols, head_dim)
    out = torch.empty(
        (groups, rows_per_group, nbytes), dtype=torch.uint8, device=x.device
    )
    total_heads = width // head_dim
    grid = (rows_per_group, triton.cdiv(total_heads, _HEADS_PER_PROGRAM))
    fp8 = current_platform.fp8_dtype()
    _quant_pack_kernel[grid](
        x,
        x.stride(0),
        out.view(fp8),
        out.view(torch.float32),
        num_rows,
        rows_per_group,
        cols,
        nbytes,
        total_heads,
        HEAD=head_dim,
        HPP=_HEADS_PER_PROGRAM,
        FP8_MAX=torch.finfo(fp8).max,
        num_warps=_NUM_WARPS,
    )
    return out


def dequant_unpack(
    packed: torch.Tensor,
    num_rows: int,
    cols: int,
    head_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Inverse of `quant_pack`: `[groups, rows_per_group, row_bytes]` ->
    `[num_rows, groups * cols]` of `dtype` (group `g` -> columns `g * cols:`)."""
    groups, rows_per_group, nbytes = packed.shape
    assert nbytes == row_bytes(cols, head_dim) and num_rows <= rows_per_group
    total_heads = groups * cols // head_dim
    y = torch.empty((num_rows, groups * cols), dtype=dtype, device=packed.device)
    grid = (num_rows, triton.cdiv(total_heads, _HEADS_PER_PROGRAM))
    _dequant_unpack_kernel[grid](
        packed.view(current_platform.fp8_dtype()),
        packed.view(torch.float32),
        y,
        y.stride(0),
        rows_per_group,
        cols,
        nbytes,
        total_heads,
        HEAD=head_dim,
        HPP=_HEADS_PER_PROGRAM,
        num_warps=_NUM_WARPS,
    )
    return y
