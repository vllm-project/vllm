# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 pack / unpack for the all-to-all of the sequence-parallel o_proj: a row is
its e4m3 codes, then its e8m0 scales (``mxfp8_e4m3_quantize``'s quant)."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _quant_pack_kernel(
    x_ptr,
    out_ptr,
    M,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    FP8_MAX: tl.constexpr,
    TINY: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, D)
    mask = rows < M
    x = tl.load(
        x_ptr + rows[:, None] * D + cols[None, :], mask=mask[:, None], other=0.0
    )
    x = tl.reshape(x.to(tl.float32), (BLOCK_M, D // 32, 32))
    amax = tl.maximum(tl.max(tl.abs(x), axis=2), TINY)
    sb = tl.minimum(tl.maximum(tl.ceil(tl.log2(amax / FP8_MAX)) + 127.0, 0.0), 254.0)
    q = (x * tl.exp2(127.0 - sb)[:, :, None]).to(tl.float8e4nv)
    row_bytes = D + D // 32
    q = tl.reshape(q.to(tl.uint8, bitcast=True), (BLOCK_M, D))
    tl.store(out_ptr + rows[:, None] * row_bytes + cols[None, :], q, mask=mask[:, None])
    groups = tl.arange(0, D // 32)
    tl.store(
        out_ptr + rows[:, None] * row_bytes + D + groups[None, :],
        sb.to(tl.uint8),
        mask=mask[:, None],
    )


@triton.jit
def _unpack_kernel(
    in_ptr,
    codes_ptr,
    scales_ptr,
    ROWS,
    WORLD: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    w = tl.program_id(1)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = rows < ROWS
    row_bytes = D + D // 32
    src = in_ptr + (w * ROWS + rows[:, None]) * row_bytes
    cols = tl.arange(0, D)
    codes = tl.load(src + cols[None, :], mask=mask[:, None])
    tl.store(
        codes_ptr + rows[:, None] * (WORLD * D) + w * D + cols[None, :],
        codes,
        mask=mask[:, None],
    )
    groups = tl.arange(0, D // 32)
    scales = tl.load(src + D + groups[None, :], mask=mask[:, None])
    tl.store(
        scales_ptr
        + rows[:, None] * (WORLD * D // 32)
        + w * (D // 32)
        + groups[None, :],
        scales,
        mask=mask[:, None],
    )


def mxfp8_quant_pack(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """``[M, D]`` bf16 -> ``[M, D + D/32]`` uint8."""
    m, d = x.shape
    assert x.is_contiguous() and d % 32 == 0 and d & (d - 1) == 0, x.shape
    if out is None:
        out = torch.empty((m, d + d // 32), dtype=torch.uint8, device=x.device)
    assert out.shape == (m, d + d // 32) and out.dtype == torch.uint8
    assert out.is_contiguous()
    block_m = 4
    _quant_pack_kernel[(triton.cdiv(m, block_m),)](
        x,
        out,
        m,
        D=d,
        BLOCK_M=block_m,
        FP8_MAX=float(torch.finfo(torch.float8_e4m3fn).max),
        TINY=float(torch.finfo(torch.float32).tiny),
        num_warps=8,
    )
    return out


def mxfp8_unpack(
    packed: torch.Tensor, world: int, rows: int, d: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[world * rows, d + d/32]`` (block w from rank w) -> codes
    ``[rows, world * d]`` and scales ``[rows, world * d / 32]``."""
    assert packed.shape == (world * rows, d + d // 32) and packed.is_contiguous()
    codes = torch.empty((rows, world * d), dtype=torch.uint8, device=packed.device)
    scales = torch.empty(
        (rows, world * d // 32), dtype=torch.uint8, device=packed.device
    )
    block_m = 4
    _unpack_kernel[(triton.cdiv(rows, block_m), world)](
        packed, codes, scales, rows, WORLD=world, D=d, BLOCK_M=block_m, num_warps=8
    )
    return codes.view(torch.float8_e4m3fn), scales
