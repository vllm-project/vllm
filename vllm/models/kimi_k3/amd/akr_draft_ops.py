# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row-major -> MFMA tile layout of the drafter MLP's activations (one launch)."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["rows"])
def _pack_tiles_kernel(x_ptr, out_ptr, rows, K: tl.constexpr, KSTEPS: tl.constexpr):
    # out [tiles, KSTEPS, 64 lanes, 8]: lane l holds row 16 * tile + l % 16 and
    # columns 32 * kstep + 8 * (l // 16) .. + 8
    tile = tl.program_id(0)
    kstep = tl.program_id(1)
    offs = tl.arange(0, 512)
    lane = offs // 8
    e = offs % 8
    row = tile * 16 + lane % 16
    col = kstep * 32 + (lane // 16) * 8 + e
    v = tl.load(x_ptr + row.to(tl.int64) * K + col, mask=row < rows, other=0.0)
    tl.store(out_ptr + (tile * KSTEPS + kstep).to(tl.int64) * 512 + offs, v)


def pack_tiles(x: torch.Tensor, out: torch.Tensor, tiles: int) -> torch.Tensor:
    """X [rows, K] bf16 contiguous -> out [tiles, K / 32, 64, 8] (rows past `rows`:
    0)."""
    rows, k = x.shape
    ksteps = k // 32
    assert out.numel() >= tiles * ksteps * 512 and tiles * 16 >= rows
    # one wave per 1 KB tile (16 B a lane): 3.3 vs 4.9 us for 8 waves at 240 rows
    _pack_tiles_kernel[(tiles, ksteps)](x, out, rows, K=k, KSTEPS=ksteps, num_warps=1)
    return out
