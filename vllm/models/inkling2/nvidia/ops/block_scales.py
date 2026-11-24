# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Operand layouts of the TRT-LLM-gen expert GEMMs.

Block scales of an `[R, C]` operand are stored in tiles of 4 columns and
128 rows (R128c4, for GEMM token tiles of 128 rows and more) or 8 rows (R8c4):
scale `(r, c)` lives at `[r // 128, c // 4, r % 32, r % 128 // 32, c % 4]`
or `[r // 8, c // 4, r % 8, c % 4]`. Expert weight rows are shuffled in
blocks of 32 for the GEMM's transposed epilogue.
"""

import torch

from vllm.triton_utils import tl, triton


def shuffle_rows(x: torch.Tensor) -> torch.Tensor:
    """`[..., R, C]` -> the same shape with rows `8 * i + j` of each block
    of 32 moved to `4 * j + i` (FlashInfer's `shuffle_matrix_a`)."""
    *lead, rows, cols = x.shape
    assert rows % 32 == 0
    blocks = x.reshape(-1, rows // 32, 8, 4, cols)
    return blocks.transpose(2, 3).reshape(x.shape).contiguous()


def swizzle_block_scales(scales: torch.Tensor) -> torch.Tensor:
    """`[..., R, C]` row-major scales -> the same shape, R128c4."""
    *lead, rows, cols = scales.shape
    assert rows % 128 == 0 and cols % 4 == 0
    tiles = scales.reshape(-1, rows // 128, 4, 32, cols // 4, 4)
    return tiles.permute(0, 1, 4, 3, 2, 5).reshape(scales.shape).contiguous()


@triton.jit
def _scale_offsets(row, cols, COLS: tl.constexpr, TILE: tl.constexpr):
    """Offsets of scales `(row, cols)` of a `COLS`-column scale matrix for
    GEMM token tiles of `TILE` rows (0: row-major)."""
    if TILE >= 128:
        tile = (row // 128) * (COLS // 4) + cols // 4
        return tile * 512 + (row % 32) * 16 + (row // 32 % 4) * 4 + cols % 4
    if TILE > 0:
        tile = (row // 8) * (COLS // 4) + cols // 4
        return tile * 32 + (row % 8) * 4 + cols % 4
    return row * COLS + cols
