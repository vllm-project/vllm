# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared CuTe device operations."""

from cutlass import cute
from cutlass.cute.nvgpu import cpasync


def simple_tma_copy(atom, src, dst, mbar=None, cache_policy=None):
    if isinstance(atom.op, cpasync.CopyBulkTensorTileG2SOp):
        gmem = src
        smem = dst
    elif isinstance(atom.op, cpasync.CopyBulkTensorTileS2GOp):
        smem = src
        gmem = dst
    else:
        raise ValueError
    s_part, g_part = cpasync.tma_partition(
        atom,
        0,
        cute.make_layout(1),
        cute.group_modes(smem, 0),
        cute.group_modes(gmem, 0),
    )
    if isinstance(atom.op, cpasync.CopyBulkTensorTileG2SOp):
        cute.copy(atom, g_part, s_part, tma_bar_ptr=mbar, cache_policy=cache_policy)
    elif isinstance(atom.op, cpasync.CopyBulkTensorTileS2GOp):
        cute.copy(atom, s_part, g_part, cache_policy=cache_policy)
    else:
        raise ValueError
