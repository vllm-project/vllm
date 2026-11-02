# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MXFP4 weight layouts the kernels read in place, and the routed-expert
addressing (``MoeDims``).

Plain integer arithmetic: the same functions index a tensor on the host (the
layout tests) and build an address from traced values in a kernel.

Routed experts (``AITER_MXFP4_MXFP4``): ``w13`` (E, 2 RIP, HIDDEN / 2) and
``w2`` (E, HIDDEN, RIP / 2) fp4x2, ``[gate; up]`` rows, each half zero-padded
from RI to RIP (the backend's 128 alignment), through
``shuffle_weight(layout=(16, 16))``: a 16-row group's 16-byte K chunks as
``[chunk][row % 16][16 B]``. Their e8m0 scales through ``e8m0_shuffle`` over
(E N, K / 32), K / 32 padded to a multiple of 8 (``scale_cols``):
``scale_index``. The online-MXFP4 linears on AITER's ASM FP4 GEMM hold the
same layouts, one "expert"; on its Triton FP4 GEMM they are row-major with
transposed scales (``fp4_row_dword``, ``scale_t_index``).
"""

from dataclasses import dataclass

ROWS = 16  # a weight row group
FP4_STEP = 128  # K a scaled-MFMA step covers (a lane: 32 fp4 of one row)


@dataclass(frozen=True)
class MoeDims:
    """A rank's routed-expert shapes, what the MoE stages read: the hidden
    size, the routed intermediate a rank holds (RI) and the weights' padded
    one (RIP)."""

    hidden: int
    ri: int
    rip: int


@dataclass(frozen=True)
class GdnDims:
    """A rank's GDN shapes, what the GDN stages read: its key and value
    heads and PROJ's entries a token (in_proj's outputs, [q | k | v | z ..])."""

    nk: int
    nv: int
    proj: int


def fp4_group_dwords(k):
    """Dwords of one 16-row group of an fp4 weight with K = ``k``."""
    return ROWS * k // 8


def fp4_tile_dword(expert_base, rg, k, st, lane):
    """Dword of lane ``lane``'s 16 B of 16-row group ``rg`` at K step ``st``:
    row 16 rg + lane % 16, fp4 K 128 st + 32 (lane / 16) .. + 32. ``expert_base``
    is the expert's first dword (``e * rows * k / 8``)."""
    return expert_base + rg * fp4_group_dwords(k) + st * 256 + lane * 4


def fp4_row_dword(rg, k, st, lane):
    """``fp4_tile_dword`` in a row-major fp4 weight (``[N, k / 2]``, the Triton
    FP4 GEMM's): the same 16 B, row 16 rg + lane % 16 at fp4 K 128 st + 32
    (lane / 16)."""
    return (rg * ROWS + lane % ROWS) * (k // 8) + st * 16 + (lane // 16) * 4


def scale_t_index(row, kb, rows):
    """Byte of e8m0 scale (row, K block ``kb``) in a transposed ``[K / 32,
    rows]`` scale (the Triton FP4 GEMM's)."""
    return kb * rows + row


def w13_base(d: MoeDims, e):
    """First dword of expert ``e``'s w13 (2 RIP rows, K = hidden). The constant
    folds first: ``e`` is i32 in the kernel and e 2 RI HIDDEN is 2^31 at e 512."""
    return e * (2 * d.rip * d.hidden // 8)


def w13_group(d: MoeDims, gate_or_up, g):
    """Row group of w13 holding intermediate columns 16 g .. of the gate (0) or
    up (1) projection: ``[gate; up]`` rows, each RIP, not interleaved."""
    return gate_or_up * (d.rip // ROWS) + g


def w2_base(d: MoeDims, e):
    """First dword of expert ``e``'s w2 (hidden rows, K = RIP)."""
    return e * (d.hidden * d.rip // 8)


def scale_index(row, col, cols):
    """Byte of e8m0 scale (row, col) in aiter's ``e8m0_shuffle`` layout over a
    (rows, ``cols``) matrix: ``view(rows/32, 2, 16, cols/8, 2, 4)
    .permute(0, 3, 5, 2, 4, 1)``. ``row`` counts across experts (e N + n)."""
    r32, a, b = row // 32, (row // 16) % 2, row % 16
    c8, d, f = col // 8, (col // 4) % 2, col % 4
    return ((((r32 * (cols // 8) + c8) * 4 + f) * 16 + b) * 2 + d) * 2 + a


def scale_cols(k):
    """Columns of a shuffled scale for K = ``k``: ``shuffle_scale`` pads K / 32
    to a multiple of 8 (with e8m0 1.0)."""
    return (k // 32 + 7) // 8 * 8


def scale_code_shift(st, rg):
    """Bit offset of K step ``st``'s code for 16-row group ``rg`` in the dword
    ``scale_index`` puts it in (its block ``4 st + lane / 16``): byte
    (st % 2) 2 + rg % 2."""
    return (st % 2) * 16 + (rg % 2) * 8


def w13_scale_index(d: MoeDims, e, n, kb):
    """Byte of w13's scale for row ``n`` (of 2 RIP), K block ``kb``."""
    return scale_index(e * 2 * d.rip + n, kb, scale_cols(d.hidden))


def w2_scale_index(d: MoeDims, e, n, kb):
    """Byte of w2's scale for row ``n`` (of hidden), K block ``kb``."""
    return scale_index(e * d.hidden + n, kb, scale_cols(d.rip))
