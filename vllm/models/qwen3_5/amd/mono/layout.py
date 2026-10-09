# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8-2.4T at TP8: the shapes the mono kernels are built for and the
addresses of the weights they read in place.

Plain integer arithmetic: the same functions index a tensor on the host (the
layout tests) and build an address from traced values in a kernel.

Routed experts (``AITER_MXFP4_MXFP4``): ``w13`` (E, 2 RI, HIDDEN / 2) and ``w2``
(E, HIDDEN, RI / 2) fp4x2, ``[gate; up]`` rows, through
``shuffle_weight(layout=(16, 16))``: a 16-row group's 16-byte K chunks as
``[chunk][row % 16][16 B]``. Their e8m0 scales through ``e8m0_shuffle`` over
(E N, K / 32): ``scale_index``.
"""

TP = 8
HIDDEN = 8192

# GDN (linear attention), per rank
HD = 128  # key and value head dim
NK = 16 // TP  # key heads: 2
NV = 128 // TP  # value heads: 16
KEY = NK * HD  # 256
VAL = NV * HD  # 2048
QKVZ = 2 * KEY + 2 * VAL  # 4608 in_proj_qkvz rows: [q | k | v | z]
BA = 2 * NV  # 32 in_proj_ba rows: [b | a]
CONV = 2 * KEY + VAL  # 2560 conv channels: q | k | v
CONV_W = 4
CORE = VAL  # 2048: the out_proj / o_proj K a rank holds

# MoE, per rank (experts TP-sharded, not EP)
E = 512
TOPK = 10
RI = 2048 // TP  # 256: routed intermediate a rank holds
SI = 2048 // TP  # 256: shared intermediate a rank holds
EPR = E // TP  # 64: router rows a rank computes (logits all-gathered)

MAX_TOKENS = 8  # an 8-row bf16 x tile is 131 KB of the CU's 160 KB LDS
MAX_U = MAX_TOKENS * TOPK  # distinct experts a step can touch

ROWS = 16  # a weight row group
FP4_STEP = 128  # K a scaled-MFMA step covers (a lane: 32 fp4 of one row)


def fp4_group_dwords(k):
    """Dwords of one 16-row group of an fp4 weight with K = ``k``."""
    return ROWS * k // 8


def fp4_tile_dword(expert_base, rg, k, st, lane):
    """Dword of lane ``lane``'s 16 B of 16-row group ``rg`` at K step ``st``:
    row 16 rg + lane % 16, fp4 K 128 st + 32 (lane / 16) .. + 32. ``expert_base``
    is the expert's first dword (``e * rows * k / 8``)."""
    return expert_base + rg * fp4_group_dwords(k) + st * 256 + lane * 4


def w13_base(e):
    """First dword of expert ``e``'s w13 (2 RI rows, K = HIDDEN)."""
    return e * (2 * RI) * HIDDEN // 8


def w13_group(gate_or_up, g):
    """Row group of w13 holding intermediate columns 16 g .. of the gate (0) or
    up (1) projection: ``[gate; up]`` rows, not interleaved."""
    return gate_or_up * (RI // ROWS) + g


def w2_base(e):
    """First dword of expert ``e``'s w2 (HIDDEN rows, K = RI)."""
    return e * HIDDEN * RI // 8


def scale_index(row, col, cols):
    """Byte of e8m0 scale (row, col) in aiter's ``e8m0_shuffle`` layout over a
    (rows, ``cols``) matrix: ``view(rows/32, 2, 16, cols/8, 2, 4)
    .permute(0, 3, 5, 2, 4, 1)``. ``row`` counts across experts (e N + n)."""
    r32, a, b = row // 32, (row // 16) % 2, row % 16
    c8, d, f = col // 8, (col // 4) % 2, col % 4
    return ((((r32 * (cols // 8) + c8) * 4 + f) * 16 + b) * 2 + d) * 2 + a


def w13_scale_index(e, n, kb):
    """Byte of w13's scale for row ``n`` (of 2 RI), K block ``kb`` (of HIDDEN / 32)."""
    return scale_index(e * 2 * RI + n, kb, HIDDEN // 32)


def w2_scale_index(e, n, kb):
    """Byte of w2's scale for row ``n`` (of HIDDEN), K block ``kb`` (of RI / 32)."""
    return scale_index(e * HIDDEN + n, kb, RI // 32)
