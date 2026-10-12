# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8-Flash-Next (``qwen4_exp``): the shapes the mono kernels are built
for at a TP size, and the addresses of the weights they read in place.

Plain integer arithmetic: the same functions index a tensor on the host (the
layout tests) and build an address from traced values in a kernel. A
``Layout`` is a build key's field, so every shape below is a compile-time
constant of a kernel.

Routed experts (``AITER_MXFP4_MXFP4``): ``w13`` (E, 2 RIP, HIDDEN / 2) and ``w2``
(E, HIDDEN, RIP / 2) fp4x2, ``[gate; up]`` rows, each half zero-padded from RI
to RIP (the backend's 128 alignment), through ``shuffle_weight(layout=(16,
16))``: a 16-row group's 16-byte K chunks as ``[chunk][row % 16][16 B]``.
Their e8m0 scales through ``e8m0_shuffle`` over (E N, K / 32), K / 32 padded
to a multiple of 8 (``scale_cols``): ``scale_index``. The online-MXFP4 linears
(in_proj_qkvz, out_proj) hold the same layouts, one "expert".
"""

from dataclasses import dataclass

BLOCKS = 256  # CTAs a launch, all resident at once: one a CU
MAX_TOKENS = 8
ROWS = 16  # a weight row group
FP4_STEP = 128  # K a scaled-MFMA step covers (a lane: 32 fp4 of one row)
TPS = (2,)  # the TP sizes the kernels are built for

# the model (``Qwen4ExpTextConfig``)
HIDDEN = 2560
HC = 4  # hyper-connection streams
HH = HC * HIDDEN  # the multi-stream state's width
LORA = 320  # hc_lowrank
INJ = HC  # injection logits
DOWN_ROWS = 336  # down + inject rows: 320 + 4 + 12 pad (16-row aligned)
HD = 128  # GDN key and value head dim
CONV_W = 4
E = 512
TOPK = 10
MOE_I = 640  # moe_intermediate_size
QSA_HEADS = 24
QSA_HD = 256


def need(ok: bool, what: str) -> None:
    """A launch argument check that ``python -O`` keeps: the kernels read
    out-of-bounds memory silently where a fault would be wanted."""
    if not ok:
        raise ValueError(f"Qwen3.8-Flash-Next mono decode: {what}")


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


@dataclass(frozen=True)
class Layout:
    """The per-rank shapes at TP size ``tp``."""

    tp: int

    def __post_init__(self):
        need(self.tp in TPS, f"tensor parallel size {self.tp}, built for {TPS}")

    # GDN (linear attention): 16 key / 48 value heads
    @property
    def NK(self) -> int:
        return 16 // self.tp

    @property
    def NV(self) -> int:
        return 48 // self.tp

    @property
    def KEY(self) -> int:
        return self.NK * HD

    @property
    def VAL(self) -> int:
        return self.NV * HD

    @property
    def QKVZ(self) -> int:
        """in_proj_qkvz rows: [q | k | v | z]."""
        return 2 * self.KEY + 2 * self.VAL

    @property
    def BA(self) -> int:
        """in_proj_ba rows: [b | a]."""
        return 2 * self.NV

    @property
    def CONV(self) -> int:
        """Conv channels: q | k | v."""
        return 2 * self.KEY + self.VAL

    @property
    def CORE(self) -> int:
        """out_proj / o_proj K a rank holds (GDN: NV HD; QSA: heads HD)."""
        assert QSA_HEADS // self.tp * QSA_HD == self.NV * HD
        return self.NV * HD

    # MoE: experts TP-sharded (not EP); the router replicated
    @property
    def RI(self) -> int:
        """Routed intermediate a rank holds."""
        return MOE_I // self.tp

    @property
    def RIP(self) -> int:
        """RI as the weights hold it: padded to the backend's 128."""
        return _round_up(self.RI, 128)


def fp4_group_dwords(k):
    """Dwords of one 16-row group of an fp4 weight with K = ``k``."""
    return ROWS * k // 8


def fp4_tile_dword(expert_base, rg, k, st, lane):
    """Dword of lane ``lane``'s 16 B of 16-row group ``rg`` at K step ``st``:
    row 16 rg + lane % 16, fp4 K 128 st + 32 (lane / 16) .. + 32. ``expert_base``
    is the expert's first dword (``e * rows * k / 8``)."""
    return expert_base + rg * fp4_group_dwords(k) + st * 256 + lane * 4


def w13_base(lay: Layout, e):
    """First dword of expert ``e``'s w13 (2 RIP rows, K = HIDDEN)."""
    return e * (2 * lay.RIP * HIDDEN // 8)


def w13_group(lay: Layout, gate_or_up, g):
    """Row group of w13 holding intermediate columns 16 g .. of the gate (0) or
    up (1) projection: ``[gate; up]`` rows, each RIP, not interleaved."""
    return gate_or_up * (lay.RIP // ROWS) + g


def w2_base(lay: Layout, e):
    """First dword of expert ``e``'s w2 (HIDDEN rows, K = RIP)."""
    return e * (HIDDEN * lay.RIP // 8)


def scale_index(row, col, cols):
    """Byte of e8m0 scale (row, col) in aiter's ``e8m0_shuffle`` layout over a
    (rows, ``cols``) matrix: ``view(rows/32, 2, 16, cols/8, 2, 4)
    .permute(0, 3, 5, 2, 4, 1)``. ``row`` counts across experts (e N + n)."""
    r32, a, b = row // 32, (row // 16) % 2, row % 16
    c8, d, f = col // 8, (col // 4) % 2, col % 4
    return ((((r32 * (cols // 8) + c8) * 4 + f) * 16 + b) * 2 + d) * 2 + a


def scale_code_shift(st, rg):
    """Bit offset of K step ``st``'s code for 16-row group ``rg`` in the dword
    ``scale_index`` puts it in (its block ``4 st + lane / 16``): byte
    (st % 2) 2 + rg % 2."""
    return (st % 2) * 16 + (rg % 2) * 8


def w13_scale_index(lay: Layout, e, n, kb):
    """Byte of w13's scale for row ``n`` (of 2 RIP), K block ``kb`` (of HIDDEN / 32)."""
    return scale_index(e * 2 * lay.RIP + n, kb, scale_cols(HIDDEN))


def scale_cols(k):
    """Columns of a shuffled scale for K = ``k``: ``shuffle_scale`` pads K / 32
    to a multiple of 8 (with e8m0 1.0)."""
    return _round_up(k // 32, 8)


def w2_scale_index(lay: Layout, e, n, kb):
    """Byte of w2's scale for row ``n`` (of HIDDEN), K block ``kb`` (of RIP / 32)."""
    return scale_index(e * HIDDEN + n, kb, scale_cols(lay.RIP))
