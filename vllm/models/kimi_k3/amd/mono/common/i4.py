# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed-int4 expert weights (gfx942): the layout vLLM leaves K3's MXFP4 experts
in on gfx942 (``Mxfp4MoEMethod._setup_kernel_k3_situ_gfx942``: requantized by
``per_1x32_i4_quant``, ``shuffle_weight(16, 16)``, ``pack_int8_to_packed_int4``,
scales by ``shuffle_scale_for_int4``) and aiter's a16wi4 dequant of it.

A weight [E, N, K] is 16-row groups of 64-K blocks of 512 B. Lane l's 8 B of a
block (byte 8 l) are row l % 16's K 16 (l / 16) .. + 15: two dwords of eight
nibbles, byte j holding K j (low nibble) and K j + 4 (high). The scales (a
group of 32 K) are bf16 [E, K / 64, N, 2]: dword (k0, n) holds row n's groups
2 k0 (low half) and 2 k0 + 1 (high), lane l's the half (l / 16) // 2.

The MFMA's other operand is read at the same K (16 bf16 a lane a block), so a
block is two ``mfma_bf16`` (four 16x16x16) of ``dequant``'s two v8bf16.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import rocdl
from flydsl.expr.typing import T

from vllm.models.kimi_k3.amd.mono.common.ops import CM_NT

BLOCK_K = 64
GROUP = 32


def group_dw(k):
    """Dwords of a 16-row group of a K = ``k`` weight."""
    return 2 * k


def expert_dw(n, k):
    """Dwords of one expert's [n, k] weight."""
    return n * k // 8


def block_load(r_w, base, blk, lane):
    """A lane's dwordx2 of 64-K block ``blk`` of the 16-row group at dword
    ``base``. Nontemporal: a weight a step reads once."""
    return fx.Vector(
        bo.buffer_load(
            r_w, base + blk * 128 + 2 * lane, vec_width=2, dtype=T.i32,
            cache_modifier=CM_NT,
        )
    )  # fmt: skip


def scale_word(r_s, e, row, blk, n, k):
    """The scale dword of (expert e, weight row ``row``, 64-K block ``blk``) of
    an [n, k] weight."""
    return fx.Int32(
        bo.buffer_load(
            r_s, (e * (k // BLOCK_K) + blk) * n + row, vec_width=1, dtype=T.i32
        )
    )


def lane_scale(word, lane):
    """The f32 group scale of this lane's 16 K in a scale dword."""
    hi = (lane // 16) >= 2
    return hi.select(word & fx.Int32(-65536), word << 16).bitcast(fx.Float32)


def _nibbles_bf16x8(raw, eff):
    """Eight int4 of ``raw`` (K j low, K j + 4 high) times ``eff`` / 16 ->
    v8bf16 K 0 .. 7, by high-16 truncation: aiter's gfx942
    ``_int4_nibble_to_bf16x8(old_pack=True)`` bit for bit."""
    odd = fx.Int32(raw).shrui(fx.Int32(4))
    lo = [
        fx.Float32(rocdl.cvt_off_f32_i4(fx.Int32(raw).ir_value(), byte_sel=j)) * eff
        for j in range(4)
    ]
    hi = [
        fx.Float32(rocdl.cvt_off_f32_i4(odd.ir_value(), byte_sel=j)) * eff
        for j in range(4)
    ]
    f = lo + hi
    words = []
    for i in range(4):
        b0 = f[2 * i].bitcast(fx.Int32)
        b1 = f[2 * i + 1].bitcast(fx.Int32)
        words.append(b0.shrui(fx.Int32(16)) | (b1 & fx.Int32(-65536)))
    return fx.Vector.from_elements(words, fx.Int32).bitcast(fx.BFloat16)


def dequant(raw2, scale):
    """A lane's block dwordx2 with its f32 group scale -> (v8bf16 K 0 .. 7,
    v8bf16 K 8 .. 15) of its 16."""
    eff = scale * fx.Float32(16.0)
    return _nibbles_bf16x8(raw2[0], eff), _nibbles_bf16x8(raw2[1], eff)
