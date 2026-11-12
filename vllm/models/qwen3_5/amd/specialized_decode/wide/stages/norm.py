# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The residual add and RMSNorm after a TP exchange (the decoder layer's
`norm(hidden, residual)`), and the next GEMMs' activations.

`combine`: a CTA a row, thread `t` columns `32 t .. 32 t + 31` (one
MXFP4 scale group, half an exchange line group). The exchanged sum `y`
(the ranks' bf16 partials added in rank order, rounded to bf16: the stock
all-reduce's output) is added to the residual and rounded to bf16 (the new
residual, stored in place); `x = bf16(res * rsqrt(mean(res^2) + eps) * (1 +
w))` (the Gemma-style norm of Qwen3.5) is stored as bf16, and with `q4`
also as MXFP4 (`mx`'s activation quantization, the W4A4 GEMMs' input):
`"plain"` of `x`, `"h32"` of `bf16(x H32)`, the block Hadamard
rotation an `InputHadamardLinearMethod` applies before quantizing; a
`"_t"` suffix scales as the Triton FP4 GEMM does (`mx.scale_code`). Without
an exchange (the step's first layer) the residual is the embedding as is.
"""

from __future__ import annotations

import math

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

from vllm.models.qwen3_5.amd.specialized_decode.wide.common.grid import TraceInvariant
from vllm.models.qwen3_5.amd.specialized_decode.wide.common.ops import (
    CM_DEV,
    bf16_round,
    ld_i32x4,
    lds_ld,
    lds_st,
    pack_bf16x8,
    rsqrt,
    st,
    traced,
    unpack_bf16x8,
    wave_sum,
)
from vllm.models.qwen3_5.amd.specialized_decode.wide.layout import (
    BLOCKS,
    HIDDEN,
    THREADS,
    WAVES,
)
from vllm.models.qwen3_5.amd.specialized_decode.wide.stages import exchange as ex_
from vllm.models.qwen3_5.amd.specialized_decode.wide.stages import mx

COLS = HIDDEN // THREADS  # a thread's columns: one scale group
assert COLS == mx.GROUP
Q4_MODES = (None, "plain", "h32", "plain_t", "h32_t")
LDS_WORDS = WAVES


def i32(v):
    return fx.Int32(v)


def hadamard32(v):
    """`v` (32 f32) times the normalized Sylvester Hadamard matrix of order
    32 (`hadamard_matrix(32)`: symmetric, so row and column vectors agree):
    the fast transform's five butterfly levels, then `1 / sqrt(32)`."""
    v = list(v)
    h = 1
    while h < 32:
        for i in range(32):
            if i & h == 0:
                v[i], v[i + h] = v[i] + v[i + h], v[i] - v[i + h]
        h *= 2
    s = fx.Float32(1.0 / math.sqrt(32.0))
    return [e * s for e in v]


class Out(TraceInvariant):
    """A combine's outputs (byte addresses): `res` the residual [rows,
    HIDDEN] bf16 (read, then updated in place), `x` the normed rows bf16,
    with `q4` their MXFP4 codes `x4` ([rows, HIDDEN / 2] bytes) and scale
    code words `x4s` ([rows, HIDDEN / 32])."""

    def __init__(self, res, x, x4=0, x4s=0, q4=None):
        if q4 not in Q4_MODES:
            raise ValueError(f"q4 {q4!r}: one of {Q4_MODES}")
        self.res, self.x = fx.Int64(res), fx.Int64(x)
        self.x4, self.x4s, self.q4 = fx.Int64(x4), fx.Int64(x4s), q4
        self.h32 = q4 is not None and q4.startswith("h32")
        self.rceil = q4 is None or not q4.endswith("_t")


@traced
def combine(g, xch, out, w, lds, rows: int, eps: float, emb=None, res_in=None):
    """`xch`: the exchange whose lines hold `y`; None: `y` is `emb`
    ([rows, HIDDEN] bf16) added to `res_in` (same), or without `res_in`
    the first layer's residual. The sum is stored to `out.res`; `w`: the
    norm weight [HIDDEN] bf16; `lds`: `LDS_WORDS` f32 words."""
    col = g.tid * i32(COLS)
    wv = [unpack_bf16x8(ld_i32x4(w, col // 2 + 4 * q)) for q in range(4)]
    for row in range(g.bid, i32(rows), i32(BLOCKS)):
        base = row * i32(HIDDEN // 2) + col // 2  # i32 words
        if const_expr(xch is not None):
            src = out.res
        elif const_expr(res_in is not None):
            src = fx.Int64(res_in)
        else:
            src = fx.Int64(emb)
        old = [unpack_bf16x8(ld_i32x4(src, base + 4 * q, CM_DEV)) for q in range(4)]
        z = []
        if const_expr(xch is None and res_in is not None):
            for q in range_constexpr(4):
                y = unpack_bf16x8(ld_i32x4(fx.Int64(emb), base + 4 * q, CM_DEV))
                z += [bf16_round(old[q][e] + y[e]) for e in range(8)]
        elif const_expr(xch is None):
            for q in range_constexpr(4):
                z += list(old[q])
        else:
            group = g.tid // 2
            seg0 = (g.tid % 2) * 4
            gates = ex_.wait_row(g, xch, row, group)
            for q in range_constexpr(4):
                y = ex_.row_sum(xch, row, group, seg0 + q, gates)
                z += [bf16_round(old[q][e] + bf16_round(y[e])) for e in range(8)]
        ss = z[0] * z[0]
        for e in range_constexpr(1, COLS):
            ss = ss + z[e] * z[e]
        ss = wave_sum(ss)
        if g.lane == 0:
            lds_st(lds + g.wave, ss)
        fx.barrier()
        tot = lds_ld(lds)
        for v in range_constexpr(1, WAVES):
            tot = tot + lds_ld(lds + v)
        fx.barrier()
        rr = rsqrt(tot / fx.Float32(HIDDEN) + fx.Float32(eps))
        xs = [
            bf16_round(z[e] * rr * (fx.Float32(1.0) + wv[e // 8][e % 8]))
            for e in range(COLS)
        ]
        for q in range_constexpr(4):
            st(out.res, base + 4 * q, pack_bf16x8(z[8 * q : 8 * q + 8]), CM_DEV)
            st(out.x, base + 4 * q, pack_bf16x8(xs[8 * q : 8 * q + 8]), CM_DEV)
        if const_expr(out.q4 is not None):
            vals = xs
            if const_expr(out.h32):
                vals = [bf16_round(v) for v in hadamard32(xs)]
            code = mx.scale_code(mx.amax(vals), out.rceil)
            words = mx.quant_f32_group(vals, code)
            st(
                out.x4,
                (row * i32(HIDDEN) + col) // i32(8),
                fx.Vector.from_elements(words, fx.Int32),
                CM_DEV,
            )
            st(out.x4s, row * i32(HIDDEN // COLS) + g.tid, code, CM_DEV)
