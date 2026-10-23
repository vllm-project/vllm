# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2 phase 2a: the reduce half of o_proj's all-reduce and the attention
boundary, on the seam CTAs j in [0, NA): columns 128 j .. of the 8192.

Stock reference: the custom all-reduce (bf16) of the o_proj partials, then
``_norm_residual_norm_fwd_kernel`` (``common/ops/fused_rmsnorm.py``) with no
post-norm weight (eps 1e-8), the per-channel gated blend ``alpha stream +
beta y``, and ``pre_ffn_norm`` (weight after the norm, gain center 1.0, eps
1e-5), bf16 out::

    attn_out = bf16(sum over src in rank order of ar[src, t, c])   (fp32 sum)
    y        = attn_out rsq(sum(attn_out^2) / 8192 + 1e-8)
    stream'  = alpha stream + beta y                                (fp32)
    ffn_in   = bf16((stream' rsq(sum(stream'^2) / 8192 + 1e-5)) (w + 1))

Wave w is token t = w (t < S live; waves past S only take the barriers); lane
l owns columns c = 128 j + 2 l, + 1. A row's sum of squares is the 64 slices'
partials (``ass`` / ``bss``, one pair a slice and row), each reduced by one
fixed butterfly (``wave_sum``), so every seam CTA of every rank derives the
same total from the same values. Pad rows are zeros everywhere (``select``:
NaN in a pad row's stream or partials never leaks); their partials are still
published, so no wait depends on a row being live.

Outputs: ``stream_out`` (fp32, plain stores), ``xrow`` (bf16 ffn_in rows,
device scope, scratch) and slice j's ``xrdy`` flag (``publish``: the xrow
stores drained, then the flag) for the router and the pooler.

``blend_norm`` (the boundary past its input, from the first sum of squares
to the bf16 row) is shared with K1's boundary (``front/boundary.py``,
DESIGN_BC 5.4): same order, same rounding points, its own regions, eps and
outputs.

The stage keeps everything in registers: ``lds_struct`` is the design's
reserved 256 B member (stage key ``"seam"``), which ``stage_seam`` does not
touch (no LDS hand-over barrier is owed).

The pattern is K3's AttnRes seam (vLLM #60683 ``mono/attention/back.py``
``stage_reduce`` and ``kda.py`` ``stage_slice`` / ``stage_mix``: per-slice
partial sums of squares through scratch pairs, ``publish`` of the x rows).
"""

import functools

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr.typing import T

from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import (
    CM_DEV,
    bf2_f32,
    bf16_pair,
    bf16_round,
    hw_rsq,
    ld_pad,
    rsrc,
    traced,
    wave_sum,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.plan import (
    WAVES,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ranks import (
    sum_partials,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.sync import (
    publish,
    uniform_addr,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.spec import (
    COLS,
    GAIN,
    HIDDEN,
    MAX_ROWS,
    NA,
    POST_ATTN_EPS,
    PRE_FFN_EPS,
    TP,
    ar_pair,
    ss_pair,
    xrdy_pair,
    xrow_elem,
)

assert COLS == 2 * 64 and NA == HIDDEN // COLS  # a lane 2 columns of a slice
assert NA == 64  # a lane reads one slice's partial of each sum of squares
assert MAX_ROWS <= WAVES  # a wave a row

INV_H = 1.0 / HIDDEN


@functools.cache
def lds_struct(s: int):
    """The stage's LDS member (reserved, 64 f32; registers carry the stage)."""

    @fx.struct
    class SeamLds:
        v: fx.Array[fx.Float32, 64, 16]

    return SeamLds


def _ld_f32x2(ptr, i):
    """fp32 elements i, i + 1 (i even)."""
    v = fx.Vector(bo.buffer_load(rsrc(ptr), i, vec_width=2, dtype=T.f32))
    return v[0], v[1]


def seam_loads(c, j):
    """Slice j's inputs that need no wait, issued before the stage's polls:
    this wave's row of the stream, the gate coefficients and the pre-FFN norm
    weight at the lane's two columns, and the row's pad flag. Waves past S
    load row S - 1 (in bounds; never used)."""
    s, lane, wave = c["S"], c["lane"], c["wave"]
    a = c["args"]
    t = fx.min(wave, s - 1)
    col = j * COLS + 2 * lane
    s0, s1 = _ld_f32x2(a["stream"], t * HIDDEN + col)
    a0, a1 = _ld_f32x2(a["attn_alpha"], col)
    b0, b1 = _ld_f32x2(a["attn_beta"], col)
    ww = fx.Int32(
        bo.buffer_load(rsrc(a["w_preffn"]), col // 2, vec_width=1, dtype=T.i32)
    )
    w0, w1 = bf2_f32(ww)
    return {
        "t": t,
        "col": col,
        "s": (s0, s1),
        "alpha": (a0, a1),
        "beta": (b0, b1),
        "w": (w0 + fx.Float32(GAIN), w1 + fx.Float32(GAIN)),
        "pad": ld_pad(a["pad"], t),
    }


def _slices_total(c, region, t):
    """Row t's sum of squares over the 64 slices' partials (pair (l, t) of
    ``region`` for lane l), one butterfly: the same value on every lane and
    in every seam CTA. The base is ``uniform_addr``'d: ``region`` was just
    put by lane 0."""
    v = c["poll"]([(uniform_addr(region), ss_pair(c["lane"], t), 1)])[0][0]
    return wave_sum(v.bitcast(fx.Float32))


@traced
def blend_norm(c, j, pre, x0, x1, regions, post_eps, pre_eps, outs):
    """The residual boundary past its input ``x`` (lane columns ``col``,
    ``col + 1`` of row ``t``; zero on a pad row), shared by K2's seam and K1's
    boundary (``front/boundary.py``)::

        y      = x rsq(sum(x^2) / 8192 + post_eps)
        n      = alpha stream + beta y          -> outs[0] (fp32, plain)
        z      = bf16((n rsq(sum(n^2) / 8192 + pre_eps)) w)  -> outs[1] (CM_DEV)

    ``pre``: the slice's loads (``t``, ``col``, ``s``, ``alpha``, ``beta``,
    ``w`` = the next norm's weight + gain center, ``pad``). ``regions``: the two
    sum-of-squares mailboxes (slice j's partial of row t at ``ss_pair(j, t)``).
    ``outs``: (stream_out base, bf16 row base, row element index of (t, col)).
    Called by the live waves (t < S); the caller publishes the slice's flag."""
    lane = c["lane"]
    first, second = regions
    stream_out, rows, row_elem = outs
    t, col, dead = pre["t"], pre["col"], pre["pad"]
    zero = fx.Float32(0.0)
    p = wave_sum(x0 * x0 + x1 * x1)
    if lane == 0:
        c["put"](first, ss_pair(j, t), p)
    tot = _slices_total(c, first, t)
    r = hw_rsq(tot * fx.Float32(INV_H) + fx.Float32(post_eps))
    s0, s1 = pre["s"]
    a0, a1 = pre["alpha"]
    b0, b1 = pre["beta"]
    n0 = dead.select(zero, a0 * s0 + b0 * (x0 * r))
    n1 = dead.select(zero, a1 * s1 + b1 * (x1 * r))
    bo.buffer_store(
        fx.Vector.from_elements([n0, n1], fx.Float32),
        rsrc(stream_out),
        t * HIDDEN + col,
    )
    q = wave_sum(n0 * n0 + n1 * n1)
    if lane == 0:
        c["put"](second, ss_pair(j, t), q)
    tot2 = _slices_total(c, second, t)
    r2 = hw_rsq(tot2 * fx.Float32(INV_H) + fx.Float32(pre_eps))
    w0, w1 = pre["w"]
    f0 = dead.select(zero, (n0 * r2) * w0)
    f1 = dead.select(zero, (n1 * r2) * w1)
    bo.buffer_store(
        bf16_pair(f0, f1),
        rsrc(rows),
        row_elem(t, col) // 2,
        cache_modifier=CM_DEV,
    )


@traced
def stage_seam(c, j, pre):
    """Slice j: ar (8 ranks, rank order) -> attn_out -> post-attention norm
    and the gated blend -> ``stream_out``; the pre-FFN norm -> ``xrow``; then
    the slice's ``xrdy`` flag."""
    s, tid, wave = c["S"], c["tid"], c["wave"]
    a, scr, peer = c["args"], c["scr"], c["peer"]
    t, col, dead = pre["t"], pre["col"], pre["pad"]
    zero = fx.Float32(0.0)
    if wave < s:
        lo, hi = sum_partials(
            c["poll"], peer["ar_own"], lambda src: ar_pair(src, t, col), TP
        )
        # the custom all-reduce's bf16 result
        x0 = dead.select(zero, bf16_round(lo))
        x1 = dead.select(zero, bf16_round(hi))
        blend_norm(
            c,
            j,
            pre,
            x0,
            x1,
            (scr["ass"], scr["bss"]),
            POST_ATTN_EPS,
            PRE_FFN_EPS,
            (a["stream_out"], scr["xrow"], xrow_elem),
        )
    publish(c["put"], scr["xrdy"], xrdy_pair(j), fx.Int32(1), tid == 0)
