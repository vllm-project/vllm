# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K1's residual boundary (DESIGN_BC 5.4, Phase B1) on the weight-free CTAs
j in [0, NA): columns 128 j .. of the 8192; and the epilogue CTAs' step
inputs (5.6), loaded at CTA entry with the boundary's.

Stock reference: ``_norm_residual_norm_fwd_kernel``
(``common/ops/fused_rmsnorm.py``) with the previous layer's ``post_ffn_norm``
(weight before the norm, gain center 1.0, eps 1e-8), the per-channel gated
blend ``alpha stream + beta y`` (``add_ffn``) and this layer's
``pre_qkv_proj_norm`` (weight after the norm, gain center 1.0, eps 1e-5), bf16
out::

    x        = f32(branch) (w_post + 1)                     (one fp32 product)
    y        = x rsq(sum(x^2) / 8192 + 1e-8)
    stream'  = alpha stream + beta y                         (fp32)
    attn_in  = bf16((stream' rsq(sum(stream'^2) / 8192 + 1e-5)) (w_pre + 1))

This is K2's seam (``stages/seam.py``) with the deferred branch (plain loads:
the previous launch wrote it) in place of the all-reduce, and the post weight
applied first; past ``x`` both run ``seam.blend_norm``. Wave w is token t = w
(t < S live; waves past S only take the barriers); lane l owns columns
c = 128 j + 2 l, + 1. A row's sums of squares are the 64 slices' partials
(``fss`` / ``nss``), one fixed butterfly each, so every boundary CTA derives
the same totals. Pad rows are zeros everywhere (``select``: NaN in a pad row's
branch or stream never leaks); their partials are still published.

Outputs: ``stream_out`` (fp32, plain stores; K2's ``stream``), ``arow`` (bf16
attn_in rows, device scope, scratch) and slice j's ``ardy`` flag (``publish``:
the arow stores drained, then the flag) for the GEMV CTAs whose K parts read
slice j (``front/qkv.py``).

Rule 8 (DESIGN_BC 10.4): the boundary issues no weight load. At its polls a
wave has in flight only ``boundary_loads``' operands and ``epilogue_inputs``'
three loads (pad, position, slot), and every boundary CTA puts before it
polls, so the boundary CTAs depend only on each other (5.3). The stage keeps
everything in registers: no LDS member, no hand-over barrier owed.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr.typing import T

from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import (
    bf2_f32,
    ld_i32,
    ld_pad,
    rsrc,
    traced,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.plan import (
    WAVES,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.sync import publish
from vllm.models.muse_spark.amd.specialized_decode.mono.spec import (
    AROW,
    COLS,
    E_UNITS,
    GAIN,
    HIDDEN,
    MAX_ROWS,
    NA,
    POST_FFN_EPS,
    RMS_EPS,
    ardy_pair,
    arow_elem,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.stages.seam import (
    blend_norm,
)

assert COLS == 2 * 64 and NA == HIDDEN // COLS  # a lane 2 columns of a slice
assert NA == 64  # a lane reads one slice's partial of each sum of squares
assert MAX_ROWS <= WAVES  # a wave a row
assert E_UNITS * MAX_ROWS <= NA  # the epilogue CTAs are boundary CTAs


def _ld_f32x2(ptr, i):
    """fp32 elements i, i + 1 (i even)."""
    v = fx.Vector(bo.buffer_load(rsrc(ptr), i, vec_width=2, dtype=T.f32))
    return v[0], v[1]


def _ld_i64(ptr, i):
    """int64 element i (two words: the high one sign-extends)."""
    w = fx.Vector(bo.buffer_load(rsrc(ptr), i * 2, vec_width=2, dtype=T.i32))
    return (fx.Int64(w[1]) << 32) | fx.Int64(fx.Uint32(w[0]))


def _gained(ptr, col):
    """A bf16 norm weight's two columns ``col``, ``col + 1`` + the gain
    center, one f32 add (``fused_rmsnorm.py``'s ``_effective_weight``)."""
    w0, w1 = bf2_f32(ld_i32(ptr, col // 2))
    return w0 + fx.Float32(GAIN), w1 + fx.Float32(GAIN)


def boundary_loads(c, j):
    """Slice j's inputs, all issued before the stage's polls (none needs a
    wait): this wave's rows of the stream and the branch, the blend
    coefficients and both norm weights at the lane's two columns, and the
    row's pad flag. Waves past S load row S - 1 (in bounds; never used). The
    keys ``seam.blend_norm`` reads are ``seam.seam_loads``'."""
    s, lane, wave = c["S"], c["lane"], c["wave"]
    a = c["args"]
    t = fx.min(wave, s - 1)
    col = j * COLS + 2 * lane
    s0, s1 = _ld_f32x2(a["stream"], t * HIDDEN + col)
    a0, a1 = _ld_f32x2(a["ffn_alpha"], col)
    b0, b1 = _ld_f32x2(a["ffn_beta"], col)
    p0, p1 = _gained(a["w_postffn"], col)
    w0, w1 = _gained(a["w_preqkv"], col)
    x0, x1 = bf2_f32(ld_i32(a["branch"], (t * HIDDEN + col) // 2))
    return {
        "t": t,
        "col": col,
        "s": (s0, s1),
        "alpha": (a0, a1),
        "beta": (b0, b1),
        "wpost": (p0, p1),
        "w": (w0, w1),
        "branch": (x0, x1),
        "pad": ld_pad(a["pad"], t),
    }


def epilogue_inputs(c, u):
    """Epilogue unit u's step inputs (DESIGN_BC 5.6), issued at CTA entry with
    ``boundary_loads`` (before any poll): token t = u // E_UNITS, its pad
    flag, position and slot (int64, ``positions[t]`` / ``slots[t]``; a slot
    < 0 inserts nothing). Every lane holds the same values (VGPRs: ``uniform``
    them where an address must be wave-uniform). A CTA past E_UNITS S (no
    epilogue unit) loads row S - 1: in bounds, never used. ``g5`` = u %
    E_UNITS is the unit's row group (``epilogue.unit_row``)."""
    s = c["S"]
    a = c["args"]
    t = fx.min(u // E_UNITS, s - 1)
    return {
        "t": t,
        "g5": u % E_UNITS,
        "pad": ld_pad(a["pad"], t),
        "pos": _ld_i64(a["positions"], t),
        "slot": _ld_i64(a["slots"], t),
    }


@traced
def stage_boundary(c, j, pre):
    """Slice j: the deferred branch -> the post-FFN norm (weight first) and
    the gated blend -> ``stream_out``; the pre-qkv norm -> ``arow``; then the
    slice's ``ardy`` flag."""
    s, tid, wave = c["S"], c["tid"], c["wave"]
    a, scr = c["args"], c["scr"]
    dead = pre["pad"]
    zero = fx.Float32(0.0)
    if wave < s:
        x0, x1 = pre["branch"]
        p0, p1 = pre["wpost"]
        # fused_rmsnorm.py's POST_INPUT_SCALE: the weight before the norm
        x0 = dead.select(zero, x0 * p0)
        x1 = dead.select(zero, x1 * p1)
        blend_norm(
            c,
            j,
            pre,
            x0,
            x1,
            (scr["fss"], scr["nss"]),
            POST_FFN_EPS,
            RMS_EPS,
            (a["stream_out"], scr[AROW], arow_elem),
        )
    publish(c["put"], scr["ardy"], ardy_pair(j), fx.Int32(1), tid == 0)
