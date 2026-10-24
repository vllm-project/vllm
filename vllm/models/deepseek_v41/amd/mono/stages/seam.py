# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/models/deepseek_v41/mono/kernels/attn_pre.py (slice, gate, norm) and
# atom/models/deepseek_v41/mono/kernels/attn_post.py (the attention's TP reduce).
"""The mono layer's mHC seam stages, each a task table of a persistent launch
handing off through the tagged mailbox:

    slice   (160, 32 hidden columns x 4 streams each): the owed post folded into
            the residual (R_new, bf16 out), the collapsed layer input (mailbox
            LIN) and the partial mix projection / sum of squares (mailbox PMIX)
    gate    (S): the 160 partials -> post / comb (Sinkhorn) and the next pre
    norm    (S): the RMSNorm -> the bf16 row, handed to the caller's store
    reduce  (160): the FFN seam's attention output, the TP ranks' partials
            (pushed to every rank's ATTN region) summed in the all-reduce's
            order -> ``pend_lds``, where its slice reads it
    push    (160): a rank's partial given in memory (an attention vLLM ran)
            to every rank's ATTN region, ahead of the slice's reduce

Every elementwise step follows aiter's ``mhc_fused_post_pre_delayed_rmsnorm``
rounding; the mix projection and the sums of squares, split-K reductions,
differ only in their split and order.
"""

import math
import operator

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from vllm.models.deepseek_v41.amd.mono.common.ops import (
    batched_rounds,
    bf2_f32,
    bf16_pair,
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    hw_exp2,
    hw_rcp,
    hw_rsq,
    lane_gather,
    rsrc,
    sum_partials,
    traced,
    xred,
)
from vllm.models.deepseek_v41.amd.mono.common.plan import (
    THREADS,
    WAVES,
    cdiv,
    pair_layout,
)
from vllm.models.deepseek_v41.amd.mono.common.sync import preg
from vllm.models.deepseek_v41.amd.mono.stages.gemv import (
    TILE,
    ld_bf,
    ld_f32,
    plus,
    token_tiles,
)

HC = 4
HIDDEN = 5120
K_MIX = HC * HIDDEN
MIX = HC * (HC + 2)  # 24 projections: pre 4 | post 4 | comb 16
EPS = 1e-20
HC_EPS = 1e-6
SINKHORN = 20
POST_MULT = 2.0
LOG2E = 1.0 / math.log(2.0)

COLS = 32  # hidden columns a slice task
SLICES = HIDDEN // COLS  # 160
KT = HC * COLS  # projection columns a slice task
# Tuned builds pad the LDS rows of R_new (``rl``, a token's KT) and of the fn
# weights (``fl``, a projection's KT) by a word: a MFMA operand's 16 rows (16
# tokens / projections at one column) then sit in 16 banks, not one (KT = 128
# words). A context's ``rl_row`` / ``fl_row`` are its rows' strides.
RL_ROW = KT + 1
FL_ROW = KT + 1


def fl_at(e):
    """A tuned ``fl``'s word of element e of the [MIX, KT] fn weights."""
    return (e // KT) * FL_ROW + e % KT


PART = MIX + 1  # a slice's partials a token: the mixes, then the sum of squares
GATE_BLOCK = 8  # slices a gate thread sums


def scratch_layout(tokens: int) -> dict[str, tuple[int, int]]:
    """Mailbox region -> (byte offset, bytes) of a seam; 8 B a pair."""
    pairs = {
        "lin": tokens * HIDDEN // 2,  # bf16 pairs
        "pmix": SLICES * tokens * PART,
    }
    return pair_layout(pairs.items())


def attn_peer_bytes(tokens: int, tp: int) -> int:
    """ATTN: ``[source rank][token][hidden / 2]`` bf16 pairs."""
    return tp * tokens * HIDDEN // 2 * 8


def attn_region_pair(src, t, s, col):
    return (src * s + t) * (HIDDEN // 2) + col // 2


# A tuned build's ATTN hand-off unit: 16 B, three bf16-pair words and the tag
# (``Mailbox.put_unit``), a quarter of the xGMI bytes tags rather than half. A
# wo_b row tile (16 hidden columns, 8 words a token) of a token tile's n tokens
# is 8 n words in token-major order, word f in unit f // 3 at slot f % 3:
# ``[token tile][source rank][wo_b tile][unit]``, every token tile but the last
# a full one. It fits the pair layout's bytes (``attn_units_fit``).
WOB_TILES = HIDDEN // 16  # 320
TILE_WORDS = 8  # words a wo_b tile's token
UNIT_WORDS = 3


def attn_units(n):
    """The units of one wo_b tile's n tokens."""
    return cdiv(TILE_WORDS * n, UNIT_WORDS)


def attn_unit(src, tile, units, wt, j, tp):
    """Unit j of wo_b tile ``wt``'s token tile ``tile`` (``units`` a wo_b tile:
    ``attn_units`` of its tokens) from source rank ``src``."""
    return (
        tile * (tp * WOB_TILES * attn_units(TILE)) + (src * WOB_TILES + wt) * units + j
    )


def attn_units_fit(s, tp):
    """Whether step width s's units fit the ATTN region (``attn_peer_bytes``)."""
    tiles = token_tiles(s)
    n = tiles[-1][1]
    last = attn_unit(tp - 1, len(tiles) - 1, attn_units(n), WOB_TILES - 1, 0, tp)
    return (last + attn_units(n)) * 16 <= attn_peer_bytes(s, tp)


def _pick3(v, q):
    return (q == 0).select(v[0], (q == 1).select(v[1], v[2]))


def attn_words(c, own, t, col, nw):
    """The ``nw`` (1 or 2) bf16-pair words of every source rank at (token t,
    columns col ..), from the ATTN units (``attn_unit``): one poll batch ->
    [rank][word]."""
    s, tp = c["S"], c["d"].tp
    tiles = token_tiles(s)
    if const_expr(len(tiles) == 1):
        tile, tl, units = 0, t, attn_units(s)
    else:
        tile = t // TILE
        tl = t % TILE
        units = (tile < len(tiles) - 1).select(
            fx.Int32(attn_units(TILE)), fx.Int32(attn_units(tiles[-1][1]))
        )
    wt = col // 16
    f0 = tl * TILE_WORDS + (col % 16) // 2
    fs = [f0 + e for e in range(nw)]
    specs = []
    for src in range_constexpr(tp):
        for e in range_constexpr(nw):
            specs.append(
                (own, attn_unit(src, tile, units, wt, fs[e] // UNIT_WORDS, tp))
            )
    vs = c["poll_units"](specs)
    return [
        [_pick3(vs[src * nw + e], fs[e] % UNIT_WORDS) for e in range(nw)]
        for src in range(tp)
    ]


def fma(a, b, c):
    return fx.Float32(fmath.fma(a, b, c))


def sigmoid(z):
    """Triton's ``tl.sigmoid``: 1 / (1 + exp2(-z log2(e)))."""
    return fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(z * fx.Float32(-LOG2E)))


def lane_f32(v, src):
    return lane_gather(v.bitcast(fx.Int32), src).bitcast(fx.Float32)


def pend_value(c, t, col, cc):
    """The owed post's sublayer output at (token t, column col): read from
    ``pend``, or where a caller staged it (``pend_lds``, ``[token][32]`` of this
    task's columns: a reduction it did first)."""
    if const_expr(c.get("pend_lds") is None):
        return ld_bf(c["args"]["pend"], t * HIDDEN + col)
    return fx.ptr_load(c["pend_lds"] + (t * COLS + cc))


# ---------------------------------------------------------------- slice
@traced
def slice_input(c, task, idx):
    """Thread ``idx``'s pair of the layer input (token idx / 16, columns
    32 task + 2 (idx % 16) ..)."""
    s, rl = c["S"], c["rl"]
    a = c["args"]
    c0 = task * COLS
    if idx < s * COLS // 2:
        t = idx // (COLS // 2)
        cc = 2 * (idx % (COLS // 2))
        outs = []
        for d in range_constexpr(2):
            acc = fx.Float32(0.0)
            for h in range_constexpr(HC):
                rb = fx.ptr_load(rl + (t * c["rl_row"] + h * COLS + cc + d))
                acc = acc + ld_f32(a["pre_in"], t * HC + h) * rb
            outs.append(acc)
        c["put_bf"](c["lin"], t * HIDDEN + c0 + cc, outs)


def slice_at(c0, e):
    """Element e of a slice task at column c0: (token, stream, hidden column)."""
    return e // KT, (e % KT) // COLS, c0 + e % COLS


def slice_loads(c, c0, e):
    """Element e's loads of ``stage_slice`` (e clamped into the step's)."""
    a = c["args"]
    t, o, col = slice_at(c0, e)
    if const_expr(not c.get("fold", True)):
        return (ld_bf(a["res_in"], (t * HC + o) * HIDDEN + col),)
    r = [ld_bf(a["res_in"], (t * HC + h) * HIDDEN + col) for h in range(HC)]
    cm = [ld_f32(a["comb_in"], (t * HC + h) * HC + o) for h in range(HC)]
    return r, cm, ld_f32(a["post_in"], t * HC + o)


def slice_row(c, c0, e, got):
    """Element e of ``stage_slice``'s R_new from ``slice_loads``' values."""
    a = c["args"]
    t, o, col = slice_at(c0, e)
    if const_expr(not c.get("fold", True)):
        # a settled residual (no owed post): stream o as it arrives
        v = got[0]
    else:
        r, cm, post = got
        # aiter's mhc_fused_post_pre_delayed_rmsnorm: ((post x + c0 r0)
        # + c1 r1) + c2 r2, each product rounded, then fma(c3, r3, ..)
        v = post * pend_value(c, t, col, e % COLS) + (r[0] * cm[0])
        for h in range_constexpr(1, HC - 1):
            v = v + cm[h] * r[h]
        # bf16 R_new is what the collapse, the sums of squares and the
        # mix projection all read
        v = bf16_round(fma(cm[HC - 1], r[HC - 1], v))
    bo.buffer_store(v.to(fx.BFloat16), rsrc(a["res_out"]), (t * HC + o) * HIDDEN + col)
    fx.ptr_store(v, c["rl"] + (t * c["rl_row"] + o * COLS + e % COLS))


def slice_fn_loads(c, c0):
    """``stage_slice``'s mix weights of this thread (a load an element of
    MIX x KT, past them the last)."""
    tid, a = c["tid"], c["args"]
    out = []
    for i in range((MIX * KT + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, MIX * KT - 1)
        j = e // KT
        o = (e % KT) // COLS
        out.append(ld_f32(a["hc_fn"], j * K_MIX + o * HIDDEN + c0 + e % COLS))
    return out


@traced
def stage_slice(c, task):
    """R_new, the layer input and the partial projections of hidden columns
    32 task .. +32, all four streams, every token."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rl, fl = c["rl"], c["fl"]
    a = c["args"]
    c0 = task * COLS
    batched_rounds(
        tid,
        s * KT,
        lambda e: slice_loads(c, c0, e),
        lambda e, got: slice_row(c, c0, e, got),
    )
    if const_expr(c["tuned"]):
        # the mix weights' loads together, past the rows'
        fns = slice_fn_loads(c, c0)
    for i in range_constexpr((MIX * KT + THREADS - 1) // THREADS):
        e = tid + THREADS * i
        if e < MIX * KT:
            # the original's two bf16 MFMAs take fn as hi = bf16(fn) and lo =
            # bf16(fn - hi): their sum, exact in fp32, is the weight it applies
            if const_expr(c["tuned"]):
                f = fns[i]
            else:
                j = e // KT
                o = (e % KT) // COLS
                f = ld_f32(a["hc_fn"], j * K_MIX + o * HIDDEN + c0 + e % COLS)
            hi = bf16_round(f)
            fx.ptr_store(hi + bf16_round(f - hi), fl + (fl_at(e) if c["tuned"] else e))
    gpu.barrier()
    # the layer input: bf16 R_new of the four streams by the incoming pre-mix,
    # products rounded, summed in order (the Triton collapse, no contraction);
    # two columns a thread, one bf16 pair a mailbox store
    for pas in range_constexpr(0, s * COLS // 2, THREADS):
        slice_input(c, task, plus(pas, tid))
    # sum of squares: token t on wave t % WAVES, two columns a lane
    for t in range_constexpr(s):
        if wave == t % WAVES:
            x0 = fx.ptr_load(rl + (t * c["rl_row"] + 2 * lane))
            x1 = fx.ptr_load(rl + (t * c["rl_row"] + 2 * lane + 1))
            sq = butterfly(x0 * x0 + x1 * x1, (32, 16, 8, 4, 2, 1))
            if lane == 0:
                c["put"](c["pmix"], (task * s + t) * PART + MIX, sq)
    # mixes: fp32 MFMA 16x16x4, A = fn rows (two 16-row tiles), B = R_new^T,
    # a token tile at a time
    for t0, n in token_tiles(s):
        slice_mix(c, task, t0, n)


@traced
def slice_mix(c, task, t0, n):
    """The mix projection partials of tokens t0 .. t0 + n (``stage_slice``)."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rl, fl, red = c["rl"], c["fl"], c["red"]
    tok = plus(t0, fx.min(lane % 16, n - 1))
    kw = KT // WAVES
    cs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)]
    for kk in range_constexpr(kw // 4):
        colk = wave * kw + lane // 16 + 4 * kk
        b = fx.ptr_load(rl + (tok * c["rl_row"] + colk))
        for h in range_constexpr(2):
            row = h * 16 + lane % 16
            av = fx.ptr_load(fl + (fx.min(row, MIX - 1) * c["fl_row"] + colk))
            av = (row < MIX).select(av, fx.Float32(0.0))
            ops = rocdl._split_mfma_operands([av, b, cs[h], 0, 0, 0])
            cs[h] = fx.Vector(rocdl.mfma_f32_16x16x4f32(T.vec(4, T.f32), *ops).result)
    for h in range_constexpr(2):
        fx.ptr_store(cs[h], red + ((wave * 2 + h) * 64 + lane) * 4)
    gpu.barrier()
    if tid < MIX * n:
        tl = tid // MIX
        t = plus(t0, tl)
        j = tid % MIX
        h = j // 16
        ln = 16 * ((j % 16) // 4) + tl
        tot = fx.Float32(0.0)
        for w in range_constexpr(WAVES):
            tot = tot + fx.ptr_load(red + (((w * 2 + h) * 64 + ln) * 4 + j % 4))
        c["put"](c["pmix"], (task * s + t) * PART + j, tot)
    gpu.barrier()


# ---------------------------------------------------------------- reduce + slice
def fn_loads(c, task):
    """``stage_slice``'s fn weights a thread stores to ``fl`` (``fn_store``)."""
    tid = c["tid"]
    a = c["args"]
    c0 = task * COLS
    out = []
    for i in range(cdiv(MIX * KT, THREADS)):
        e = fx.min(tid + THREADS * i, MIX * KT - 1)
        j = e // KT
        o = (e % KT) // COLS
        out.append(ld_f32(a["hc_fn"], j * K_MIX + o * HIDDEN + c0 + e % COLS))
    return out


@traced
def fn_store(c, fs):
    """``fn_loads``' weights as the original's two bf16 MFMAs take them: hi =
    bf16(fn), lo = bf16(fn - hi), their sum exact in fp32 -> ``fl``."""
    tid = c["tid"]
    for i in range_constexpr(len(fs)):
        e = tid + THREADS * i
        if e < MIX * KT:
            f = fs[i]
            hi = bf16_round(f)
            fx.ptr_store(hi + bf16_round(f - hi), c["fl"] + fl_at(e))


QUAD = 4  # columns a thread of ``stage_reduce_slice``


@traced
def stage_reduce_slice(c, task, reduce=True):
    """``stage_reduce`` then ``stage_slice`` (fold) of the FFN seam, the same
    values (tuned builds; ``reduce`` False: K1's seam, its pend row instead of
    the TP partials). A thread takes four columns of a token through it all: the TP
    partials summed in rank order, R_new of the four streams, and the layer
    input -- from registers, no LDS hand-off, so the layer input leaves as soon
    as the partials land. Every operand that does not depend on them (the
    residual, comb / post / pre, the fn weights) is loaded before their poll."""
    s, tid = c["S"], c["tid"]
    a = c["args"]
    c0 = task * COLS
    n = s * COLS // QUAD
    assert n <= THREADS
    fs = fn_loads(c, task)
    idx = fx.min(tid, n - 1)
    t = idx // (COLS // QUAD)
    col = c0 + QUAD * (idx % (COLS // QUAD))
    rw = [
        fx.Vector(
            bo.buffer_load(
                rsrc(a["res_in"]),
                ((t * HC + h) * HIDDEN + col) // 2,
                vec_width=2,
                dtype=T.i32,
            )
        )
        for h in range(HC)
    ]
    cmv = [
        fx.Vector(
            bo.buffer_load(
                rsrc(a["comb_in"]), (t * HC + h) * HC, vec_width=4, dtype=T.f32
            )
        )
        for h in range(HC)
    ]
    postv = fx.Vector(
        bo.buffer_load(rsrc(a["post_in"]), t * HC, vec_width=4, dtype=T.f32)
    )
    prev = fx.Vector(
        bo.buffer_load(rsrc(a["pre_in"]), t * HC, vec_width=4, dtype=T.f32)
    )
    if const_expr(reduce):
        if tid < n:
            own = preg(c["sym"], 0, "attn")
            ws = attn_words(c, own, t, col, 2)
            reduce_slice_quad(c, task, t, col, ws, rw, cmv, postv, prev)
    else:
        # K1: the owed post's sublayer output is the caller's ``pend`` row
        pw = fx.Vector(
            bo.buffer_load(
                rsrc(a["pend"]), (t * HIDDEN + col) // 2, vec_width=2, dtype=T.i32
            )
        )
        if tid < n:
            reduce_slice_quad(c, task, t, col, None, rw, cmv, postv, prev, pw)
    fn_store(c, fs)
    gpu.barrier()
    slice_rest(c, task)


@traced
def reduce_slice_quad(c, task, t, col, ws, rw, cmv, postv, prev, pw=None):
    """A thread's (token t, columns col ..  col + 3) of ``stage_reduce_slice``
    (``ws``: the TP partials' words; None: ``pw``, the pend row's)."""
    rl = c["rl"]
    a = c["args"]
    pend = []
    if const_expr(ws is None):
        for p in range_constexpr(2):
            lo, hi = bf2_f32(fx.Int32(pw[p]))
            pend += [lo, hi]
    else:
        for p in range_constexpr(2):
            acc0, acc1 = bf2_f32(ws[0][p])
            for src in range_constexpr(1, len(ws)):
                lo, hi = bf2_f32(ws[src][p])
                acc0 = acc0 + lo
                acc1 = acc1 + hi
            pend += [bf16_round(acc0), bf16_round(acc1)]
    cc = col - task * COLS
    r = []
    for h in range_constexpr(HC):
        lo0, hi0 = bf2_f32(fx.Int32(rw[h][0]))
        lo1, hi1 = bf2_f32(fx.Int32(rw[h][1]))
        r.append([lo0, hi0, lo1, hi1])
    rows = []
    for o in range_constexpr(HC):
        cm = [fx.Float32(cmv[h][o]) for h in range(HC)]
        post = fx.Float32(postv[o])
        vs = []
        for k in range_constexpr(QUAD):
            v = post * pend[k] + (r[0][k] * cm[0])
            for h in range_constexpr(1, HC - 1):
                v = v + cm[h] * r[h][k]
            v = bf16_round(fma(cm[HC - 1], r[HC - 1][k], v))
            fx.ptr_store(v, rl + (t * c["rl_row"] + o * COLS + cc + k))
            vs.append(v)
        bo.buffer_store(
            fx.Vector.from_elements(
                [
                    bf16_pair(vs[0], vs[1]).bitcast(fx.Int32),
                    bf16_pair(vs[2], vs[3]).bitcast(fx.Int32),
                ],
                fx.Int32,
            ),
            rsrc(a["res_out"]),
            ((t * HC + o) * HIDDEN + col) // 2,
        )
        rows.append(vs)
    outs = []
    for k in range_constexpr(QUAD):
        acc = fx.Float32(0.0)
        for h in range_constexpr(HC):
            acc = acc + fx.Float32(prev[h]) * rows[h][k]
        outs.append(acc)
    c["put_bf"](c["lin"], t * HIDDEN + col, outs)


@traced
def slice_rest(c, task):
    """``stage_slice`` past the layer input: the sums of squares and the mix
    projection partials (``fl`` filled, ``rl`` complete)."""
    s, lane, wave = c["S"], c["lane"], c["wave"]
    rl = c["rl"]
    # sum of squares: token t on wave t % WAVES, two columns a lane -- a wave's
    # tokens side by side (independent reductions, interleaved), not one
    # branch a token
    ts = [wave + WAVES * j for j in range(cdiv(s, WAVES))]
    xs = []
    for j in range_constexpr(len(ts)):
        tc = fx.min(ts[j], s - 1)
        x0 = fx.ptr_load(rl + (tc * c["rl_row"] + 2 * lane))
        x1 = fx.ptr_load(rl + (tc * c["rl_row"] + 2 * lane + 1))
        xs.append(x0 * x0 + x1 * x1)
    for off in (32, 16, 8, 4, 2, 1):
        xs = [xred(v, off, operator.add) for v in xs]
    for j in range_constexpr(len(ts)):
        if (lane == 0) & (ts[j] < s):
            c["put"](c["pmix"], (task * s + ts[j]) * PART + MIX, xs[j])
    slice_mix_all(c, task)


RED_TILE = WAVES * 2 * 64 * 4  # ``slice_mix``'s MFMA partials of a token tile


def red_words(s, tuned=False):
    """The seam's LDS ``red``: ``slice_mix``'s partials of a token tile, or
    (``tuned``) ``slice_mix_all``'s of every token tile."""
    return RED_TILE * (len(token_tiles(s)) if tuned else 1)


@traced
def slice_mix_all(c, task):
    """``slice_mix`` of every token tile, the same sums: all tiles' MFMAs, one
    barrier, then all tiles' wave sums (two barriers in all, not two a tile)."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rl, fl, red = c["rl"], c["fl"], c["red"]
    kw = KT // WAVES
    tiles = token_tiles(s)
    for ti in range_constexpr(len(tiles)):
        t0, n = tiles[ti]
        tok = plus(t0, fx.min(lane % 16, n - 1))
        cs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)]
        for kk in range_constexpr(kw // 4):
            colk = wave * kw + lane // 16 + 4 * kk
            b = fx.ptr_load(rl + (tok * c["rl_row"] + colk))
            for h in range_constexpr(2):
                row = h * 16 + lane % 16
                av = fx.ptr_load(fl + (fx.min(row, MIX - 1) * c["fl_row"] + colk))
                av = (row < MIX).select(av, fx.Float32(0.0))
                ops = rocdl._split_mfma_operands([av, b, cs[h], 0, 0, 0])
                cs[h] = fx.Vector(
                    rocdl.mfma_f32_16x16x4f32(T.vec(4, T.f32), *ops).result
                )
        for h in range_constexpr(2):
            fx.ptr_store(
                cs[h], red + (ti * RED_TILE + ((wave * 2 + h) * 64 + lane) * 4)
            )
    gpu.barrier()
    total = MIX * s
    for pas in range_constexpr(0, total, THREADS):
        u = pas + tid
        if u < total:
            ti = u // (MIX * 16)
            loc = u % (MIX * 16)
            tl = loc // MIX
            j = loc % MIX
            t = ti * 16 + tl
            h = j // 16
            ln = 16 * ((j % 16) // 4) + tl
            tot = fx.Float32(0.0)
            for w in range_constexpr(WAVES):
                tot = tot + fx.ptr_load(
                    red + (ti * RED_TILE + ((w * 2 + h) * 64 + ln) * 4 + j % 4)
                )
            c["put"](c["pmix"], (task * s + t) * PART + j, tot)
    gpu.barrier()


# ---------------------------------------------------------------- gate
def _row(v, lane, q):
    """Comb lanes 8..23 hold ``comb[r][c]`` at lane 8 + 4 r + c: element q of
    the row."""
    return lane_f32(v, (lane - 8) // 4 * 4 + 8 + q)


def _col(v, lane, q):
    return lane_f32(v, 8 + 4 * q + (lane - 8) % 4)


@traced
def stage_gate(c, t):
    """Token t's 160 slices, in slice order, -> post / comb / next pre."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    red = c["red"]
    a = c["args"]
    groups = SLICES // GATE_BLOCK  # 20
    if tid < groups * PART:
        g = tid // PART
        j = tid % PART
        vals = c["poll"](
            [
                (c["pmix"], ((g * GATE_BLOCK + k) * s + t) * PART + j, 1)
                for k in range(GATE_BLOCK)
            ]
        )
        acc = vals[0][0].bitcast(fx.Float32)
        for k in range_constexpr(1, GATE_BLOCK):
            acc = acc + vals[k][0].bitcast(fx.Float32)
        fx.ptr_store(acc, red + (g * PART + j))
    gpu.barrier()
    if wave == 0:
        j = fx.min(lane, MIX - 1)
        mix = fx.ptr_load(red + j)
        sq = fx.ptr_load(red + MIX)
        for g in range_constexpr(1, groups):
            mix = mix + fx.ptr_load(red + (g * PART + j))
            sq = sq + fx.ptr_load(red + (g * PART + MIX))
        sc = [ld_f32(a["hc_scale"], i) for i in range(3)]
        base = ld_f32(a["hc_base"], j)
        # aiter's mhc_fused_post_pre_delayed_rmsnorm reduce: one rstd =
        # rsq(fma(sum, 1/K, eps)), then each gate fma(mix rstd, scale, base)
        v = mix * hw_rsq(fma(sq, fx.Float32(1.0 / K_MIX), fx.Float32(EPS)))
        if lane < HC:
            pre = sigmoid(fma(v, sc[0], base))
            bo.buffer_store(pre + fx.Float32(HC_EPS), rsrc(a["pre_out"]), t * HC + lane)
        if (lane >= HC) & (lane < 2 * HC):
            post = sigmoid(fma(v, sc[1], base))
            bo.buffer_store(
                post * fx.Float32(POST_MULT), rsrc(a["post_out"]), t * HC + lane - HC
            )
        cv = fma(v, sc[2], base)
        row = [_row(cv, lane, q) for q in range(4)]
        # Triton's exp: exp2 of (x - max) log2(e)
        m = fx.max(fx.max(row[0], row[1]), fx.max(row[2], row[3]))
        cv = hw_exp2((cv - m) * fx.Float32(LOG2E))
        # fast_dividef is x rcp(y); its first "+ eps" contracts into the fma
        row = [_row(cv, lane, q) for q in range(4)]
        cv = fma(cv, hw_rcp((row[0] + row[1]) + (row[2] + row[3])), fx.Float32(HC_EPS))
        col = [_col(cv, lane, q) for q in range(4)]
        cv = cv * hw_rcp(((col[0] + col[1]) + (col[2] + col[3])) + fx.Float32(HC_EPS))
        for _ in range_constexpr(SINKHORN - 1):
            row = [_row(cv, lane, q) for q in range(4)]
            cv = cv * hw_rcp(
                ((row[0] + row[1]) + (row[2] + row[3])) + fx.Float32(HC_EPS)
            )
            col = [_col(cv, lane, q) for q in range(4)]
            cv = cv * hw_rcp(
                ((col[0] + col[1]) + (col[2] + col[3])) + fx.Float32(HC_EPS)
            )
        if (lane >= 2 * HC) & (lane < MIX):
            bo.buffer_store(cv, rsrc(a["comb_out"]), t * HC * HC + lane - 2 * HC)
    gpu.barrier()


# ---------------------------------------------------------------- norm
@traced
def stage_norm(c, t, store, tail_barrier=True):
    """Token t's RMSNorm (rstd = rsq(fma(sum, 1/H, eps)), bf16((x rstd) w));
    256 threads, 3 chunks of 8 at 8 t + 2048 c, each chunk's bf16 values handed
    to ``store(t, col, ys, live)`` by every thread (``live``: a thread's chunk
    in the row; threads 256.. mirror thread 255)."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    tt = fx.min(tid, 255)
    mine = tid < 256
    cols = [tt * 8 + 2048 * ch for ch in range(3)]
    xs = []
    if const_expr(c["tuned"]):
        # the norm weights in flight before the row's wait (they do not depend
        # on it), then the three chunks' pairs in one poll batch: one round trip
        wv = [
            fx.Vector(
                bo.buffer_load(
                    rsrc(a["norm_w"]),
                    fx.min(cols[ch], HIDDEN - 8) // 2,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            for ch in range(3)
        ]
        bases = [t * HIDDEN + fx.min(cols[ch], HIDDEN - 8) for ch in range(3)]
        words = c["poll"](
            [(c["lin"], bases[ch] // 2 + k, 1) for ch in range(3) for k in range(4)]
        )
        for ch in range_constexpr(3):
            v = []
            for k in range_constexpr(4):
                w = words[4 * ch + k][0]
                v += [bf_lo(w), bf_hi(w)]
            xs.append([(cols[ch] < HIDDEN).select(x, fx.Float32(0.0)) for x in v])
    else:
        for ch in range_constexpr(3):
            base = t * HIDDEN + fx.min(cols[ch], HIDDEN - 8)
            words = c["poll"]([(c["lin"], base // 2 + k, 1) for k in range(4)])
            v = []
            for k in range_constexpr(4):
                v += [bf_lo(words[k][0]), bf_hi(words[k][0])]
            xs.append([(cols[ch] < HIDDEN).select(x, fx.Float32(0.0)) for x in v])
    acc = fx.Float32(0.0)
    for ch in range_constexpr(3):
        for x in xs[ch]:
            acc = acc + x * x
    acc = butterfly(acc, (1, 2, 4, 8, 16, 32))
    if (lane == 0) & mine:
        fx.ptr_store(acc, red + wave)
    gpu.barrier()
    tot = butterfly(fx.ptr_load(red + lane % 4), (1, 2))
    r = hw_rsq(fma(tot, fx.Float32(1.0 / HIDDEN), fx.Float32(EPS)))
    for ch in range_constexpr(3):
        col = fx.min(cols[ch], HIDDEN - 8)
        if const_expr(c["tuned"]):
            nw = []
            for k in range_constexpr(4):
                nw += [bf_lo(fx.Int32(wv[ch][k])), bf_hi(fx.Int32(wv[ch][k]))]
            ys = [bf16_round((xs[ch][j] * r) * nw[j]) for j in range(8)]
        else:
            ys = [
                bf16_round((xs[ch][j] * r) * ld_bf(a["norm_w"], col + j))
                for j in range(8)
            ]
        store(t, col, ys, mine & (cols[ch] < HIDDEN))
    if const_expr(tail_barrier):
        # (a caller that publishes next has the drain's barrier instead)
        gpu.barrier()


# ---------------------------------------------------------------- reduce
@traced
def stage_reduce(c, task):
    """The attention output at this slice's 32 columns: the TP ranks' partials
    in the all-reduce's order (``sum_partials``), in fp32 -> bf16 ->
    ``pend_lds``."""
    s, tid = c["S"], c["tid"]
    for pas in range_constexpr(0, s * COLS // 2, THREADS):
        reduce_attn_pair(c, task, plus(pas, tid))
    gpu.barrier()


@traced
def reduce_attn_pair(c, task, idx):
    """Thread ``idx``'s column pair of token idx / 16 (``stage_reduce``)."""
    s = c["S"]
    c0 = task * COLS
    if idx < s * COLS // 2:
        t = idx // (COLS // 2)
        col = c0 + 2 * (idx % (COLS // 2))
        own = preg(c["sym"], 0, "attn")
        acc0, acc1 = sum_partials(
            c["poll"],
            own,
            lambda src: attn_region_pair(src, t, s, col),
            c["d"].tp,
        )
        lds = c["pend_lds"] + (t * COLS + col - c0)
        fx.ptr_store(bf16_round(acc0), lds)
        fx.ptr_store(bf16_round(acc1), lds + 1)


@traced
def stage_push(c, task, part):
    """This rank's attention partial (``part``: bf16 [S, HIDDEN], wo_b's
    unreduced output) at this slice's 32 columns, every token, to every rank's
    ATTN region: the pairs ``stage_reduce`` polls, as the bits vLLM wrote."""
    s, tid, tp = c["S"], c["tid"], c["d"].tp
    for pas in range_constexpr(0, s * COLS // 2, THREADS):
        idx = plus(pas, tid)
        if idx < s * COLS // 2:
            t = idx // (COLS // 2)
            col = task * COLS + 2 * (idx % (COLS // 2))
            w = bo.buffer_load(
                rsrc(part), (t * HIDDEN + col) // 2, vec_width=1, dtype=T.i32
            )
            pair = attn_region_pair(c["rank"], t, s, col)
            for p in range_constexpr(tp):
                c["put_words"](preg(c["peer_addr"](p), 0, "attn"), pair, [w])
