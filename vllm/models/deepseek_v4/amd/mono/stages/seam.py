# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/models/deepseek_v41/mono/kernels/attn_pre.py (slice, gate, norm) and
# atom/models/deepseek_v41/mono/kernels/attn_post.py (the attention's TP reduce).
"""The mono layer's mHC seam stages, each a task table of a persistent launch
handing off through the tagged mailbox:

    slice   (224, 32 hidden columns x 4 streams each): the owed post folded into
            the residual (R_new, bf16 out at device scope) and the partial mix
            projection / sum of squares (mailbox PMIX)
    gate    (S): the 224 partials -> post / comb (Sinkhorn) and the pre-mix,
            also to the mailbox PREM
    norm    (S): PREM, then R_new's four streams -> the collapse -> the RMSNorm
            -> the bf16 row, handed to the caller's store
    reduce  (224): the FFN seam's attention output, the TP ranks' partials
            (pushed to the ATTN region) summed in the all-reduce's order ->
            ``pend_lds``, where its slice reads it

The collapse takes the gate's own pre-mix (aiter's ``mhc_pre``: the four streams
by pre in fp32, then bf16). Every other elementwise step follows aiter's
``mhc_fused_post_pre_delayed_rmsnorm`` rounding; the mix projection and the sums
of squares, split-K reductions, differ only in their split and order.
"""

import math

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from vllm.models.deepseek_v4.amd.mono.common.ops import (
    CM_DEV,
    batched_rounds,
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    div_rn,
    hw_exp2,
    hw_rcp,
    hw_rsq,
    lane_gather,
    rsrc,
    traced,
)
from vllm.models.deepseek_v4.amd.mono.common.plan import (
    THREADS,
    WAVES,
    cdiv,
    pair_layout,
)
from vllm.models.deepseek_v4.amd.mono.common.sync import preg
from vllm.models.deepseek_v4.amd.mono.dims import (
    HC,
    HC_EPS,
    HC_POST_MULT,
    HC_SINKHORN,
    HIDDEN,
    NORM_EPS,
    Dims,
)
from vllm.models.deepseek_v4.amd.mono.stages import allreduce
from vllm.models.deepseek_v4.amd.mono.stages.gemv import (
    ld_bf,
    ld_f32,
    plus,
    token_tiles,
)

MIX = HC * (HC + 2)  # 24 projections: pre 4 | post 4 | comb 16
LOG2E = 1.0 / math.log(2.0)

COLS = 32  # hidden columns a slice task
SLICES = HIDDEN // COLS  # 224
KT = HC * COLS  # projection columns a slice task
PART = MIX + 1  # a slice's partials a token: the mixes, then the sum of squares
# slices a gate thread sums: its groups' partials take at most THREADS threads
GATE_BLOCK = 14
NORM_CHUNK = 2048  # a norm pass's columns: 256 threads x 8
assert SLICES % GATE_BLOCK == 0 and SLICES // GATE_BLOCK * PART <= THREADS
# the seam's mailbox regions
REGIONS = ("pmix", "prem")


def scratch_layout(tokens: int) -> dict[str, tuple[int, int]]:
    """Mailbox region -> (byte offset, bytes) of a seam; 8 B a pair."""
    pairs = {
        "pmix": SLICES * tokens * PART,
        "prem": tokens * HC,  # the gate's pre-mix, the norm's collapse
    }
    return pair_layout(pairs.items())


def attn_peer_bytes(tokens: int, d: Dims) -> int:
    """ATTN: the attention output's all-reduce (``allreduce``)."""
    return allreduce.region_bytes(tokens, d)


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
def slice_at(c0, e):
    """Element e of a slice task at column c0: (token, stream, hidden column)."""
    return e // KT, (e % KT) // COLS, c0 + e % COLS


def slice_loads(c, c0, e):
    """Element e's loads of ``stage_slice`` (e clamped into the step's)."""
    a, hidden = c["args"], HIDDEN
    t, o, col = slice_at(c0, e)
    if const_expr(not c.get("fold", True)):
        return (ld_bf(a["res_in"], (t * HC + o) * hidden + col),)
    r = [ld_bf(a["res_in"], (t * HC + h) * hidden + col) for h in range(HC)]
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
    # the norms' collapse reads R_new back from res_out in other CTAs: device
    # scope
    bo.buffer_store(
        v.to(fx.BFloat16),
        rsrc(a["res_out"]),
        (t * HC + o) * HIDDEN + col,
        cache_modifier=CM_DEV,
    )
    fx.ptr_store(v, c["rl"] + (t * KT + o * COLS + e % COLS))


@traced
def stage_slice(c, task):
    """R_new and the partial projections of hidden columns 32 task .. +32, all
    four streams, every token."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rl, fl = c["rl"], c["fl"]
    a, hidden = c["args"], HIDDEN
    c0 = task * COLS
    batched_rounds(
        tid,
        s * KT,
        lambda e: slice_loads(c, c0, e),
        lambda e, got: slice_row(c, c0, e, got),
    )
    for i in range_constexpr((MIX * KT + THREADS - 1) // THREADS):
        e = tid + THREADS * i
        if e < MIX * KT:
            j = e // KT
            o = (e % KT) // COLS
            # the original's two bf16 MFMAs take fn as hi = bf16(fn) and lo =
            # bf16(fn - hi): their sum, exact in fp32, is the weight it applies
            f = ld_f32(a["hc_fn"], j * HC * hidden + o * hidden + c0 + e % COLS)
            hi = bf16_round(f)
            fx.ptr_store(hi + bf16_round(f - hi), fl + e)
    # R_new landed device-wide before any partial announces it
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    # sum of squares: token t on wave t % WAVES, two columns a lane
    for t in range_constexpr(s):
        if wave == t % WAVES:
            x0 = fx.ptr_load(rl + (t * KT + 2 * lane))
            x1 = fx.ptr_load(rl + (t * KT + 2 * lane + 1))
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
        b = fx.ptr_load(rl + (tok * KT + colk))
        for h in range_constexpr(2):
            row = h * 16 + lane % 16
            av = fx.ptr_load(fl + (fx.min(row, MIX - 1) * KT + colk))
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


# ---------------------------------------------------------------- gate
def _row(v, lane, q):
    """Comb lanes 8..23 hold ``comb[r][c]`` at lane 8 + 4 r + c: element q of
    the row."""
    return lane_f32(v, (lane - 8) // 4 * 4 + 8 + q)


def _col(v, lane, q):
    return lane_f32(v, 8 + 4 * q + (lane - 8) % 4)


@traced
def stage_gate(c, t):
    """Token t's 224 slices, in slice order, -> post / comb / pre."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    red = c["red"]
    a = c["args"]
    blk = GATE_BLOCK
    groups = SLICES // blk  # 16
    if tid < groups * PART:
        g = tid // PART
        j = tid % PART
        vals = c["poll"](
            [(c["pmix"], ((g * blk + k) * s + t) * PART + j, 1) for k in range(blk)]
        )
        acc = vals[0][0].bitcast(fx.Float32)
        for k in range_constexpr(1, blk):
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
        v = mix * hw_rsq(fma(sq, fx.Float32(1.0 / (HC * HIDDEN)), fx.Float32(NORM_EPS)))
        if lane < HC:
            pre = sigmoid(fma(v, sc[0], base)) + fx.Float32(HC_EPS)
            bo.buffer_store(pre, rsrc(a["pre_out"]), t * HC + lane)
            c["put"](c["prem"], t * HC + lane, pre)
        if (lane >= HC) & (lane < 2 * HC):
            post = sigmoid(fma(v, sc[1], base))
            bo.buffer_store(
                post * fx.Float32(HC_POST_MULT),
                rsrc(a["post_out"]),
                t * HC + lane - HC,
            )
        cv = fma(v, sc[2], base)
        row = [_row(cv, lane, q) for q in range(4)]
        # Triton's exp: exp2 of (x - max) log2(e)
        m = fx.max(fx.max(row[0], row[1]), fx.max(row[2], row[3]))
        cv = hw_exp2((cv - m) * fx.Float32(LOG2E))
        # fast_dividef is x rcp(y); its first "+ eps" contracts into the fma
        row = [_row(cv, lane, q) for q in range(4)]
        eps = fx.Float32(HC_EPS)
        cv = fma(cv, hw_rcp((row[0] + row[1]) + (row[2] + row[3])), eps)
        col = [_col(cv, lane, q) for q in range(4)]
        cv = cv * hw_rcp(((col[0] + col[1]) + (col[2] + col[3])) + eps)
        for _ in range_constexpr(HC_SINKHORN - 1):
            row = [_row(cv, lane, q) for q in range(4)]
            cv = cv * hw_rcp(((row[0] + row[1]) + (row[2] + row[3])) + eps)
            col = [_col(cv, lane, q) for q in range(4)]
            cv = cv * hw_rcp(((col[0] + col[1]) + (col[2] + col[3])) + eps)
        if (lane >= 2 * HC) & (lane < MIX):
            bo.buffer_store(cv, rsrc(a["comb_out"]), t * HC * HC + lane - 2 * HC)
    gpu.barrier()


# ---------------------------------------------------------------- norm
@traced
def stage_norm(c, t, store):
    """Token t's input collapse (``collapse_chunk``) and its RMSNorm, aiter's
    ``mhc_pre``: rstd = rsq(sum / H + eps) of the fp32 collapse, bf16((x w)
    rstd); 256 threads, chunks of 8 at 8 t + 2048 c, each chunk's bf16 values
    handed to ``store(t, col, ys, live)`` by every thread (``live``: a thread's
    chunk in the row; threads 256.. mirror thread 255)."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    a, hidden = c["args"], HIDDEN
    nch = cdiv(hidden, NORM_CHUNK)
    tt = fx.min(tid, 255)
    mine = tid < 256
    cols = [tt * 8 + NORM_CHUNK * ch for ch in range(nch)]
    xs = []
    acc = fx.Float32(0.0)
    got = c["poll"]([(c["prem"], t * HC + h, 1) for h in range(HC)])
    pre = [got[h][0].bitcast(fx.Float32) for h in range(HC)]
    for ch in range_constexpr(nch):
        col = fx.min(cols[ch], hidden - 8)
        xf = collapse_chunk(c, t, col, pre)
        xf = [(cols[ch] < hidden).select(x, fx.Float32(0.0)) for x in xf]
        for x in xf:
            acc = acc + x * x
        xs.append([bf16_round(x) for x in xf])
    acc = butterfly(acc, (1, 2, 4, 8, 16, 32))
    if (lane == 0) & mine:
        fx.ptr_store(acc, red + wave)
    gpu.barrier()
    tot = butterfly(fx.ptr_load(red + lane % 4), (1, 2))
    mean = div_rn(tot, fx.Float32(float(hidden)), fx.Float32(1.0 / hidden))
    r = hw_rsq(mean + fx.Float32(NORM_EPS))
    for ch in range_constexpr(nch):
        col = fx.min(cols[ch], hidden - 8)
        ys = [
            bf16_round((xs[ch][j] * ld_bf(a["norm_w"], col + j)) * r) for j in range(8)
        ]
        store(t, col, ys, mine & (cols[ch] < hidden))
    gpu.barrier()


def collapse_chunk(c, t, col, pre):
    """Token t's layer input at columns col .. col + 8, fp32: R_new's four
    streams (res_out, written device-wide by this launch's slices, whose
    partials the gate waited on) by the pre-mix, summed pairwise as aiter's
    ``mhc_pre`` reduces across its stream lanes."""
    a, hidden = c["args"], HIDDEN
    rs = []
    for h in range(HC):
        w = fx.Vector(
            bo.buffer_load(
                rsrc(a["res_out"]),
                ((t * HC + h) * hidden + col) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        v = []
        for k in range(4):
            v += [bf_lo(w[k]), bf_hi(w[k])]
        rs.append(v)
    return [
        (pre[0] * rs[0][j] + pre[1] * rs[1][j])
        + (pre[2] * rs[2][j] + pre[3] * rs[3][j])
        for j in range(8)
    ]


# ---------------------------------------------------------------- reduce
@traced
def stage_reduce(c, task):
    """The attention output at this slice's 32 columns: the TP ranks' partials
    in the all-reduce's order (``sum_partials``), in fp32 -> bf16 ->
    ``pend_lds``."""
    s, tid = c["S"], c["tid"]
    if const_expr(c.get("two_hop", False)):
        # this slice's columns' owner sums them for every rank (``allreduce``)
        if allreduce.owner(c["d"], task * COLS) == c["rank"]:
            for pas in range_constexpr(0, s * COLS // 2, THREADS):
                reduce_attn_pair(c, task, plus(pas, tid), "owned")
        else:
            for pas in range_constexpr(0, s * COLS // 2, THREADS):
                reduce_attn_pair(c, task, plus(pas, tid), "gathered")
    else:
        for pas in range_constexpr(0, s * COLS // 2, THREADS):
            reduce_attn_pair(c, task, plus(pas, tid))
    gpu.barrier()


@traced
def reduce_attn_pair(c, task, idx, how="sum"):
    """Thread ``idx``'s column pair of token idx / 16 (``stage_reduce``;
    ``how``: the ``allreduce`` way, ``reduce_pair`` / ``_owned`` /
    ``_gathered``)."""
    s, d = c["S"], c["d"]
    c0 = task * COLS
    if idx < s * COLS // 2:
        t = idx // (COLS // 2)
        col = c0 + 2 * (idx % (COLS // 2))
        own = preg(c["sym"], 0, "attn")
        if const_expr(how == "owned"):
            acc0, acc1 = allreduce.reduce_owned(c, own, 0, "attn", t, s, col, d)
        elif const_expr(how == "gathered"):
            acc0, acc1 = allreduce.reduce_gathered(c, own, t, s, col, d)
        else:
            acc0, acc1 = allreduce.reduce_pair(c, own, t, s, col, d)
        lds = c["pend_lds"] + (t * COLS + col - c0)
        fx.ptr_store(bf16_round(acc0), lds)
        fx.ptr_store(bf16_round(acc1), lds + 1)
