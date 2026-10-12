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

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from vllm.models.deepseek_v41.amd.mono.common.ops import (
    batched_rounds,
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
)
from vllm.models.deepseek_v41.amd.mono.common.plan import THREADS, WAVES, pair_layout
from vllm.models.deepseek_v41.amd.mono.common.sync import preg
from vllm.models.deepseek_v41.amd.mono.stages.gemv import (
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
                rb = fx.ptr_load(rl + (t * KT + h * COLS + cc + d))
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
    r = [ld_bf(a["res_in"], (t * HC + h) * HIDDEN + col) for h in range(HC)]
    cm = [ld_f32(a["comb_in"], (t * HC + h) * HC + o) for h in range(HC)]
    return r, cm, ld_f32(a["post_in"], t * HC + o)


def slice_row(c, c0, e, got):
    """Element e of ``stage_slice``'s R_new from ``slice_loads``' values."""
    a = c["args"]
    t, o, col = slice_at(c0, e)
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
    fx.ptr_store(v, c["rl"] + (t * KT + o * COLS + e % COLS))


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
    for i in range_constexpr((MIX * KT + THREADS - 1) // THREADS):
        e = tid + THREADS * i
        if e < MIX * KT:
            j = e // KT
            o = (e % KT) // COLS
            # the original's two bf16 MFMAs take fn as hi = bf16(fn) and lo =
            # bf16(fn - hi): their sum, exact in fp32, is the weight it applies
            f = ld_f32(a["hc_fn"], j * K_MIX + o * HIDDEN + c0 + e % COLS)
            hi = bf16_round(f)
            fx.ptr_store(hi + bf16_round(f - hi), fl + e)
    gpu.barrier()
    # the layer input: bf16 R_new of the four streams by the incoming pre-mix,
    # products rounded, summed in order (the Triton collapse, no contraction);
    # two columns a thread, one bf16 pair a mailbox store
    for pas in range_constexpr(0, s * COLS // 2, THREADS):
        slice_input(c, task, plus(pas, tid))
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
def stage_norm(c, t, store):
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
        ys = [
            bf16_round((xs[ch][j] * r) * ld_bf(a["norm_w"], col + j)) for j in range(8)
        ]
        store(t, col, ys, mine & (cols[ch] < HIDDEN))
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
