# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 FFN launch: one persistent kernel from the unreduced ``o_proj``
partial to the reduced MoE output (``MiniMaxM3DecoderLayer.forward`` after its
attention).

``BLOCKS`` x ``THREADS``, S rows, one launch per sparse layer and rank:

    push    (192 x 32 columns): the partial's bf16 pairs to every rank
    reduce  (192): the ranks' pairs in rank order, f32 -> bf16; + residual
            in f32 -> residual out (bf16), the f32 sum, its sum of squares
    norm    (S): Gemma RMSNorm of the f32 sum -> X (bf16)
    router  (128 experts): FP32 weight x f32(X) -> logits
    route   (S): sigmoid, + bias for selection, top-4, renormalized x 2.0
    table   (every CTA): the union of picked experts in LDS, slot 0 the shared
            expert (expert 128) over all S rows
    up      (48 x slots): 16 gate + 16 up rows of a slot's expert, A16W4 MFMA,
            SwiGLU-OAI -> bf16 MID
    down    (384 x 16 rows): every slot's A16W4 GEMV of MID, routed weights
            and shared expert combined -> pushed to every rank
    final   (192): the ranks' partials in rank order -> ``out``
"""

from dataclasses import dataclass
from enum import Enum

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.deepseek_v41.amd.mono.common.ops import (
    CM_DEV,
    CM_SYS,
    ballot,
    bf2_f32,
    bf16_pair,
    bf16_round,
    butterfly,
    hw_exp2,
    hw_rcp,
    lanes_below,
    memrealtime,
    peer_bases,
    popcount,
    rsrc,
    traced,
)
from vllm.models.deepseek_v41.amd.mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    first_task,
    key_tuple,
    pair_layout,
)
from vllm.models.deepseek_v41.amd.mono.common.sync import preg, sreg

from .a16w4 import scale_f32, scale_loads, step_mfma, weight_loads
from .sources import SOURCES
from .sync import ERROR_WORDS, BoundedMailbox

_STREAM = fx.Stream(None)
HIDDEN = 6144
INTER = 768  # a TP4 rank's expert intermediate
EXPERTS = 128
SHARED = 128  # the fused shared expert's index in w13 / w2
TOPK = 4
ROUTED_SCALE = 2.0
EPS = 1e-6
ALPHA, BETA, LIMIT = 1.702, 1.0, 7.0
LOG2E = 1.4426950408889634
MAX_TOKENS = 16

COLS = 32  # columns of a push / reduce / final task
SEAM_TASKS = HIDDEN // COLS  # 192
UP_TILES = INTER // 16  # 48 gate/up tile pairs an expert
DOWN_TASKS = HIDDEN // 16  # 384
K1_STEPS = HIDDEN // 128
K2_STEPS = INTER // 128
UP_WAVE_STEPS = K1_STEPS // WAVES
assert UP_WAVE_STEPS * WAVES == K1_STEPS and UP_WAVE_STEPS % 2 == 0

# CTA bases of the stages placed by count
NORM0 = SEAM_TASKS
ROUTER0 = NORM0 + MAX_TOKENS
SHARED0 = (ROUTER0 + EXPERTS) % BLOCKS
ROUTED0 = SHARED0 + UP_TILES

ERROR_BYTES = 256  # the error word's line, ahead of the two parity halves

# sites of a timed-out wait (the error word's fourth field)
SITE_REDUCE, SITE_NORM, SITE_ROUTER, SITE_ROUTE = 1, 2, 3, 4
SITE_TABLE, SITE_UP, SITE_DOWN, SITE_FINAL, SITE_EPOCH = 5, 6, 7, 8, 9


class Experts(Enum):
    """Which expert terms the combine sums: all of them in production; one side
    alone for stage parity."""

    ALL = "all"
    SHARED = "shared"
    ROUTED = "routed"


@dataclass(frozen=True)
class FfnBuild:
    tokens: int
    tp: int = 4
    experts: Experts = Experts.ALL
    timeline: bool = False


def slots(s: int) -> int:
    return 1 + TOPK * s


def peer_half_bytes(tp: int) -> int:
    """One parity's peer regions: the attention partials, then the MoE's."""
    return 2 * tp * MAX_TOKENS * (HIDDEN // 2) * 8


def peer_bytes(tp: int) -> int:
    return ERROR_BYTES + 2 * peer_half_bytes(tp)


def scratch_layout(s: int) -> dict[str, tuple[int, int]]:
    tagged = pair_layout(
        [
            ("ssq", s * SEAM_TASKS),
            ("xrdy", s),
            ("logit", s * EXPERTS),
            ("route", s * 2 * TOPK),
            ("upf", slots(s) * UP_TILES),
        ],
        align=256,
    )
    off = -(-max(o + n for o, n in tagged.values()) // 256) * 256
    out = dict(tagged)
    for name, nbytes in (
        ("sres", s * HIDDEN * 4),
        ("xn", s * HIDDEN * 2),
        ("mid", slots(s) * 16 * INTER * 2),
        ("tl", BLOCKS * TL_POINTS * 8),
    ):
        out[name] = (off, nbytes)
        off += -(-nbytes // 256) * 256
    return out


def scratch_bytes(s: int) -> int:
    return max(o + n for o, n in scratch_layout(s).values())


EPOCH_MARKS = 4
TL_STAGES = (
    "push",
    "reduce",
    "norm",
    "router",
    "route",
    "shared",
    "table",
    "up",
    "down",
    "final",
    "epoch",
)
TL_POINTS = 1 + len(TL_STAGES)
EPOCH_WORDS = EPOCH_MARKS + BLOCKS

# LDS route table (int32 words)
T_PICK_E = 0
T_PICK_W = T_PICK_E + TOPK * MAX_TOKENS
T_MASK = T_PICK_W + TOPK * MAX_TOKENS
T_ESLOT = T_MASK + EXPERTS
T_SLOT_E = T_ESLOT + EXPERTS
T_SLOT_MASK = T_SLOT_E + 80
T_SLOT_TOK = T_SLOT_MASK + 80
T_SLOT_PICK = T_SLOT_TOK + 80 * 16
T_MISC = T_SLOT_PICK + 80 * 16
T_WORDS = T_MISC + 16
assert slots(MAX_TOKENS) <= 80


# ---------------------------------------------------------------- helpers
def ld_dev(base, i, n=1, dtype=None):
    dtype = T.i32 if dtype is None else dtype
    v = bo.buffer_load(rsrc(base), i, vec_width=n, dtype=dtype, cache_modifier=CM_DEV)
    return fx.Vector(v) if n > 1 else v


def st_dev(v, base, i):
    bo.buffer_store(v, rsrc(base), i, cache_modifier=CM_DEV)


def put_word(c, addr, i, word):
    """Pair i of ``addr`` := (word, tag) at its space's scope."""
    c["mb"].put_words(addr, i, [word])


@traced
def flag_after_stores(c, addr, i, who):
    """This CTA's plain stores drained, then pair i := 1 (thread ``who``)."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if who:
        put_word(c, addr, i, fx.Int32(1))


def attn_pair(c, src, t, colpair):
    return (src * MAX_TOKENS + t) * (HIDDEN // 2) + colpair


def ffn_pair(c, src, t, colpair):
    return (c["tp"] + src) * MAX_TOKENS * (HIDDEN // 2) + attn_pair(c, 0, t, colpair)


@traced
def block_sum(c, v, slot):
    """``v`` summed over the CTA: each wave's butterfly, then waves in order
    (LDS ``red`` words ``slot * WAVES ..``). Every thread gets the sum."""
    lane, wave, red = c["lane"], c["wave"], c["red"]
    v = butterfly(v, (32, 16, 8, 4, 2, 1))
    if lane == 0:
        fx.ptr_store(v, red + (slot * WAVES + wave))
    gpu.barrier()
    tot = fx.Float32(0.0)
    for w in range_constexpr(WAVES):
        tot = tot + fx.ptr_load(red + (slot * WAVES + w))
    return tot


# ---------------------------------------------------------------- seam
@traced
def stage_push(c, task):
    """Columns 32 task ..: this rank's partial pairs to every rank's ATTN."""
    s, tid, a = c["S"], c["tid"], c["args"]
    if tid < s * COLS // 2:
        t = tid // (COLS // 2)
        cp = task * (COLS // 2) + tid % (COLS // 2)
        word = fx.Int32(
            bo.buffer_load(
                rsrc(a["part"]), t * (HIDDEN // 2) + cp, vec_width=1, dtype=T.i32
            )
        )
        for p in range_constexpr(c["tp"]):
            put_word(c, c["peer"][p], attn_pair(c, c["rank"], t, cp), word)


@traced
def stage_reduce(c, task):
    """Columns 32 task ..: the ranks' pairs summed in rank order (the
    all-reduce's bf16), + residual in f32 -> residual out, the f32 sum (SRES)
    and the task's sum of squares (SSQ, a pair a token)."""
    s, tid, a = c["S"], c["tid"], c["args"]
    if tid < s * COLS // 2:
        t = tid // (COLS // 2)
        cp = task * (COLS // 2) + tid % (COLS // 2)
        got = c["mb"].poll(
            [(c["own"], attn_pair(c, src, t, cp), 1) for src in range(c["tp"])],
            site=SITE_REDUCE,
        )
        lo, hi = bf2_f32(got[0][0])
        for src in range_constexpr(1, c["tp"]):
            l2, h2 = bf2_f32(got[src][0])
            lo, hi = lo + l2, hi + h2
        rw = fx.Int32(
            bo.buffer_load(
                rsrc(a["res_in"]), t * (HIDDEN // 2) + cp, vec_width=1, dtype=T.i32
            )
        )
        rl, rh = bf2_f32(rw)
        sl = bf16_round(lo) + rl
        sh = bf16_round(hi) + rh
        bo.buffer_store(
            bf16_pair(sl, sh).bitcast(fx.Int32),
            rsrc(a["res_out"]),
            t * (HIDDEN // 2) + cp,
        )
        st_dev(
            fx.Vector.from_elements([sl, sh], fx.Float32),
            c["sres"].value,
            t * HIDDEN + 2 * cp,
        )
        sq = butterfly(sl * sl + sh * sh, (8, 4, 2, 1))
        rocdl.s_waitcnt(vmcnt=0)
        if tid % (COLS // 2) == 0:
            c["mb"].put(c["ssq"], t * SEAM_TASKS + task, sq)


@traced
def stage_norm(c, t):
    """Token t's Gemma RMSNorm of SRES -> XN, then XRDY."""
    tid, a = c["tid"], c["args"]
    part = fx.Float32(0.0)
    if tid < SEAM_TASKS:
        got = c["mb"].poll([(c["ssq"], t * SEAM_TASKS + tid, 1)], site=SITE_NORM)
        part = got[0][0].bitcast(fx.Float32)
    var = block_sum(c, part, 0) / fx.Float32(HIDDEN)
    rstd = fx.Float32(1.0) / fx.Float32(fmath.sqrt(var + fx.Float32(EPS)))
    per = HIDDEN // THREADS  # 12
    for q in range_constexpr(per // 2):
        col = (q * THREADS + tid) * 2
        sv = ld_dev(c["sres"].value, t * HIDDEN + col, 2, T.f32)
        ww = fx.Int32(
            bo.buffer_load(rsrc(a["norm_w"]), col // 2, vec_width=1, dtype=T.i32)
        )
        wl, wh = bf2_f32(ww)
        ol = sv[0] * rstd * (fx.Float32(1.0) + wl)
        oh = sv[1] * rstd * (fx.Float32(1.0) + wh)
        st_dev(
            bf16_pair(ol, oh).bitcast(fx.Int32),
            c["xn"].value,
            t * (HIDDEN // 2) + col // 2,
        )
    flag_after_stores(c, c["xrdy"], t, tid == 0)


@traced
def await_x(c, tokens_n):
    """XRDY of the first ``tokens_n`` tokens (thread t polls token t)."""
    tid = c["tid"]
    if tid < tokens_n:
        c["mb"].poll([(c["xrdy"], tid, 1)], site=SITE_ROUTER)
    gpu.barrier()


# ---------------------------------------------------------------- router
@traced
def stage_router(c, e):
    """Expert e's logits of every token: f32 weight row x f32(X), the CTA's
    partials summed per wave then in wave order -> LOGIT."""
    s, tid, a = c["S"], c["tid"], c["args"]
    wv = [
        fx.Vector(
            bo.buffer_load(
                rsrc(a["router_w"]),
                e * HIDDEN + j * 4 * THREADS + tid * 4,
                vec_width=4,
                dtype=T.f32,
            )
        )
        for j in range(HIDDEN // (4 * THREADS))
    ]
    await_x(c, s)
    xs = [
        [
            ld_dev(c["xn"].value, (t * HIDDEN + j * 4 * THREADS + tid * 4) // 2, 2)
            for j in range(HIDDEN // (4 * THREADS))
        ]
        for t in range(s)
    ]
    for t in range_constexpr(s):
        acc = fx.Float32(0.0)
        for j in range_constexpr(len(wv)):
            for h in range_constexpr(2):
                lo, hi = bf2_f32(fx.Int32(xs[t][j][h]))
                acc = acc + wv[j][2 * h] * lo
                acc = acc + wv[j][2 * h + 1] * hi
        tot = block_sum(c, acc, t % 2)
        if tid == 0:
            c["mb"].put(c["logit"], t * EXPERTS + e, tot)


# ---------------------------------------------------------------- route
def _sigmoid(x):
    return hw_rcp(fx.Float32(1.0) + hw_exp2(x * fx.Float32(-LOG2E)))


@traced
def stage_route(c, t):
    """Token t on wave 0: two experts a lane, top-4 of sigmoid + bias (ties to
    the lower expert), weights the selected sigmoids / their sum x 2.0."""
    lane, wave, a = c["lane"], c["wave"], c["args"]
    if wave == 0:
        got = c["mb"].poll(
            [(c["logit"], t * EXPERTS + lane + 64 * i, 1) for i in range(2)],
            site=SITE_ROUTE,
        )
        sc = [_sigmoid(got[i][0].bitcast(fx.Float32)) for i in range(2)]
        bias = [
            fx.Float32(
                bo.buffer_load(rsrc(a["bias"]), lane + 64 * i, vec_width=1, dtype=T.f32)
            )
            for i in range(2)
        ]
        val = [sc[i] + bias[i] for i in range(2)]
        taken = [fx.Int32(0), fx.Int32(0)]
        total = fx.Float32(0.0)
        my_id, my_w = fx.Int32(0), fx.Float32(0.0)
        neg = fx.Float32(float("-inf"))
        for k in range_constexpr(TOPK):
            v0 = (taken[0] == 0).select(val[0], neg)
            v1 = (taken[1] == 0).select(val[1], neg)
            pick1 = v1 > v0
            v = pick1.select(v1, v0)
            idx = pick1.select(fx.Int32(lane + 64), fx.Int32(lane))
            best = butterfly(v, (32, 16, 8, 4, 2, 1), fx.max)
            cand = (v == best).select(idx, fx.Int32(1 << 20))
            win = butterfly(cand, (32, 16, 8, 4, 2, 1), fx.min)
            mine0 = win == lane
            mine1 = win == lane + 64
            w_mine = mine0.select(sc[0], mine1.select(sc[1], fx.Float32(0.0)))
            w = butterfly(w_mine, (32, 16, 8, 4, 2, 1), fx.max)
            taken[0] = mine0.select(fx.Int32(1), taken[0])
            taken[1] = mine1.select(fx.Int32(1), taken[1])
            total = total + w
            my_id = (lane == k).select(win, my_id)
            my_w = (lane == k).select(w, my_w)
        scale = fx.Float32(ROUTED_SCALE) / total
        if lane < TOPK:
            c["mb"].put(c["route"], t * 2 * TOPK + lane, my_id)
            c["mb"].put(c["route"], t * 2 * TOPK + TOPK + lane, my_w * scale)


# ---------------------------------------------------------------- table
def tab_ld(c, i):
    return fx.ptr_load(c["tab"] + i)


def tab_st(c, v, i):
    fx.ptr_store(fx.Int32(v), c["tab"] + i)


@traced
def load_table(c):
    """Every token's picks -> the slots in LDS: slot 0 the shared expert over
    every row, slots 1 .. n_u the picked experts ascending; a slot's columns its
    rows ascending, each column's pick (t * 4 + k; the shared expert's 4 S + t)."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    n = s * TOPK
    if tid < n:
        t = tid // TOPK
        got = c["mb"].poll(
            [
                (c["route"], t * 2 * TOPK + tid % TOPK, 1),
                (c["route"], t * 2 * TOPK + TOPK + tid % TOPK, 1),
            ],
            site=SITE_TABLE,
        )
        tab_st(c, got[0][0], T_PICK_E + tid)
        tab_st(c, got[1][0], T_PICK_W + tid)
    if tid < EXPERTS:
        tab_st(c, 0, T_MASK + tid)
    gpu.barrier()
    if tid < n:
        e = tab_ld(c, T_PICK_E + tid)
        fx.llvm.atomic_or(
            c["tab"] + (T_MASK + e), fx.Int32(1) << (tid // TOPK), syncscope="workgroup"
        )
    gpu.barrier()
    e = tid
    m = fx.Int32(0)
    if tid < EXPERTS:
        m = tab_ld(c, T_MASK + e)
    ball = ballot(m != 0)
    if (wave == 0) & (lane == 0):
        tab_st(c, popcount(ball), T_MISC + 1)
    gpu.barrier()
    below = (wave == 1).select(tab_ld(c, T_MISC + 1), fx.Int32(0))
    slot = 1 + below + lanes_below(ball, lane)
    if tid < EXPERTS:
        tab_st(c, slot, T_ESLOT + e)
        if m != 0:
            tab_st(c, e, T_SLOT_E + slot)
            tab_st(c, m, T_SLOT_MASK + slot)
    if (wave == 1) & (lane == 0):
        tab_st(c, below + popcount(ball), T_MISC)
    if tid == 0:
        tab_st(c, SHARED, T_SLOT_E)
        tab_st(c, (1 << s) - 1, T_SLOT_MASK)
    if tid < s:
        tab_st(c, tid, T_SLOT_TOK + tid)
        tab_st(c, n + tid, T_SLOT_PICK + tid)
    gpu.barrier()
    if tid < n:
        t = tid // TOPK
        sl = tab_ld(c, T_ESLOT + tab_ld(c, T_PICK_E + tid))
        col = popcount(tab_ld(c, T_SLOT_MASK + sl) & ((fx.Int32(1) << t) - 1))
        tab_st(c, t, T_SLOT_TOK + sl * 16 + col)
        tab_st(c, tid, T_SLOT_PICK + sl * 16 + col)
    gpu.barrier()


def n_slots(c):
    return 1 + tab_ld(c, T_MISC)


def slot_count(c, u):
    return popcount(tab_ld(c, T_SLOT_MASK + u))


# ---------------------------------------------------------------- experts
def _swiglu(g, u):
    g = fx.min(g, fx.Float32(LIMIT))
    u = fx.max(fx.min(u, fx.Float32(LIMIT)), fx.Float32(-LIMIT))
    return g * _sigmoid(g * fx.Float32(ALPHA)) * (u + fx.Float32(BETA))


def x_operands(c, base, row, k_dwords, st0, steps):
    """The lane's B operands of steps st0 ..: row ``row`` of a bf16 [*, K]
    scratch (``k_dwords`` a row), 8 K values an MFMA."""
    lane = c["lane"]
    at = row * k_dwords + (lane // 16) * 16
    return [
        [ld_dev(base, at + (st0 + i) * 64 + d * 4, 4) for d in range(4)]
        for i in range(steps)
    ]


@traced
def stage_up(c, u, j, weights=None):
    """Slot u's tile pair j: gate rows 16 j .., up rows 768 + 16 j .. of its
    expert over this wave's K steps; waves summed in order; SwiGLU-OAI ->
    bf16 MID[u][column][16 j ..], then UPF."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    e = tab_ld(c, T_SLOT_E + u)
    cnt = slot_count(c, u)
    st0 = wave * UP_WAVE_STEPS
    rgs = [e * (2 * UP_TILES) + j, e * (2 * UP_TILES) + UP_TILES + j]
    if const_expr(weights is None):
        weights = up_weights(c, rgs, st0)
    wvs, svs = weights
    tok = tab_ld(c, T_SLOT_TOK + u * 16 + fx.min(lane % 16, cnt - 1))
    xs = x_operands(c, c["xn"].value, tok, HIDDEN // 2, st0, UP_WAVE_STEPS)
    accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)]
    for i in range_constexpr(UP_WAVE_STEPS):
        for h in range_constexpr(2):
            accs[h] = step_mfma(accs[h], wvs[h][i], scale_f32(svs[h], rgs[h], i), xs[i])
    for h in range_constexpr(2):
        fx.ptr_store(accs[h], red + ((wave * 2 + h) * 64 + lane) * 4)
    gpu.barrier()
    if tid < 256:
        col = tid // 16
        r = tid % 16
        if col < cnt:
            at = (16 * (r // 4) + col) * 4 + r % 4
            g = fx.Float32(0.0)
            up = fx.Float32(0.0)
            for w in range_constexpr(WAVES):
                g = g + fx.ptr_load(red + ((w * 2) * 64 * 4 + at))
                up = up + fx.ptr_load(red + ((w * 2 + 1) * 64 * 4 + at))
            v = _swiglu(g, up).to(fx.BFloat16)
            bo.buffer_store(
                v,
                rsrc(c["mid"].value),
                (u * 16 + col) * INTER + j * 16 + r,
                cache_modifier=CM_DEV,
            )
    flag_after_stores(c, c["upf"], u * UP_TILES + j, tid == 0)


def up_weights(c, rgs, st0):
    a = c["args"]
    wvs = [weight_loads(a["w13"], rg, HIDDEN, st0, UP_WAVE_STEPS) for rg in rgs]
    svs = [scale_loads(a["s13"], rg, HIDDEN, st0, UP_WAVE_STEPS) for rg in rgs]
    return wvs, svs


@traced
def stage_down(c, tasks):
    """Rows 16 task .. of every task in ``tasks``: a slot a wave, round robin,
    its MID columns loaded once for every task; a column's result ->
    LDS CONTRIB[task][pick], then each token's routed picks (top-k order) x
    their weights + the shared expert -> bf16 pairs pushed to every rank's FFN
    region."""
    s, tid, lane, wave, a = c["S"], c["tid"], c["lane"], c["wave"], c["args"]
    nt = len(tasks)
    nu = n_slots(c)
    for u in range(wave, nu, WAVES):
        e = tab_ld(c, T_SLOT_E + u)
        rgs = [e * DOWN_TASKS + task for task in tasks]
        wvs = [weight_loads(a["w2"], rg, INTER, 0, K2_STEPS) for rg in rgs]
        svs = [scale_loads(a["s2"], rg, INTER, 0, K2_STEPS) for rg in rgs]
        cnt = slot_count(c, u)
        c["mb"].poll(
            [(c["upf"], u * UP_TILES + fx.min(lane, UP_TILES - 1), 1)], site=SITE_DOWN
        )
        col = fx.min(lane % 16, cnt - 1)
        xs = x_operands(c, c["mid"].value, u * 16 + col, INTER // 2, 0, K2_STEPS)
        pick = tab_ld(c, T_SLOT_PICK + u * 16 + col)
        for ti in range_constexpr(nt):
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for i in range_constexpr(K2_STEPS):
                acc = step_mfma(acc, wvs[ti][i], scale_f32(svs[ti], rgs[ti], i), xs[i])
            if lane % 16 < cnt:
                for q in range_constexpr(4):
                    fx.ptr_store(
                        acc[q],
                        c["contrib"] + ((ti * 80 + pick) * 16 + 4 * (lane // 16) + q),
                    )
    gpu.barrier()
    if tid < nt * s * 8:
        ti = tid // (s * 8)
        t = tid % (s * 8) // 8
        r2 = tid % 8
        task = tasks[0]
        for q in range_constexpr(1, nt):
            task = (ti == q).select(tasks[q], task)
        # AITER's stage 2: bf16(weight x acc) a pick, added in bf16 (its
        # atomics' order is a race; here the shared expert, then top-k order)
        vals = []
        for h in range_constexpr(2):
            r = 2 * r2 + h
            tot = fx.Float32(0.0)
            if const_expr(c["experts"] != Experts.ROUTED):
                tot = bf16_round(
                    fx.ptr_load(c["contrib"] + ((ti * 80 + TOPK * s + t) * 16 + r))
                )
            if const_expr(c["experts"] != Experts.SHARED):
                for k in range_constexpr(TOPK):
                    w = tab_ld(c, T_PICK_W + t * TOPK + k).bitcast(fx.Float32)
                    v = fx.ptr_load(c["contrib"] + ((ti * 80 + t * TOPK + k) * 16 + r))
                    tot = bf16_round(tot + bf16_round(w * v))
            vals.append(tot)
        word = bf16_pair(vals[0], vals[1]).bitcast(fx.Int32)
        cp = task * 8 + r2
        for p in range_constexpr(c["tp"]):
            put_word(c, c["peer"][p], ffn_pair(c, c["rank"], t, cp), word)
    gpu.barrier()


@traced
def stage_final(c, task):
    s, tid, a = c["S"], c["tid"], c["args"]
    if tid < s * COLS // 2:
        t = tid // (COLS // 2)
        cp = task * (COLS // 2) + tid % (COLS // 2)
        got = c["mb"].poll(
            [(c["own"], ffn_pair(c, src, t, cp), 1) for src in range(c["tp"])],
            site=SITE_FINAL,
        )
        lo, hi = bf2_f32(got[0][0])
        for src in range_constexpr(1, c["tp"]):
            l2, h2 = bf2_f32(got[src][0])
            lo, hi = lo + l2, hi + h2
        bo.buffer_store(
            bf16_pair(lo, hi).bitcast(fx.Int32), rsrc(a["out"]), t * (HIDDEN // 2) + cp
        )


# ---------------------------------------------------------------- the launch
@traced
def stamp(c, k):
    # compile-time gate outside, traced condition inside
    if const_expr(c["timeline"]):  # noqa: SIM102
        if c["tid"] == 0:
            bo.buffer_store(
                memrealtime(), rsrc(c["tl"].value), fx.block_idx.x * TL_POINTS + k
            )


@traced
def run_ffn(c, bid):
    s, tid = c["S"], c["tid"]
    stamp(c, 0)
    for task in range(first_task(bid, 0), SEAM_TASKS, BLOCKS):
        stage_push(c, task)
    gpu.barrier()
    stamp(c, 1)
    for task in range(first_task(bid, 0), SEAM_TASKS, BLOCKS):
        stage_reduce(c, task)
    gpu.barrier()
    stamp(c, 2)
    for t in range(first_task(bid, NORM0), s, BLOCKS):
        stage_norm(c, t)
    stamp(c, 3)
    for e in range(first_task(bid, ROUTER0), EXPERTS, BLOCKS):
        stage_router(c, e)
    stamp(c, 4)
    for t in range(first_task(bid, NORM0), s, BLOCKS):
        stage_route(c, t)
    gpu.barrier()
    stamp(c, 5)
    sh = first_task(bid, SHARED0)
    if sh < UP_TILES:
        rgs = [SHARED * (2 * UP_TILES) + sh, SHARED * (2 * UP_TILES) + UP_TILES + sh]
        weights = up_weights(c, rgs, c["wave"] * UP_WAVE_STEPS)
        await_x(c, s)
        if tid == 0:
            tab_st(c, SHARED, T_SLOT_E)
            tab_st(c, (1 << s) - 1, T_SLOT_MASK)
        if tid < s:
            tab_st(c, tid, T_SLOT_TOK + tid)
        gpu.barrier()
        stage_up(c, 0, sh, weights)
    stamp(c, 6)
    load_table(c)
    stamp(c, 7)
    total = (n_slots(c) - 1) * UP_TILES
    for unit in range(first_task(bid, ROUTED0), total, BLOCKS):
        stage_up(c, 1 + unit // UP_TILES, unit % UP_TILES)
    stamp(c, 8)
    if bid < DOWN_TASKS - BLOCKS:
        stage_down(c, [bid, bid + BLOCKS])
    else:
        stage_down(c, [bid])
    stamp(c, 9)
    for task in range(first_task(bid, 0), SEAM_TASKS, BLOCKS):
        stage_final(c, task)
    stamp(c, 10)


def kernel_name(key: FfnBuild) -> str:
    name = f"minimax_m3_mono_ffn_s{key.tokens}_tp{key.tp}"
    if key.experts is not Experts.ALL:
        name += f"_{key.experts.value}"
    if key.timeline:
        name += "_tl"
    return name


def build_ffn(key: FfnBuild):
    s, tp = key.tokens, key.tp
    assert 1 <= s <= MAX_TOKENS
    layout = scratch_layout(s)
    half = peer_half_bytes(tp)
    keyed = key_tuple(key, SOURCES)

    @fx.struct
    class Lds:
        red: fx.Array[fx.Float32, WAVES * 2 * 64 * 4, 16]
        tab: fx.Array[fx.Int32, T_WORDS, 16]
        contrib: fx.Array[fx.Float32, 2 * 80 * 16, 16]

    @flyc.kernel(name=kernel_name(key), known_block_size=[THREADS, 1, 1])
    def kffn(
        part: Int64,
        res_in: Int64,
        res_out: Int64,
        out: Int64,
        norm_w: Int64,
        router_w: Int64,
        bias: Int64,
        w13: Int64,
        s13: Int64,
        w2: Int64,
        s2: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        mode: Int32,
    ):
        _ = keyed
        if mode != 0:
            bid = fx.block_idx.x
            tid = fx.thread_idx.x
            lds = fx.SharedAllocator().allocate(Lds).peek()
            ep = fx.Int32(bo.buffer_load(rsrc(epoch), 0, vec_width=1, dtype=T.i32))
            tag = ep + 1
            if tid == 0:
                bo.buffer_store(
                    tag, rsrc(epoch), EPOCH_MARKS + bid, cache_modifier=CM_DEV
                )
            raw = peer_bases(peers, tp)
            par = fx.Int64(ep & 1) * fx.Int64(half) + fx.Int64(ERROR_BYTES)
            mb = BoundedMailbox(tag, sym, raw, rank, ep)
            c = {
                "S": s,
                "tp": tp,
                "experts": key.experts,
                "timeline": key.timeline,
                "tid": tid,
                "lane": tid % 64,
                "wave": tid // 64,
                "rank": rank,
                "mb": mb,
                "red": lds.red.ptr,
                "tab": lds.tab.ptr,
                "contrib": lds.contrib.ptr,
                "own": preg(sym + par, 0, "peer"),
                "peer": [preg(b + par, 0, "peer") for b in raw],
                "args": {
                    "part": part,
                    "res_in": res_in,
                    "res_out": res_out,
                    "out": out,
                    "norm_w": norm_w,
                    "router_w": router_w,
                    "bias": bias,
                    "w13": w13,
                    "s13": s13,
                    "w2": w2,
                    "s2": s2,
                },
            }
            for region, (off, _) in layout.items():
                c[region] = sreg(scratch, off, region)
            run_ffn(c, bid)
            epoch_end(c, epoch, tag)
            stamp(c, 11)

    @flyc.jit
    def launch(
        part: Int64,
        res_in: Int64,
        res_out: Int64,
        out: Int64,
        norm_w: Int64,
        router_w: Int64,
        bias: Int64,
        w13: Int64,
        s13: Int64,
        w2: Int64,
        s2: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        mode: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kffn(
            part,
            res_in,
            res_out,
            out,
            norm_w,
            router_w,
            bias,
            w13,
            s13,
            w2,
            s2,
            scratch,
            sym,
            peers,
            rank,
            epoch,
            mode,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch


@traced
def epoch_end(c, epoch, tag):
    """CTA 0, once every CTA marked this launch: the epoch moves on."""
    tid = c["tid"]
    if fx.block_idx.x == 0:
        if tid < BLOCKS:
            mark = ld_dev(epoch, EPOCH_MARKS + tid)
            n = fx.Int32(0)
            while (fx.Int32(mark) != tag) & (n < (1 << 24)):
                rocdl.s_sleep(1)
                mark = ld_dev(epoch, EPOCH_MARKS + tid)
                n = n + 1
        gpu.barrier()
        if tid == 0:
            bo.buffer_store(tag, rsrc(epoch), 0, cache_modifier=CM_DEV)


__all__ = [
    "BLOCKS",
    "CM_SYS",
    "EPOCH_WORDS",
    "ERROR_WORDS",
    "Experts",
    "FfnBuild",
    "MAX_TOKENS",
    "build_ffn",
    "peer_bytes",
    "scratch_bytes",
]
