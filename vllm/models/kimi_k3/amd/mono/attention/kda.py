# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K1 ``kda_pre``: one Kimi-K3 KDA layer from the AttnRes seam to the gated core
output, for a spec-verify step (R requests x L tokens, S = R L rows).

One launch a layer, ``BLOCKS`` CTAs x ``THREADS`` threads, in three roles so that
no CTA that polls a mailbox has weight loads in flight (a poll's load retires
behind every earlier load):

    attn CTAs 0 .. 55   slice (128 hidden columns each): prefix (+ delta) ->
                        bf16 prefix out, per-source partial sum of squares and
                        qk dot (PART); gate (token t on CTA t): softmax weights
                        (WGT); mix: the softmax mix, its partial sum of squares
                        (MSQ), every slice's -> output RMSNorm -> bf16 x rows
                        (plain device-scope stores, an XRDY flag a slice)
    gemv CTAs 56 ..     in_proj, two 16-row groups each: the first group's
                        weights issued at kernel start, the second's as x lands
                        -> q|k|v|g|f_a|beta (PROJ)
    kda tasks           (R x 12 heads x 8 v-chunks) on CTAs 0 ..: inputs that
                        need no hand-off loaded first (metadata, conv window,
                        state rows, f_b rows), then conv update of the owned
                        channels (q / k out through QK), f_b MFMA + lower-bound
                        gate, l2norm, the recurrence on 16 state rows (a state a
                        token written), partial norms (ONRM), gated RMSNorm ->
                        core_out

Rounding follows the original path where it is elementwise (the AttnRes prefix
add, every bf16 hand-off between the original kernels); split-K reductions
differ only in their order.
"""

import math
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.kimi_k3.amd.mono.common.debug import stamp, stamp_begin, stamp_flush
from vllm.models.kimi_k3.amd.mono.common.ops import (
    CM_DEV,
    CM_NT,
    bf_hi,
    bf_lo,
    butterfly,
    hw_exp2,
    hw_rsq,
    ld_i32,
    lds_bytes,
    mfma_bf16,
    row_sum,
    rsrc,
    traced,
    uniform,
    wave_max,
)
from vllm.models.kimi_k3.amd.mono.common.plan import (
    BLOCKS,
    GFX942,
    LDS_BYTES,
    THREADS,
    WAVES,
    KernelAbi,
    first_task,
    key_tuple,
    pair_layout,
    source_digest,
)
from vllm.models.kimi_k3.amd.mono.common.sync import Mailbox, publish, sreg
from vllm.models.kimi_k3.amd.mono.stages import gemv

HIDDEN = 7168
NA = 56  # attn CTAs
COLS = HIDDEN // NA  # 128 columns a slice
QC = COLS // 4  # threads a (token, source) row: 4 columns each
GSL = 7  # slices a gate / mix-norm thread sums (NA = 8 x 7)
HD = 128
NH = 12  # local heads at TP8
PROJ = NH * HD  # 1536
QKV = 3 * PROJ
G2_0 = QKV
FA_0 = G2_0 + PROJ  # 6144
BETA_0 = FA_0 + HD  # 6272
NPROJ = 6288  # in_proj rows a rank holds (6284 + 4 pad)
ROWS = 16
GEMV_TASKS = NPROJ // ROWS  # 393
GEMV_CTAS = (GEMV_TASKS + 1) // 2  # 197
KCH = 128  # K a chunk: 4 MFMAs of 32
NKCH = HIDDEN // KCH  # 56
PER_WAVE = NKCH // WAVES  # 7
VCH = 16  # state rows a KDA task
NVCH = HD // VCH  # 8
CONV_W = 4
LOG2E = 1.0 / math.log(2.0)
LDS_PAD = 8  # bf16 a padded x row
# timeline points: start, slice, gate, mix, load_x, gemv0, gemv1, k.stage,
# k.conv, k.gates, k.recur, k.norm
TL_POINTS = 15

assert NA + GEMV_CTAS <= BLOCKS and NA == 8 * GSL

_STREAM = fx.Stream(None)
# every mono source file: in every build's JIT cache key
_SOURCES = source_digest(".")


@dataclass(frozen=True)
class KdaPreBuild:
    tokens: int  # S
    qlen: int  # L, tokens a request
    nblocks: int  # AttnRes block sources (the prefix is one more)
    delta: bool
    state_len: int  # conv state entries: CONV_W - 1 + num_spec
    eps: float = 1e-5  # AttnRes norm
    out_eps: float = 1e-5  # input_layernorm
    onorm_eps: float = 1e-5
    lower_bound: float = -5.0
    debug: bool = False
    timeline: bool = False
    write_idx: int = -1  # a block-write layer's block: the updated prefix stored there
    kda0: int = 0  # the first KDA task's CTA
    xp: int = 0  # experiment switch (timing only)


ABI = KernelAbi(
    (
        "prefix",
        "delta",
        "blocks",
        "blk_sm",
        "blk_sr",
        "ares_nw",
        "ares_qk",
        "in_nw",
        "w_in",
        "w_fb",
        "conv_w",
        "conv_st",
        "cs_seq",
        "cs_dim",
        "cs_tok",
        "a_log",
        "dt_bias",
        "on_w",
        "rstate",
        "rs_seq",
        "st_idx",
        "st_stride",
        "num_acc",
        "core_out",
        "x_dbg",
        "proj_dbg",
        "scratch",
        "layer",
        "epoch",
        "tl",
    )
)


def scratch_layout(key: KdaPreBuild) -> dict:
    s, ns = key.tokens, key.nblocks + 1
    assert s % key.qlen == 0
    pairs = {
        "part": NA * s * ns * 2,
        "wgt": s * (ns + 1),
        "msq": NA * s,
        "xrdy": NA,
        "proj": s * NPROJ,
        "qk": s * NH * 2 * HD,
        "onrm": s * NH * NVCH,
    }
    lay = pair_layout(pairs.items())
    end = max(o + n for o, n in lay.values())
    # the plain bf16 x rows (not pairs), 16 B aligned
    lay["x"] = ((end + 15) // 16 * 16, s * HIDDEN * 2)
    return lay


def scratch_bytes(key: KdaPreBuild) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


TAG_SLOTS = 256  # a step's mailbox tags: epoch x 256 + a launch's slot


def step_tag(epoch, slot):
    """This launch's mailbox tag: the step's epoch (a device counter the runner
    bumps once a step, in the graph) and the launch's slot, so no step's or
    other launch's pair ever passes for this one's -- nothing is zeroed."""
    e = uniform(ld_i32(epoch, 0))
    return (e % (1 << 22)) * TAG_SLOTS + slot + 1


def ld_bf(ptr, i, cm=0):
    return fx.Float32(
        fx.BFloat16(
            bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.bf16, cache_modifier=cm)
        )
    )


def ld_bf4(ptr, i):
    """bf16 elements i .. i + 4 (i a multiple of 4) as four f32."""
    w = fx.Vector(bo.buffer_load(rsrc(ptr), i // 2, vec_width=2, dtype=T.i32))
    return [bf_lo(w[0]), bf_hi(w[0]), bf_lo(w[1]), bf_hi(w[1])]


def bf4_words(v):
    """Four f32 -> two words of packed bf16."""
    return fx.Vector.from_elements(v, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)


def ld_f32(ptr, i):
    return fx.Float32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.f32))


def bf16r(v):
    return fx.Float32(fx.Float32(v).to(fx.BFloat16))


def sigmoid(z):
    return fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(z * fx.Float32(-LOG2E)))


def f32_of(w):
    return w.bitcast(fx.Float32)


def sum_polled(vals):
    acc = f32_of(vals[0][0])
    for k in range(1, len(vals)):
        acc = acc + f32_of(vals[k][0])
    return acc


# ---------------------------------------------------------------- attn CTAs
@traced
def stage_slice(c, task):
    """Columns 128 task ..: every token's sources into LDS ``vals`` (4 columns a
    thread), the prefix (+ delta) rounded to bf16 and stored back, per-(token,
    source) partial sum of squares and qk dot -> PART."""
    s, ns, nb, tid, lane = c["S"], c["NS"], c["NB"], c["tid"], c["lane"]
    a = c["args"]
    c0 = task * COLS
    n = s * ns * QC  # threads' work items
    for i in range_constexpr((n + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, n - 1)
        live = (tid + THREADS * i) < n
        row = e // QC  # (token, source)
        t = row // ns
        src = row % ns
        col = c0 + (e % QC) * 4
        is_pre = src == nb
        blk = ld_bf4(
            a["blocks"], t * a["blk_sm"] + fx.min(src, nb - 1) * a["blk_sr"] + col
        )
        pre = ld_bf4(a["prefix"], t * HIDDEN + col)
        if const_expr(c["delta"]):
            if const_expr(c.get("delta_lds") is not None):
                # the delta reduced into LDS first (K2a: the all-reduced o_proj)
                dl = [
                    fx.ptr_load(c["delta_lds"] + (t * COLS + (e % QC) * 4 + j))
                    for j in range(4)
                ]
            else:
                dl = ld_bf4(a["delta"], t * HIDDEN + col)
            if const_expr(c.get("reset", False)):
                # a block-write layer's MLP seam: the prefix restarts at the
                # attention's output (the input prefix is not read)
                pre = dl
            else:
                pre = [bf16r(pre[j] + dl[j]) for j in range(4)]
            if live & is_pre:
                bo.buffer_store(
                    bf4_words(pre), rsrc(a["prefix"]), (t * HIDDEN + col) // 2
                )
        if const_expr(c.get("write_idx", -1) >= 0):  # noqa: SIM102 (traced: `and` does not combine traced values)
            # the block-write layer's attention seam: the updated prefix is
            # block write_idx
            if live & is_pre:
                bo.buffer_store(
                    bf4_words(pre),
                    rsrc(a["blocks"]),
                    (t * a["blk_sm"] + c["write_idx"] * a["blk_sr"] + col) // 2,
                )
        v = [
            live.select(is_pre.select(pre[j], blk[j]), fx.Float32(0.0))
            for j in range(4)
        ]
        w = ld_bf4(a["ares_nw"], col)
        q = ld_bf4(a["ares_qk"], col)
        # a thread past the end holds a clamped copy of the last element: it
        # must not store (it raced the owner with zeros)
        if live:
            for j in range_constexpr(4):
                fx.ptr_store(v[j], c["vals"] + (row * COLS + (e % QC) * 4 + j))
        sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
        dt = (
            v[0] * (w[0] * q[0])
            + v[1] * (w[1] * q[1])
            + v[2] * (w[2] * q[2])
            + v[3] * (w[3] * q[3])
        )
        sq = butterfly(sq, (16, 8, 4, 2, 1))
        dt = butterfly(dt, (16, 8, 4, 2, 1))
        if live & (lane % 32 == 0):
            at = ((task * s + t) * ns + src) * 2
            c["put"](c["part"], at, sq)
            c["put"](c["part"], at + 1, dt)
    gpu.barrier()


@traced
def stage_gate(c, t):
    """Token t: the 56 slices' partials -> source weights e_s and their sum."""
    s, ns, tid, lane, wave, red = (
        c["S"],
        c["NS"],
        c["tid"],
        c["lane"],
        c["wave"],
        c["red"],
    )
    nq = ns * 2
    if tid < 8 * nq:
        g = tid // nq
        q = tid % nq
        vals = c["poll"](
            [(c["part"], (((g * GSL + k) * s + t) * ns) * 2 + q, 1) for k in range(GSL)]
        )
        fx.ptr_store(sum_polled(vals), red + (g * nq + q))
    gpu.barrier()
    if wave == 0:
        src = fx.min(lane, ns - 1)
        sq = fx.ptr_load(red + src * 2)
        dt = fx.ptr_load(red + src * 2 + 1)
        for g in range_constexpr(1, 8):
            sq = sq + fx.ptr_load(red + (g * nq + src * 2))
            dt = dt + fx.ptr_load(red + (g * nq + src * 2 + 1))
        rstd = hw_rsq(sq * fx.Float32(1.0 / HIDDEN) + fx.Float32(c["eps"]))
        logit = dt * rstd * fx.Float32(LOG2E)
        logit = (lane < ns).select(logit, fx.Float32(-3.0e38))
        m = wave_max(logit)
        ex = (lane < ns).select(hw_exp2(logit - m), fx.Float32(0.0))
        den = butterfly(ex, (32, 16, 8, 4, 2, 1))
        if lane < ns:
            c["put"](c["wgt"], t * (ns + 1) + lane, ex)
        if lane == 0:
            c["put"](c["wgt"], t * (ns + 1) + ns, den)
    gpu.barrier()


GATE_OFF = 512  # red words of stage_gate_local's group sums


@traced
def stage_gate_local(c):
    """Every token's source weights computed in this CTA (``stage_gate``'s sums
    in its order, so every CTA's are bit-identical): the 56 slices' partials
    polled in one round trip (no gate CTA, no WGT hand-off) -> red[t (ns + 1)
    + src], the sum at + ns."""
    s, ns, tid, lane, wave, red = (
        c["S"],
        c["NS"],
        c["tid"],
        c["lane"],
        c["wave"],
        c["red"],
    )
    nq = ns * 2
    n = 8 * s * nq  # (group, token, partial) items
    per = (n + THREADS - 1) // THREADS
    specs = []
    for r in range_constexpr(per):
        item = fx.min(tid + THREADS * r, n - 1)
        g = item // (s * nq)
        t = (item // nq) % s
        q = item % nq
        specs += [
            (c["part"], (((g * GSL + k) * s + t) * ns) * 2 + q, 1) for k in range(GSL)
        ]
    got = c["poll"](specs, per * GSL)
    for r in range_constexpr(per):
        acc = f32_of(got[r * GSL][0])
        for k in range_constexpr(1, GSL):
            acc = acc + f32_of(got[r * GSL + k][0])
        if tid + THREADS * r < n:
            fx.ptr_store(acc, red + (GATE_OFF + tid + THREADS * r))
    gpu.barrier()
    # wave t: token t's weights (stage_gate's math)
    if wave < s:
        t = wave
        src = fx.min(lane, ns - 1)
        base = GATE_OFF + t * nq
        sq = fx.ptr_load(red + (base + src * 2))
        dt = fx.ptr_load(red + (base + src * 2 + 1))
        for g in range_constexpr(1, 8):
            sq = sq + fx.ptr_load(red + (base + g * s * nq + src * 2))
            dt = dt + fx.ptr_load(red + (base + g * s * nq + src * 2 + 1))
        rstd = hw_rsq(sq * fx.Float32(1.0 / HIDDEN) + fx.Float32(c["eps"]))
        logit = dt * rstd * fx.Float32(LOG2E)
        logit = (lane < ns).select(logit, fx.Float32(-3.0e38))
        m = wave_max(logit)
        ex = (lane < ns).select(hw_exp2(logit - m), fx.Float32(0.0))
        den = butterfly(ex, (32, 16, 8, 4, 2, 1))
        if lane < ns:
            fx.ptr_store(ex, red + (t * (ns + 1) + lane))
        if lane == 0:
            fx.ptr_store(den, red + (t * (ns + 1) + ns))
    gpu.barrier()


@traced
def stage_mix(c, task):
    """Columns 128 task ..: the softmax mix of ``vals`` (4 columns a thread), its
    sum of squares (MSQ), every slice's -> rstd, then bf16 x = mixed rstd w ->
    the plain x rows, and the slice's XRDY flag."""
    s, ns, tid, lane, red = c["S"], c["NS"], c["tid"], c["lane"], c["red"]
    a = c["args"]
    c0 = task * COLS
    nw = s * (ns + 1)
    if const_expr(c.get("gate_local", False)):
        stage_gate_local(c)
    else:
        if tid < nw:
            v = c["poll"]([(c["wgt"], tid, 1)])[0][0]
            fx.ptr_store(f32_of(v), red + tid)
        gpu.barrier()
    mine = tid < s * QC
    t = fx.min(tid // QC, s - 1)
    cc = (tid % QC) * 4
    mixed = []
    for j in range_constexpr(4):
        acc = fx.Float32(0.0)
        for src in range_constexpr(ns):
            acc = acc + fx.ptr_load(red + (t * (ns + 1) + src)) * fx.ptr_load(
                c["vals"] + ((t * ns + src) * COLS + cc + j)
            )
        mixed.append(acc / fx.ptr_load(red + (t * (ns + 1) + ns)))
    sq = (
        mixed[0] * mixed[0]
        + mixed[1] * mixed[1]
        + mixed[2] * mixed[2]
        + mixed[3] * mixed[3]
    )
    sq = butterfly(sq, (16, 8, 4, 2, 1))
    if mine & (lane % 32 == 0):
        c["put"](c["msq"], task * s + t, sq)
    gpu.barrier()
    if tid < 8 * s:
        g = tid // s
        tt = tid % s
        vals = c["poll"]([(c["msq"], (g * GSL + k) * s + tt, 1) for k in range(GSL)])
        fx.ptr_store(sum_polled(vals), red + (128 + g * s + tt))
    gpu.barrier()
    tot = fx.ptr_load(red + (128 + t))
    for g in range_constexpr(1, 8):
        tot = tot + fx.ptr_load(red + (128 + g * s + t))
    r = hw_rsq(tot * fx.Float32(1.0 / HIDDEN) + fx.Float32(c["out_eps"]))
    wn = ld_bf4(a["in_nw"], c0 + cc)
    xw = bf4_words([mixed[j] * r * wn[j] for j in range(4)])
    if mine:
        at = (t * HIDDEN + c0 + cc) // 2
        bo.buffer_store(xw, rsrc(c["xbuf"]), at, cache_modifier=CM_DEV)
        if const_expr(c["debug"]):
            bo.buffer_store(xw, rsrc(a["x_dbg"]), at)
    publish(c["put"], c["xrdy"], task, fx.Int32(1), tid == 0)
    gpu.barrier()


# ---------------------------------------------------------------- gemv CTAs
def gemv_loads(c, rg, chunks):
    """This wave's weight operands of rows 16 rg .. for its K chunks ``chunks``
    (indices into wave, wave + 8, ..): a lane 64 contiguous bytes of its row a
    chunk (4 dwordx4). ``rg`` past the last group loads the last group."""
    lane, wave = c["lane"], c["wave"]
    r_w = rsrc(c["args"]["w_in"])
    row = fx.min(rg, GEMV_TASKS - 1) * ROWS + lane % ROWS
    g = lane // ROWS
    out = []
    for i in chunks:
        k = (wave + WAVES * i) * KCH + g * 8
        base = (row * HIDDEN + k) // 2  # dwords
        out.append(
            [
                fx.Vector(
                    bo.buffer_load(
                        r_w,
                        base + 16 * q,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_NT,
                    )
                )
                for q in range(4)
            ]
        )
    return out


@traced
def load_x(c):
    """Every slice's x ready -> the bf16 x rows into LDS (rows padded)."""
    s, tid = c["S"], c["tid"]
    if tid < NA:
        c["poll"]([(c["xrdy"], tid, 1)])
    gpu.barrier()
    words = s * HIDDEN // 8  # 16 B chunks
    row = HIDDEN + LDS_PAD
    for i in range_constexpr((words + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, words - 1)
        v = fx.Vector(
            bo.buffer_load(
                rsrc(c["xbuf"]), e * 4, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV
            )
        )
        t = e // (HIDDEN // 8)
        k = e % (HIDDEN // 8) * 8
        if tid + THREADS * i < words:
            fx.ptr_store(v, c["xl"] + (t * row + k) // 2)
    gpu.barrier()


@traced
def stage_gemv(c, rg, ops):
    """Rows 16 rg .. of in_proj against the LDS x -> bf16 PROJ (and proj_dbg)."""
    s, lane, wave = c["S"], c["lane"], c["wave"]
    g = lane // ROWS
    t = fx.min(lane % ROWS, s - 1)
    row = HIDDEN + LDS_PAD
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(PER_WAVE):
        k = (wave + WAVES * i) * KCH + g * 8
        for q in range_constexpr(4):
            xb = fx.Vector(
                fx.ptr_load(
                    c["xl"] + (t * row + k + 32 * q) // 2,
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
            )
            acc = mfma_bf16(
                ops[i][q].bitcast(fx.BFloat16), xb.bitcast(fx.BFloat16), acc
            )
    gemv_out(c, rg, acc)


GEMV_AHEAD = 4  # gfx942: rounds of both groups' weights in flight


@traced
def stage_gemv_pair(c, rg0, rg1):
    """gfx942: both row groups against x a K window at a time (LDS holds
    ``gemv.win_rounds`` rounds of the S rows): the next window's x is loaded
    under the current one's MFMAs, before the weights of GEMV_AHEAD rounds on
    (loads retire in order: x waits on no weights it does not need)."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    ahead = min(GEMV_AHEAD, PER_WAVE)
    w = [None] * PER_WAVE
    for i in range_constexpr(ahead):
        w[i] = (gemv_loads(c, rg0, [i])[0], gemv_loads(c, rg1, [i])[0])
    if tid < NA:
        c["poll"]([(c["xrdy"], tid, 1)])
    gpu.barrier()
    stamp(c["on"], c["tls"], tid, 4)
    r = gemv.win_rounds(s, HIDDEN)
    wk = r * gemv.ROUND_K
    xrow = gemv.win_row(s, HIDDEN)
    nw = gemv.windows(s, HIDDEN)
    acc0 = fx.Vector.filled(4, 0.0, fx.Float32)
    acc1 = fx.Vector.filled(4, 0.0, fx.Float32)
    pre = gemv.window_loads(tid, c["xbuf"], HIDDEN, 0, s, min(wk, HIDDEN), CM_DEV)
    for p in range_constexpr(nw):
        kw = min(wk, HIDDEN - p * wk)
        if p > 0:
            gpu.barrier()
        gemv.window_store(tid, pre, s, kw, c["xl"], xrow)
        if p + 1 < nw:
            k1 = (p + 1) * wk
            pre = gemv.window_loads(
                tid, c["xbuf"], HIDDEN, k1, s, min(wk, HIDDEN - k1), CM_DEV
            )
        gpu.barrier()
        for i in range_constexpr(p * r, min((p + 1) * r, PER_WAVE)):
            if i + ahead < PER_WAVE:
                w[i + ahead] = (
                    gemv_loads(c, rg0, [i + ahead])[0],
                    gemv_loads(c, rg1, [i + ahead])[0],
                )
            kk = (wave + WAVES * i) * KCH - p * wk
            acc0 = gemv.chunk_mfmas(lane, c["xl"], xrow, s, w[i][0], kk, acc0)
            acc1 = gemv.chunk_mfmas(lane, c["xl"], xrow, s, w[i][1], kk, acc1)
    gemv_out(c, rg0, acc0)
    stamp(c["on"], c["tls"], tid, 5)
    gemv_out(c, rg1, acc1)


@traced
def gemv_out(c, rg, acc):
    """The waves' accumulators of rows 16 rg .. summed -> bf16 PROJ (and
    proj_dbg)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if (tid < ROWS * s) & (rg < GEMV_TASKS):
        r = tid % ROWS
        tt = tid // ROWS
        v = bf16r(row_sum(red, r, tt))
        c["put"](c["proj"], tt * NPROJ + rg * ROWS + r, v)
        if const_expr(c["debug"]):
            bo.buffer_store(
                v.to(fx.BFloat16),
                rsrc(c["args"]["proj_dbg"]),
                tt * NPROJ + rg * ROWS + r,
            )
    gpu.barrier()


# ---------------------------------------------------------------- kda
def kda_stage(c, req, h, ch):
    """The task's inputs that need no hand-off, issued before its first poll:
    metadata, the owned channels' conv window / weights, the initial state rows,
    the f_b rows of this wave and the per-head constants."""
    L, sl, tid, lane, wave = c["L"], c["SL"], c["tid"], c["lane"], c["wave"]
    a = c["args"]
    k = {}
    # every value a buffer resource is built on wave-uniform (else a waterfall)
    k["acc_idx"] = uniform(ld_i32(a["num_acc"], req)) - 1
    k["slot_c"] = uniform(ld_i32(a["st_idx"], req * a["st_stride"]))
    k["slot0"] = uniform(ld_i32(a["st_idx"], req * a["st_stride"] + k["acc_idx"]))
    k["slots"] = [
        uniform(ld_i32(a["st_idx"], req * a["st_stride"] + j)) for j in range(L)
    ]
    # conv: thread < 48 owns channel (part, d)
    ct = fx.min(tid, 3 * VCH - 1)
    part = ct // VCH
    chan = part * PROJ + h * HD + ch * VCH + ct % VCH
    cs = a["conv_st"] + fx.Int64(fx.max(k["slot_c"], 0)) * fx.Int64(a["cs_seq"]) * 2
    co = chan * a["cs_dim"]  # the channel's first entry (bf16 elements)
    k["cs"], k["co"], k["chan"], k["part"] = cs, co, chan, part
    k["cw"] = fx.Vector(
        bo.buffer_load(rsrc(a["conv_w"]), chan * CONV_W, vec_width=4, dtype=T.f32)
    )
    k["win"] = [
        ld_bf(cs, co + (k["acc_idx"] + j) * a["cs_tok"]) for j in range(CONV_W - 1)
    ]
    k["old"] = [
        ld_bf(cs, co + (k["acc_idx"] + 1 + j) * a["cs_tok"]) for j in range(sl - L)
    ]
    # recurrence: thread -> state row v = tid / 32, k columns (tid % 32) 4 ..
    vr = ch * VCH + tid // 32
    k0 = (tid % 32) * 4
    k["head_off"] = (h * HD + vr) * HD + k0
    base0 = a["rstate"] + fx.Int64(fx.max(k["slot0"], 0)) * fx.Int64(a["rs_seq"]) * 4
    k["st"] = fx.Vector(
        bo.buffer_load(rsrc(base0), k["head_off"], vec_width=4, dtype=T.f32)
    )
    # f_b: wave w's rows 16 w .. of head h, a lane 64 B of its row
    fr = h * HD + wave * 16 + lane % 16
    fb = (fr * HD + (lane // 16) * 32) // 2
    k["fb"] = [
        fx.Vector(bo.buffer_load(rsrc(a["w_fb"]), fb + 4 * q, vec_width=4, dtype=T.i32))
        for q in range(4)
    ]
    k["aa"] = hw_exp2(ld_f32(a["a_log"], h) * fx.Float32(LOG2E))
    k["dtb"] = [
        ld_f32(a["dt_bias"], h * HD + wave * 16 + 4 * (lane // 16) + i)
        for i in range(4)
    ]
    return k


@traced
def kda_conv(c, k, req, h, ch):
    """Owned channels: the spec conv update of request ``req``'s L tokens -> new
    conv state; q / k out through QK, v to LDS."""
    L, sl, tid = c["L"], c["SL"], c["tid"]
    a = c["args"]
    stamp(c["on"], c["tls"], tid, 12)
    if tid < 3 * VCH:
        xs = c["poll"](
            [(c["proj"], (req * L + j) * NPROJ + k["chan"], 1) for j in range(L)]
        )
        xv = [f32_of(xs[j][0]) for j in range(L)]
        stamp(c["on"], c["tls"], tid, 13)
        rs = rsrc(k["cs"])
        if const_expr(c["xp"] == 2):
            # force the staged conv window before timing the stores
            fx.ptr_store(k["old"][0] + k["win"][0], c["red"] + tid)
            stamp(c["on"], c["tls"], tid, 13)
        if (k["slot_c"] > 0) & (c["xp"] != 1):
            for j in range_constexpr(sl):
                nv = k["old"][j] if j < sl - L else xv[j - (sl - L)]
                bo.buffer_store(
                    fx.Float32(nv).to(fx.BFloat16), rs, k["co"] + j * a["cs_tok"]
                )
        stamp(c["on"], c["tls"], tid, 14)
        seq = k["win"] + xv
        for j in range_constexpr(L):
            acc = fx.Float32(0.0)
            for q in range_constexpr(CONV_W):
                acc = acc + k["cw"][q] * seq[j + q]
            o = bf16r(acc * sigmoid(acc))
            if k["part"] < 2:
                c["put"](
                    c["qk"],
                    ((req * NH + h) * L + j) * 2 * HD
                    + k["part"] * HD
                    + ch * VCH
                    + tid % VCH,
                    o,
                )
            if k["part"] == 2:
                fx.ptr_store(o, c["vloc"] + (j * VCH + tid % VCH))
    gpu.barrier()


@traced
def kda_gates(c, k, req, h):
    """f_a, beta and the head's q / k (one poll batch a thread); g1 = f_b f_a by
    MFMA (a wave 16 rows) -> lower-bound gate -> decay in LDS; q / k l2-normed
    (q scaled)."""
    L, tid, lane, wave = c["L"], c["tid"], c["lane"], c["wave"]
    n_fa = L * HD
    n = n_fa + L * 2 * HD
    per = (n + THREADS - 1) // THREADS
    specs = []
    for i in range_constexpr(per):
        e = fx.min(tid + THREADS * i, n - 1)
        fa_at = (req * L + e // HD) * NPROJ + FA_0 + e % HD
        qk_at = (req * NH + h) * L * 2 * HD + (e - n_fa)
        specs.append((e, fa_at, qk_at))
    # two regions, two batches: the f_a pairs, then the q / k pairs
    got_fa = c["poll"](
        [(c["proj"], fx.min(fa, (req * L + L) * NPROJ - 1), 1) for _, fa, _ in specs]
    )
    got_qk = c["poll"](
        [(c["qk"], fx.max(qk, (req * NH + h) * L * 2 * HD), 1) for _, _, qk in specs]
    )
    for i in range_constexpr(per):
        e = specs[i][0]
        if tid + THREADS * i < n_fa:
            fx.ptr_store(f32_of(got_fa[i][0]), c["fa"] + e)
        if (tid + THREADS * i >= n_fa) & (tid + THREADS * i < n):
            fx.ptr_store(f32_of(got_qk[i][0]), c["qkl"] + (e - n_fa))
    if tid < L:
        b = c["poll"]([(c["proj"], (req * L + tid) * NPROJ + BETA_0 + h, 1)])[0][0]
        fx.ptr_store(sigmoid(f32_of(b)), c["beta"] + tid)
    gpu.barrier()
    # g1 rows 16 wave .. x tokens: A = f_b rows (staged), B = f_a [token][K]
    t = fx.min(lane % 16, L - 1)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for q in range_constexpr(4):
        kb = (lane // 16) * 32 + 8 * q
        xs = [fx.ptr_load(c["fa"] + (t * HD + kb + e)) for e in range(8)]
        xb = fx.Vector.from_elements(xs, fx.Float32).to(fx.BFloat16)
        acc = mfma_bf16(k["fb"][q].bitcast(fx.BFloat16), xb, acc)
    for i in range_constexpr(4):
        kk = wave * 16 + 4 * (lane // 16) + i
        g = bf16r(acc[i]) + k["dtb"][i]
        gate = fx.Float32(c["lb"]) * sigmoid(k["aa"] * g)
        if lane % 16 < L:
            fx.ptr_store(hw_exp2(gate * fx.Float32(LOG2E)), c["decay"] + (t * HD + kk))
    # l2norm: q / k row r (2 L rows of 128) on wave r % 8
    for rr in range_constexpr((2 * L + WAVES - 1) // WAVES):
        row = fx.min(wave + WAVES * rr, 2 * L - 1)
        x0 = fx.ptr_load(c["qkl"] + (row * HD + lane))
        x1 = fx.ptr_load(c["qkl"] + (row * HD + 64 + lane))
        ss = butterfly(x0 * x0 + x1 * x1, (32, 16, 8, 4, 2, 1))
        sc = hw_rsq(ss + fx.Float32(1e-6))
        sc = (row % 2 == 0).select(sc * fx.Float32(HD**-0.5), sc)
        if wave + WAVES * rr < 2 * L:
            fx.ptr_store(x0 * sc, c["qkl"] + (row * HD + lane))
            fx.ptr_store(x1 * sc, c["qkl"] + (row * HD + 64 + lane))
    gpu.barrier()


@traced
def kda_recur(c, k, req, h, ch):
    """State rows 16 ch .. of head h: the recurrence over the L tokens, a state
    written a token, bf16 outputs to LDS ``ol`` and their partial norms (ONRM)."""
    L, tid, lane = c["L"], c["tid"], c["lane"]
    a = c["args"]
    v = tid // 32
    k0 = (tid % 32) * 4
    live0 = k["slot0"] > 0
    stv = [live0.select(k["st"][e], fx.Float32(0.0)) for e in range(4)]
    for j in range_constexpr(L):
        qv = [fx.ptr_load(c["qkl"] + ((2 * j) * HD + k0 + e)) for e in range(4)]
        kv = [fx.ptr_load(c["qkl"] + ((2 * j + 1) * HD + k0 + e)) for e in range(4)]
        dk = [fx.ptr_load(c["decay"] + (j * HD + k0 + e)) for e in range(4)]
        stv = [stv[e] * dk[e] for e in range(4)]
        p = stv[0] * kv[0] + stv[1] * kv[1] + stv[2] * kv[2] + stv[3] * kv[3]
        p = butterfly(p, (16, 8, 4, 2, 1))
        vn = (fx.ptr_load(c["vloc"] + (j * VCH + v)) - p) * fx.ptr_load(c["beta"] + j)
        stv = [stv[e] + vn * kv[e] for e in range(4)]
        o = stv[0] * qv[0] + stv[1] * qv[1] + stv[2] * qv[2] + stv[3] * qv[3]
        o = bf16r(butterfly(o, (16, 8, 4, 2, 1)))
        o = live0.select(o, fx.Float32(0.0))
        slot = k["slots"][j]
        if (slot > 0) & live0:
            dst = rsrc(a["rstate"] + fx.Int64(slot) * fx.Int64(a["rs_seq"]) * 4)
            bo.buffer_store(
                fx.Vector.from_elements(stv, fx.Float32), dst, k["head_off"]
            )
        if lane % 32 == 0:
            fx.ptr_store(o, c["ol"] + (j * VCH + v))
    gpu.barrier()
    if tid < L:
        ss = fx.Float32(0.0)
        for e in range_constexpr(VCH):
            x = fx.ptr_load(c["ol"] + (tid * VCH + e))
            ss = ss + x * x
        c["put"](c["onrm"], ((req * NH + h) * L + tid) * NVCH + ch, ss)
    gpu.barrier()


@traced
def kda_norm(c, req, h, ch):
    """Rows 16 ch ..: gated RMSNorm over the head's 128 (every chunk's partial)
    -> core_out."""
    L, tid = c["L"], c["tid"]
    a = c["args"]
    if tid < L * VCH:
        j = tid // VCH
        e = tid % VCH
        vr = ch * VCH + e
        got = c["poll"](
            [(c["onrm"], ((req * NH + h) * L + j) * NVCH + q, 1) for q in range(NVCH)]
            + [(c["proj"], (req * L + j) * NPROJ + G2_0 + h * HD + vr, 1)]
        )
        ss = sum_polled(got[:NVCH])
        r = hw_rsq(ss * fx.Float32(1.0 / HD) + fx.Float32(c["onorm_eps"]))
        y = (
            fx.ptr_load(c["ol"] + (j * VCH + e))
            * r
            * ld_bf(a["on_w"], vr)
            * sigmoid(f32_of(got[NVCH][0]))
        )
        bo.buffer_store(
            y.to(fx.BFloat16), rsrc(a["core_out"]), (req * L + j) * PROJ + h * HD + vr
        )
    gpu.barrier()


@traced
def kda_tasks(c, on, tls):
    s, L, tid, bid = c["S"], c["L"], c["tid"], c["bid"]
    for task in range(first_task(bid, c["kda0"]), (s // L) * NH * NVCH, BLOCKS):
        req = task // (NH * NVCH)
        h = (task // NVCH) % NH
        ch = task % NVCH
        k = kda_stage(c, req, h, ch)
        stamp(on, tls, tid, 7)
        kda_conv(c, k, req, h, ch)
        stamp(on, tls, tid, 8)
        kda_gates(c, k, req, h)
        stamp(on, tls, tid, 9)
        kda_recur(c, k, req, h, ch)
        stamp(on, tls, tid, 10)
        kda_norm(c, req, h, ch)
        stamp(on, tls, tid, 11)


# ---------------------------------------------------------------- kernel
_BUILDS: dict = {}


def build(key: KdaPreBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    s, L = key.tokens, key.qlen
    ns = key.nblocks + 1
    lay = scratch_layout(key)
    # gfx942: x a K window at a time (``stage_gemv_pair``)
    xrow = gemv.win_row(s, HIDDEN) if GFX942 else HIDDEN + LDS_PAD
    assert L <= 16 and key.nblocks >= 1

    @fx.struct
    class AttnLds:
        vals: fx.Array[fx.Float32, s * ns * COLS, 16]

    @fx.struct
    class GemvLds:
        xl: fx.Array[fx.Int32, s * xrow // 2, 16]

    @fx.struct
    class KdaLds:
        fa: fx.Array[fx.Float32, L * HD, 16]
        qkl: fx.Array[fx.Float32, L * 2 * HD, 16]
        decay: fx.Array[fx.Float32, L * HD, 16]
        vloc: fx.Array[fx.Float32, L * VCH, 16]
        ol: fx.Array[fx.Float32, L * VCH, 16]
        beta: fx.Array[fx.Float32, 16, 16]

    @fx.union
    class RoleLds:
        attn: AttnLds
        gemv: GemvLds
        kda: KdaLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    assert lds_bytes(Smem) + lds_bytes(RoleLds) <= LDS_BYTES, "LDS overflow"
    keyed = key_tuple(key, _SOURCES)
    name = (
        f"k3_mono_kda_pre_s{s}_l{L}_nb{key.nblocks}_d{int(key.delta)}"
        f"_g{int(key.debug)}_t{int(key.timeline)}"
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kda_pre(
        prefix: Int64,
        delta: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        in_nw: Int64,
        w_in: Int64,
        w_fb: Int64,
        conv_w: Int64,
        conv_st: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        a_log: Int64,
        dt_bias: Int64,
        on_w: Int64,
        rstate: Int64,
        rs_seq: Int32,
        st_idx: Int64,
        st_stride: Int32,
        num_acc: Int64,
        core_out: Int64,
        x_dbg: Int64,
        proj_dbg: Int64,
        scratch: Int64,
        layer: Int32,
        epoch: Int64,
        tl: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        role = alloc.allocate(RoleLds)
        al, gl, kl = role.attn.peek(), role.gemv.peek(), role.kda.peek()
        mb = Mailbox(step_tag(epoch, layer) - 1)
        c = {
            "S": s,
            "L": L,
            "NS": ns,
            "NB": key.nblocks,
            "SL": key.state_len,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "delta": key.delta,
            "debug": key.debug,
            "eps": key.eps,
            "out_eps": key.out_eps,
            "onorm_eps": key.onorm_eps,
            "lb": key.lower_bound,
            "kda0": key.kda0,
            "xp": key.xp,
            "write_idx": key.write_idx,
            "gate_local": False,
            "put": mb.put,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "vals": al.vals.ptr,
            "xl": gl.xl.ptr,
            "fa": kl.fa.ptr,
            "qkl": kl.qkl.ptr,
            "decay": kl.decay.ptr,
            "vloc": kl.vloc.ptr,
            "ol": kl.ol.ptr,
            "beta": kl.beta.ptr,
            "xbuf": scratch + fx.Int64(lay["x"][0]),
            "args": {
                "prefix": prefix,
                "delta": delta,
                "blocks": blocks,
                "blk_sm": blk_sm,
                "blk_sr": blk_sr,
                "ares_nw": ares_nw,
                "ares_qk": ares_qk,
                "in_nw": in_nw,
                "w_in": w_in,
                "w_fb": w_fb,
                "conv_w": conv_w,
                "conv_st": conv_st,
                "cs_seq": cs_seq,
                "cs_dim": cs_dim,
                "cs_tok": cs_tok,
                "a_log": a_log,
                "dt_bias": dt_bias,
                "on_w": on_w,
                "rstate": rstate,
                "rs_seq": rs_seq,
                "st_idx": st_idx,
                "st_stride": st_stride,
                "num_acc": num_acc,
                "core_out": core_out,
                "x_dbg": x_dbg,
                "proj_dbg": proj_dbg,
            },
        }
        for region in ("part", "wgt", "msq", "xrdy", "proj", "qk", "onrm"):
            c[region] = sreg(scratch, lay[region][0], region)
        on, tls = key.timeline, lds.tls.ptr
        c["on"], c["tls"] = on, tls
        stamp_begin(on, tls, tid, TL_POINTS)
        if bid < NA:
            stage_slice(c, bid)
            stamp(on, tls, tid, 1)
            if bid < s:
                stage_gate(c, bid)
            stamp(on, tls, tid, 2)
            stage_mix(c, bid)
            stamp(on, tls, tid, 3)
        if bid >= NA:
            gi = bid - NA
            rg0 = 2 * gi
            rg1 = 2 * gi + 1
            if const_expr(GFX942):
                stage_gemv_pair(c, rg0, rg1)
            else:
                ops0 = gemv_loads(c, rg0, range(PER_WAVE))
                load_x(c)
                stamp(on, tls, tid, 4)
                # the second group's first half in flight under the first's MFMAs
                ops1 = gemv_loads(c, rg1, range(4))
                stage_gemv(c, rg0, ops0)
                stamp(on, tls, tid, 5)
                ops1 = ops1 + gemv_loads(c, rg1, range(4, PER_WAVE))
                stage_gemv(c, rg1, ops1)
            stamp(on, tls, tid, 6)
        kda_tasks(c, on, tls)
        stamp_flush(on, tls, tl, tid, bid, TL_POINTS)

    @flyc.jit
    def launch(
        prefix: Int64,
        delta: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        in_nw: Int64,
        w_in: Int64,
        w_fb: Int64,
        conv_w: Int64,
        conv_st: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        a_log: Int64,
        dt_bias: Int64,
        on_w: Int64,
        rstate: Int64,
        rs_seq: Int32,
        st_idx: Int64,
        st_stride: Int32,
        num_acc: Int64,
        core_out: Int64,
        x_dbg: Int64,
        proj_dbg: Int64,
        scratch: Int64,
        layer: Int32,
        epoch: Int64,
        tl: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kda_pre(
            prefix,
            delta,
            blocks,
            blk_sm,
            blk_sr,
            ares_nw,
            ares_qk,
            in_nw,
            w_in,
            w_fb,
            conv_w,
            conv_st,
            cs_seq,
            cs_dim,
            cs_tok,
            a_log,
            dt_bias,
            on_w,
            rstate,
            rs_seq,
            st_idx,
            st_stride,
            num_acc,
            core_out,
            x_dbg,
            proj_dbg,
            scratch,
            layer,
            epoch,
            tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(kda_pre, launch)
    _BUILDS[key] = launch
    return launch


def ptr(t):
    return 0 if t is None else t.data_ptr()


def kda_pre(
    key: KdaPreBuild,
    *,
    prefix,
    delta,
    blocks,
    ares_nw,
    ares_qk,
    in_nw,
    w_in,
    w_fb,
    conv_w,
    conv_state,
    a_log,
    dt_bias,
    on_w,
    rstate,
    st_idx,
    num_acc,
    core_out,
    scratch,
    layer,
    epoch,
    x_dbg=None,
    proj_dbg=None,
    tl=None,
):
    """Host launcher. ``conv_state`` [slots, dim, state_len] or a transposed view
    of [slots, state_len, dim] (strides read); ``rstate`` [slots, 12, 128, 128]
    fp32; ``st_idx`` [R, >= L] int32; ``scratch`` zeroed before the step's first
    layer (tags are a layer's); ``epoch`` a device int32 bumped once a step and
    ``layer`` this launch's slot below ``TAG_SLOTS``."""
    assert w_in.shape == (NPROJ, HIDDEN) and w_in.is_contiguous()
    assert rstate.dtype == torch.float32 and rstate[0].numel() == NH * HD * HD
    assert rstate[0].is_contiguous(), (
        "a slot's state must be dense; slots may be strided"
    )
    assert conv_state.dtype == torch.bfloat16
    f = build(key)
    f(
        prefix.data_ptr(),
        ptr(delta),
        blocks.data_ptr(),
        blocks.stride(0),
        blocks.stride(1),
        ares_nw.data_ptr(),
        ares_qk.data_ptr(),
        in_nw.data_ptr(),
        w_in.data_ptr(),
        w_fb.data_ptr(),
        conv_w.data_ptr(),
        conv_state.data_ptr(),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        a_log.data_ptr(),
        dt_bias.data_ptr(),
        on_w.data_ptr(),
        rstate.data_ptr(),
        rstate.stride(0),
        st_idx.data_ptr(),
        st_idx.stride(0),
        num_acc.data_ptr(),
        core_out.data_ptr(),
        ptr(x_dbg),
        ptr(proj_dbg),
        scratch.data_ptr(),
        layer,
        epoch.data_ptr(),
        ptr(tl),
        stream=torch.cuda.current_stream(),
    )
