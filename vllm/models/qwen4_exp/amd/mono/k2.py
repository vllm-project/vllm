# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2 ``layer_post``: one Flash-Next decoder layer from the attention's core
output (the out_proj / o_proj input, the same width on GDN and QSA layers) to
the layer's materialized multi-stream state, every TP reduction inside the
kernel. One launch a layer and rank, ``BLOCKS`` x ``THREADS``, S <= 8 rows:

    o_proj   (160 x 16 rows, K 3072 a rank; CTAs 0 .. 159): GEMV of the core
             rows -> bf16 partial -> every rank's AR
    combine  (CTAs 0 .. 19, ``hc.stage_combine``): AR in rank order -> attn
             out; state + attn 2 sigmoid(inj_a / 4) -> STATE1; MSQ
    hc down  (168 tasks, CTAs 88 .. 255) and hc up (160, CTAs 0 .. 159): the
             mlp_hyper_connection's mix of STATE1 -> BIN (block input), its
             injection logits -> INJ
    router   (32 x 16 experts, CTAs 160 .. 191): bf16 logits of BIN -> RLOG
    sgate    (CTAs 192 .. 192+S-1, token t): sigmoid(bf16(x . w_sg)) -> SGATE
    route    (every CTA, wave t token t): the 512 logits' top-10 in descending
             order, softmax-renormalized (``_route_mp``)
    ug       (|U| x 20 tasks, every CTA): the union U's w13 rows (the fused
             shared expert its last slot) against x's MXFP4 codes; silu(g) u
             -> INTER a pick
    down     (160 x 16 hidden rows, CTAs 0 .. 159): w2 of U against the picks'
             INTER through MXFP4, times the route weight, summed in top-k
             order -> bf16 partial -> every rank's FINAL
    final    (CTAs 0 .. 19): FINAL in rank order -> moe out; STATE1 + moe 2
             sigmoid(INJ / 4) -> ``out``; its sums of squares -> ``msq_out``
             (the next layer's front reads them)

The routed experts' numerics are the Qwen3.8 mono's (the stock one-stage
MXFP4 decode MoE: x and silu(g) u through MXFP4, RCEIL scales), with the
intermediate the weights hold padded from RI to RIP: the pad columns' w13 rows
are not run and their INTER codes are zero. The shared expert is fused as
expert 512 (``VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS``), weighted by
sigmoid(bf16(x . w_sg)) in fp32. Peer regions are double-buffered by the step
epoch's parity.

Variants: ``oq`` (an online-MXFP4 out_proj on the ASM FP4 GEMM: the core rows
through MXFP4, as ``per_1x32_f4_quant_hip`` quantizes the GEMM's input) and
``og`` (o_proj's input is core x sigmoid(gate), the QSA output gate).
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.kimi_k3.amd.mono.common.ops import (
    CM_DEV,
    bf16_pair,
    bf16_round,
    bf_hi,
    bf_lo,
    ld_i32,
    load_ptr64,
    row_sum,
    rsrc,
    traced,
    uniform,
    wave_sum,
    xred,
)
from vllm.models.kimi_k3.amd.mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    KernelAbi,
    key_tuple,
    pair_layout,
)
from vllm.models.kimi_k3.amd.mono.common.ranks import peer_bases
from vllm.models.kimi_k3.amd.mono.common.sync import Mailbox, preg, shift, sreg
from vllm.models.kimi_k3.amd.mono.stages import gemv
from vllm.models.qwen3_5.amd.mono.layer import (
    _route_mp,
    fp4_tile,
    load_route,
    mfma_f4,
    mx_scale,
    pk_fp4_f32,
    pow2,
    put_route,
    quant_x,
    scale_word,
    sigmoid,
    silu_mul,
    step_tag,
)
from vllm.models.qwen4_exp.amd.mono import fp4, hc
from vllm.models.qwen4_exp.amd.mono import layout as L
from vllm.models.qwen4_exp.amd.mono.layout import need
from vllm.models.qwen4_exp.amd.mono.sources import digest

H, HH, HC, ROWS = L.HIDDEN, L.HH, L.HC, L.ROWS
E, TOPK = L.E, L.TOPK
KT = TOPK + 1  # picks a token: the top-k + the fused shared expert
MAX_U = L.MAX_TOKENS * TOPK + 1
OTASKS = H // ROWS  # 160 o_proj / down row groups, CTAs 0 ..
D0 = BLOCKS - hc.DOWN_TASKS  # 88: the hc down CTAs, to the last
R_TASKS = E // ROWS  # 32
R0 = OTASKS  # 160: router CTAs
RT0 = R0 + R_TASKS  # 192: shared gate CTAs, a token each
PS0 = RT0 + L.MAX_TOKENS  # 200: the fused shared expert's early ug CTAs
PS_MIN = 4  # rows from which the fused shared expert's ug runs before the route
UG_AHEAD = 1  # ug tasks a CTA prefetches past the running one
DB = 4  # a wave's down slots whose w2 is in flight together
UG_COLS = 16
UG_STEPS = H // L.FP4_STEP // 4  # a wave's K quarter: 5 steps
TAG_SLOTS = 256
XROW = H + gemv.LDS_PAD

assert hc.UP_TASKS == OTASKS and D0 <= R0 and hc.NRED <= D0
assert E == THREADS, "load_route: an expert a thread"

_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class K2Build:
    tokens: int
    tp: int = 2
    eps: float = 1e-6  # the HC norms
    oq: bool = False  # out_proj MXFP4 (ASM FP4 GEMM layout)
    og: bool = False  # o_proj's input is core x sigmoid(gate)

    @property
    def u1(self) -> bool:
        """One row: U is the token's picks in top-k order (no union pass)."""
        return self.tokens == 1

    @property
    def ps(self) -> bool:
        """The fused shared expert's ug tasks run before the route."""
        return self.tokens >= PS_MIN


ABI = KernelAbi(
    (
        "core",
        "gate",
        "state",
        "inj_a",
        "w_o",
        "w_os",
        "hc_w",
        "w_down",
        "w_up",
        "w_gate",
        "w_sg",
        "w13",
        "w13s",
        "w2",
        "w2s",
        "out",
        "msq_out",
        "scratch",
        "peers",
        "rank",
        "epoch",
        "layer",
    )
)


def peer_pairs(s, tp):
    """One parity's peer regions, pairs: name -> count."""
    return {"ar": tp * s * H // 2, "final": tp * s * H // 2}


def peer_layout(s, tp):
    return pair_layout(peer_pairs(s, tp).items())


def peer_bytes(s, tp):
    lay = peer_layout(s, tp)
    return 2 * max(o + n for o, n in lay.values())


def irow(lay):
    """Bytes a pick's fp4 INTER row (RIP) takes in LDS (padded)."""
    return lay.RIP // 2 + 16


def iq_scales(lay, s):
    """Dword of LDS ``il`` where the fp4 INTER rows' e8m0 codes start."""
    return s * KT * irow(lay) // 4


def scratch_layout(key: K2Build) -> dict:
    s = key.tokens
    lay = L.Layout(key.tp)
    pairs = hc.regions(s) | {
        "inj": s * HC,
        "rlog": s * E // 2,
        "sgate": s,
        "route": s * 2 * TOPK,
        "inter": s * KT * lay.RI,
    }
    out = pair_layout(pairs.items())
    end = max(o + n for o, n in out.values())
    for name, n in (("state1", s * HH * 2), ("bin", hc.bin_bytes(s))):
        end = (end + 15) // 16 * 16
        out[name] = (end, n)
        end += n
    return out


def scratch_bytes(key: K2Build) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def f32_of(w):
    return w.bitcast(fx.Float32)


# ---------------------------------------------------------------- o_proj
def o_loads(c):
    """o_proj task bid's weights for this wave."""
    lane, wave, a = c["lane"], c["wave"], c["args"]
    k = c["lay"].CORE
    if const_expr(c["oq"]):
        return fp4.fp4_gemv_loads(lane, wave, a["w_o"], a["w_os"], c["bid"], k)
    return gemv.loads(lane, wave, a["w_o"], c["bid"] * ROWS, k)


@traced
def gated_rows_to_lds(c, k):
    """``og``: bf16(core x sigmoid(gate)) (fp32 math, as the compiled
    ``o * torch.sigmoid(gate)``) -> the o_proj input rows in LDS."""
    tid, s, xl = c["tid"], c["S"], c["xl"]
    a = c["args"]
    words = s * k // 8
    for i in range_constexpr((words + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, words - 1)
        t = e // (k // 8)
        kk = e % (k // 8) * 8
        at = (t * k + kk) // 2
        v = fx.Vector(bo.buffer_load(rsrc(a["core"]), at, vec_width=4, dtype=T.i32))
        g = fx.Vector(bo.buffer_load(rsrc(a["gate"]), at, vec_width=4, dtype=T.i32))
        y = fx.Vector.from_elements(
            [
                bf16_pair(
                    bf_lo(v[d]) * sigmoid(bf_lo(g[d])),
                    bf_hi(v[d]) * sigmoid(bf_hi(g[d])),
                ).bitcast(fx.Int32)
                for d in range(4)
            ],
            fx.Int32,
        )
        if tid + THREADS * i < words:
            fx.ptr_store(y, xl + (t * (k + gemv.LDS_PAD) + kk) // 2)
    gpu.barrier()


@traced
def run_oproj(c, ops):
    """o_proj rows 16 bid ..: the core rows (plain loads: the previous launch
    wrote them) -> LDS, GEMV -> bf16 partials -> every rank's AR region, this
    rank's slot."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    k = c["lay"].CORE
    xrow = k + gemv.LDS_PAD
    if const_expr(c["og"]):
        gated_rows_to_lds(c, k)
    else:
        gemv.rows_to_lds(tid, c["args"]["core"], k, s, c["xl"], xrow)
    if const_expr(c["oq"]):
        quant_x(c, k)
        acc = fp4.fp4_gemv(lane, wave, c["xl"], xrow, k, s, c["bid"], ops)
    else:
        acc = gemv.mfmas(lane, wave, c["xl"], xrow, k, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < s * (ROWS // 2):
        tt = tid // (ROWS // 2)
        rp = tid % (ROWS // 2)
        v0 = bf16_round(row_sum(red, 2 * rp, tt))
        v1 = bf16_round(row_sum(red, 2 * rp + 1, tt))
        at = (c["rank"] * s + tt) * H + c["bid"] * ROWS + 2 * rp
        for p in range_constexpr(c["TP"]):
            c["put_bf"](c["peer"]["ar"][p], at, [v0, v1])
    gpu.barrier()


# ---------------------------------------------------------------- combines
def inj_a_of(c):
    """The attention injection logits a previous launch wrote ([S, HC] bf16)."""

    def get(t):
        w = fx.Vector(
            bo.buffer_load(rsrc(c["args"]["inj_a"]), t * 2, vec_width=2, dtype=T.i32)
        )
        return [bf_lo(w[0]), bf_hi(w[0]), bf_lo(w[1]), bf_hi(w[1])]

    return get


def inj_m_of(c):
    """The mlp injection logits this launch's hc down put to INJ."""

    def get(t):
        got = c["poll"]([(c["inj"], t * HC, 2), (c["inj"], t * HC + 2, 2)])
        return [f32_of(got[k // 2][k % 2]) for k in range(HC)]

    return get


def stage_attn_combine(c, j):
    a = c["args"]
    hc.stage_combine(
        c,
        j,
        c["peer"]["ar_own"],
        inj_a_of(c),
        a["state"],
        0,
        c["state1"],
        CM_DEV,
        lambda jj, t, k, v: c["put"](c["hc"]["msq"], (jj * c["S"] + t) * HC + k, v),
    )


def stage_final(c, j):
    a = c["args"]

    def msq_put(jj, t, k, v):
        bo.buffer_store(v, rsrc(a["msq_out"]), (jj * c["S"] + t) * HC + k)

    hc.stage_combine(
        c,
        j,
        c["peer"]["final_own"],
        inj_m_of(c),
        c["state1"],
        CM_DEV,
        a["out"],
        0,
        msq_put,
    )


# ---------------------------------------------------------------- router / sgate
def router_loads(c, i):
    return gemv.loads(c["lane"], c["wave"], c["args"]["w_gate"], i * ROWS, H)


@traced
def stage_router(c, i, ops):
    """Experts 16 i ..: bf16 logits of the block-input rows in LDS -> RLOG."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    acc = gemv.mfmas(lane, wave, c["xl"], XROW, H, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < (ROWS // 2) * s:
        r = 2 * (tid % (ROWS // 2))
        t = tid // (ROWS // 2)
        vs = [bf16_round(row_sum(red, r + d, t)) for d in range(2)]
        c["put_bf"](c["rlog"], t * E + i * ROWS + r, vs)
    gpu.barrier()


@traced
def stage_sgate(c, t):
    """Token t's shared-expert weight: sigmoid(bf16(x . w_sg)) in fp32 (the
    top-k kernel's fused sigmoid on the combined gate's bf16 logit) -> SGATE."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    n = H // 8
    kk = fx.min(tid, n - 1) * 8
    xw = fx.Vector(
        fx.ptr_load(
            c["xl"] + (t * XROW + kk) // 2,
            result_type=fx.Vector.make_type(4, fx.Int32),
        )
    )
    ww = fx.Vector(bo.buffer_load(rsrc(a["w_sg"]), kk // 2, vec_width=4, dtype=T.i32))
    acc = fx.Float32(0.0)
    for d in range_constexpr(4):
        acc = acc + bf_lo(xw[d]) * bf_lo(ww[d]) + bf_hi(xw[d]) * bf_hi(ww[d])
    acc = wave_sum((tid < n).select(acc, fx.Float32(0.0)))
    if lane == 0:
        fx.ptr_store(acc, red + wave)
    gpu.barrier()
    if tid == 0:
        tot = fx.ptr_load(red + 0)
        for w in range_constexpr(1, WAVES):
            tot = tot + fx.ptr_load(red + w)
        c["put"](c["sgate"], t, sigmoid(bf16_round(tot)))
    gpu.barrier()


# ---------------------------------------------------------------- route
@traced
def route_token(c, t):
    """Token t on this wave: the 512 bf16 logits, 8 a lane (expert lane + 64 i;
    a NaN logit, a warmup's garbage row, ranks last) -> the top-10."""
    lane = c["lane"]
    per = E // 64
    specs = [(c["rlog"], (t * E + lane + 64 * i) // 2, 1) for i in range(per)]
    got = c["poll"](specs, batch=len(specs))
    odd = (lane % 2) == 1
    lgs = []
    for i in range_constexpr(per):
        w = got[i][0]
        lg = odd.select(bf_hi(w), bf_lo(w))
        lgs.append((lg == lg).select(lg, fx.Float32(-(2.0**127))))
    return _route_mp(c, lgs)


@traced
def local_route(c):
    """Every CTA routes its tokens itself, wave t token t, into its LDS route
    (``load_route``'s layout); CTA RT0 also puts ROUTE, for the tests. Waits
    on the last router group's last token first, so the full polls don't
    spin."""
    lane, wave, rt, kt, s = c["lane"], c["wave"], c["rt"], c["KT"], c["S"]
    if c["tid"] == 0:
        c["poll"]([(c["rlog"], (s * E - 2) // 2, 1)])
    gpu.barrier()
    if wave < s:
        rank, e, w = route_token(c, wave)
        if lane < TOPK:
            fx.ptr_store(e, rt["route"] + (wave * 2 * kt + rank))
            fx.ptr_store(w.bitcast(fx.Int32), rt["route"] + (wave * 2 * kt + kt + rank))
            if c["bid"] == RT0:
                put_route(c, wave, rank, e, w)


# ---------------------------------------------------------------- ug
def ug_loads(c, task, live, shared=False):
    """Task ``task``'s w13 operands for this wave (gate or up by wave % 2, K
    quarter wave / 2: 5 steps), a scale word a step; out of bounds (no memory
    access, zeros) where ``live`` is false. ``shared``: the fused shared
    expert's task ``task`` (no union yet)."""
    lane, wave = c["lane"], c["wave"]
    a, rt, lay = c["args"], c["rt"], c["lay"]
    groups = lay.RI // UG_COLS
    slot = task // groups
    g = task % groups
    kq = wave // 2
    rg = L.w13_group(lay, wave % 2, g)
    wb, sb = lay.RIP * H, 2 * lay.RIP * H // 32
    if const_expr(shared):
        e = 0
        r_w = rsrc(a["w13"] + fx.Int64(E * wb), live.select(fx.Int32(wb), fx.Int32(0)))
        r_s = rsrc(a["w13s"] + fx.Int64(E * sb), live.select(fx.Int32(sb), fx.Int32(0)))
    else:
        e = uniform(fx.ptr_load(rt["expert"] + slot))
        n_w, n_s = (E + 1) * wb, (E + 1) * sb
        r_w = rsrc(a["w13"], live.select(fx.Int32(n_w), fx.Int32(0)))
        r_s = rsrc(a["w13s"], live.select(fx.Int32(n_s), fx.Int32(0)))
    base = L.w13_base(lay, e)
    row = rg * ROWS + lane % ROWS
    sts = [kq * UG_STEPS + i for i in range(UG_STEPS)]
    tiles = [fp4_tile(r_w, L.fp4_tile_dword(base, rg, H, st, lane)) for st in sts]
    words = [
        scale_word(r_s, L.w13_scale_index(lay, e, row, 4 * st + lane // 16))
        for st in sts
    ]
    return tiles + words


@traced
def stage_ug(c, task, ops, shared=False):
    """Slot task / 20, columns 16 (task % 20) ..: the gate and up rows against
    the x rows' MXFP4 codes in LDS; silu(g) u -> fp32 -> INTER of each picking
    token (``shared``: every token's pick ``TOPK``)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    rt, lay = c["rt"], c["lay"]
    groups = lay.RI // UG_COLS
    slot = task // groups
    g = task % groups
    kq = wave // 2
    rg = L.w13_group(lay, wave % 2, g)
    tiles, words = ops[:UG_STEPS], ops[UG_STEPS:]
    t = fx.min(lane % 16, s - 1)
    xr = fp4.xq_row(H)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(UG_STEPS):
        st = kq * UG_STEPS + i
        code = (words[i] >> L.scale_code_shift(st, rg)) & 0xFF
        xq = (t * xr + st * (L.FP4_STEP // 2)) // 4 + 4 * (lane // 16)
        xb = fx.Vector(
            fx.ptr_load(c["xl"] + xq, result_type=fx.Vector.make_type(4, fx.Int32))
        )
        sb = fx.ptr_load(
            c["xl"] + (fp4.xq_scales(s, H) + t * (H // 32) + 4 * st + lane // 16)
        )
        acc = mfma_f4(tiles[i], xb, acc, code, sb)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < s * (UG_COLS // 2):
        tt = tid // (UG_COLS // 2)
        cp = tid % (UG_COLS // 2)
        if const_expr(shared):
            k = fx.Int32(TOPK)
        else:
            k = fx.ptr_load(rt["kof"] + (slot * 8 + tt))
        v = [
            silu_mul(
                row_sum(red, 2 * cp + d, tt, waves=(0, 2, 4, 6)),
                row_sum(red, 2 * cp + d, tt, waves=(1, 3, 5, 7)),
            ).bitcast(fx.Int32)
            for d in range(2)
        ]
        if k >= 0:
            pick = tt * KT + k
            c["put_words"](c["inter"], pick * lay.RI + g * UG_COLS + 2 * cp, v)
    gpu.barrier()


# ---------------------------------------------------------------- down
@traced
def load_inter(c):
    """Every pick's INTER (fp32, RI columns) through MXFP4 by 32 columns (RCEIL,
    a block's 16 column pairs on 16 lanes) -> LDS fp4 rows of RIP, the pad
    columns zero codes; then the e8m0 codes. One poll batch."""
    s, tid, lay = c["S"], c["tid"], c["lay"]
    ri, rip = lay.RI, lay.RIP
    ir = irow(lay)
    nb = rip // 32
    n = s * KT * ri // 2
    per = (n + THREADS - 1) // THREADS
    got = c["poll"](
        [(c["inter"], 2 * fx.min(tid + THREADS * i, n - 1), 2) for i in range(per)]
    )
    for i in range_constexpr(per):
        p = fx.min(tid + THREADS * i, n - 1)
        pick = p // (ri // 2)
        cp = p % (ri // 2)
        h0, h1 = f32_of(got[i][0]), f32_of(got[i][1])
        amax = fx.max(fx.Float32(fmath.absf(h0)), fx.Float32(fmath.absf(h1)))
        for off in range_constexpr(4):
            amax = xred(amax, 1 << off, fx.max)
        e = mx_scale(amax)
        v = pk_fp4_f32(fx.Int32(0), h0, h1, pow2(e), 0)
        v = v << (8 * (cp % 4))
        for off in range_constexpr(2):
            v = xred(v, 1 << off, lambda a, b: a | b)
        ok = tid + THREADS * i < n
        if ok & (cp % 4 == 0):
            fx.ptr_store(v, c["il"] + (pick * ir + cp) // 4)
        if ok & (cp % 16 == 0):
            fx.ptr_store(e, c["il"] + (iq_scales(lay, s) + pick * nb + cp // 16))
    pw = (rip - ri) // 8  # pad dwords a pick
    pb = (rip - ri) // 32  # pad blocks a pick
    for i in range_constexpr((s * KT * pw + THREADS - 1) // THREADS):
        j = tid + THREADS * i
        if j < s * KT * pw:
            fx.ptr_store(
                fx.Int32(0), c["il"] + ((j // pw) * ir + ri // 2) // 4 + j % pw
            )
    if tid < s * KT * pb:
        fx.ptr_store(
            fx.Int32(127),
            c["il"] + (iq_scales(lay, s) + (tid // pb) * nb + ri // 32 + tid % pb),
        )
    gpu.barrier()


def down_loads(c, rgd, nu, b0):
    """The w2 tiles and scale words of a wave's batch from slot b0 (slots b0,
    b0 + 8, ...; past |U| the last slot's)."""
    lane, a, rt, lay = c["lane"], c["args"], c["rt"], c["lay"]
    r_w, r_s = rsrc(a["w2"]), rsrc(a["w2s"])
    row = rgd * ROWS + lane % ROWS
    w2_steps = lay.RIP // L.FP4_STEP
    tiles, words = [], []
    for j in range(DB):
        sl = fx.min(b0 + WAVES * j, nu - 1)
        e = uniform(fx.ptr_load(rt["expert"] + sl))
        base = L.w2_base(lay, e)
        tiles += [
            fp4_tile(r_w, L.fp4_tile_dword(base, rgd, lay.RIP, st, lane))
            for st in range(w2_steps)
        ]
        words += [
            scale_word(r_s, L.w2_scale_index(lay, e, row, 4 * st + lane // 16))
            for st in range(0, w2_steps, 2)
        ]
    return tiles + words


@traced
def stage_down(c, rgd, nu, ops0):
    """Hidden rows 16 rgd ..: every slot's w2 against its picks' INTER (wave w
    the slots w, w + 8, ...; a batch in flight under the last one's MFMAs) ->
    fp32 (t, k) contributions times the route weight in LDS -> summed in top-k
    order -> bf16 partial -> every rank's FINAL."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rt, kt, lay = c["rt"], c["KT"], c["lay"]
    w2_steps = lay.RIP // L.FP4_STEP
    nw = (w2_steps + 1) // 2
    ir = irow(lay)
    t = fx.min(lane % 16, s - 1)
    nt = DB * w2_steps
    for i in range_constexpr((s * kt * ROWS + THREADS - 1) // THREADS):
        j = tid + THREADS * i
        if j < s * kt * ROWS:
            fx.ptr_store(fx.Float32(0.0), c["contrib"] + j)
    gpu.barrier()
    b0 = wave
    ops = ops0
    while b0 < nu:
        nb = b0 + WAVES * DB
        ops_n = down_loads(c, rgd, nu, fx.min(nb, nu - 1))
        for j in range_constexpr(DB):
            sl = fx.min(b0 + WAVES * j, nu - 1)
            live = (b0 + WAVES * j) < nu
            k = fx.ptr_load(rt["kof"] + (sl * 8 + t))
            picked = (k >= 0) & (lane % 16 < s)
            pick = t * kt + fx.max(k, 0)
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for st in range_constexpr(w2_steps):
                word = ops[nt + j * nw + st // 2]
                code = (word >> L.scale_code_shift(st, rgd)) & 0xFF
                xq = (pick * ir + st * (L.FP4_STEP // 2)) // 4 + 4 * (lane // 16)
                xb = fx.Vector(
                    fx.ptr_load(
                        c["il"] + xq, result_type=fx.Vector.make_type(4, fx.Int32)
                    )
                )
                xb = fx.Vector.from_elements(
                    [picked.select(xb[d], fx.Int32(0)) for d in range(4)], fx.Int32
                )
                sb = fx.ptr_load(
                    c["il"]
                    + (iq_scales(lay, s) + pick * (lay.RIP // 32) + 4 * st + lane // 16)
                )
                acc = mfma_f4(ops[j * w2_steps + st], xb, acc, code, sb)
            if live & picked:
                wt = f32_of(fx.ptr_load(rt["route"] + (t * 2 * kt + kt + fx.max(k, 0))))
                for i in range_constexpr(4):
                    row = 4 * (lane // 16) + i
                    fx.ptr_store(
                        acc[i] * wt,
                        c["contrib"] + ((t * kt + fx.max(k, 0)) * ROWS + row),
                    )
        b0 = nb
        ops = ops_n
    gpu.barrier()
    if tid < (ROWS // 2) * s:
        rp = tid % (ROWS // 2)
        tt = tid // (ROWS // 2)
        ys = []
        for d in range_constexpr(2):
            r = 2 * rp + d
            y = fx.Float32(0.0)
            for k in range_constexpr(kt):
                y = y + fx.ptr_load(c["contrib"] + ((tt * kt + k) * ROWS + r))
            ys.append(bf16_round(y))
        at = (c["rank"] * s + tt) * H + rgd * ROWS + 2 * rp
        for p in range_constexpr(c["TP"]):
            c["put_bf"](c["peer"]["final"][p], at, ys)
    gpu.barrier()


# ---------------------------------------------------------------- kernel
_BUILDS: dict = {}


def build(key: K2Build):
    if key in _BUILDS:
        return _BUILDS[key]
    s = key.tokens
    need(1 <= s <= L.MAX_TOKENS, f"{s} rows, the kernels take 1..{L.MAX_TOKENS}")
    lay = L.Layout(key.tp)
    tp = key.tp
    scr = scratch_layout(key)
    # every width at the widest's offsets: a step's parity half never overlaps
    # the other half of a step of another width
    play = peer_layout(L.MAX_TOKENS, tp)
    half = peer_bytes(L.MAX_TOKENS, tp) // 2
    xwords = s * (max(lay.CORE, H, hc.DK) + gemv.LDS_PAD) // 2
    need(PS0 + lay.RI // UG_COLS <= BLOCKS, "the shared expert's early ug CTAs")
    need(lay.RI % 32 == 0 and lay.RIP % L.FP4_STEP == 0, "the routed intermediate")

    @fx.struct
    class RouteLds:
        route: fx.Array[fx.Int32, s * 2 * KT, 16]
        flag: fx.Array[fx.Int32, E, 16]
        expert: fx.Array[fx.Int32, MAX_U, 16]
        kof: fx.Array[fx.Int32, MAX_U * 8, 16]
        scan: fx.Array[fx.Int32, WAVES, 16]

    @fx.struct
    class XLds:
        xl: fx.Array[fx.Int32, xwords, 16]

    @fx.struct
    class DownLds:
        il: fx.Array[fx.Int32, s * KT * (irow(lay) // 4 + lay.RIP // 32), 16]
        contrib: fx.Array[fx.Float32, s * KT * ROWS, 16]

    @fx.union
    class StageLds:
        x: XLds
        down: DownLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        rs: fx.Array[fx.Float32, L.MAX_TOKENS * HC, 16]
        ll: fx.Array[fx.Int32, s * hc.LROW // 2, 16]

    keyed = key_tuple(key, digest())
    name = f"flashnext_mono_k2_s{s}_t{tp}_o{int(key.oq)}_a{int(key.og)}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k2(
        core: Int64,
        gate: Int64,
        state: Int64,
        inj_a: Int64,
        w_o: Int64,
        w_os: Int64,
        hc_w: Int64,
        w_down: Int64,
        w_up: Int64,
        w_gate: Int64,
        w_sg: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        out: Int64,
        msq_out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        rl = alloc.allocate(RouteLds).peek()
        st_lds = alloc.allocate(StageLds)
        xl, dl = st_lds.x.peek(), st_lds.down.peek()
        mb = Mailbox(step_tag(epoch, layer) - 1)
        par = uniform(ld_i32(epoch, 0)) % 2
        bases = peer_bases(peers, tp)
        own = load_ptr64(peers, rank)
        peer = {}
        for region in ("ar", "final"):
            off = play[region][0]
            peer[region] = [
                shift(preg(b, off, region), fx.Int64(par) * half) for b in bases
            ]
            peer[region + "_own"] = shift(preg(own, off, region), fx.Int64(par) * half)
        c = {
            "S": s,
            "TP": tp,
            "lay": lay,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "rank": rank,
            "eps": key.eps,
            "fse": True,
            "oq": key.oq,
            "og": key.og,
            "u1": key.u1,
            "KT": KT,
            "put": mb.put,
            "put_bf": mb.put_bf,
            "put_words": mb.put_words,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "rs": lds.rs.ptr,
            "ll": lds.ll.ptr,
            "peer": peer,
            "xl": xl.xl.ptr,
            "il": dl.il.ptr,
            "contrib": dl.contrib.ptr,
            "rt": {
                "route": rl.route.ptr,
                "flag": rl.flag.ptr,
                "expert": rl.expert.ptr,
                "kof": rl.kof.ptr,
                "scan": rl.scan.ptr,
            },
            "state1": scratch + fx.Int64(scr["state1"][0]),
            "args": {
                "core": core,
                "gate": gate,
                "state": state,
                "inj_a": inj_a,
                "w_o": w_o,
                "w_os": w_os,
                "w_gate": w_gate,
                "w_sg": w_sg,
                "w13": w13,
                "w13s": w13s,
                "w2": w2,
                "w2s": w2s,
                "out": out,
                "msq_out": msq_out,
            },
        }
        for region in ("inj", "rlog", "sgate", "route", "inter"):
            c[region] = sreg(scratch, scr[region][0], region)
        c["hc"] = {
            "state": c["state1"],
            "state_cm": CM_DEV,
            "w_norm": hc_w,
            "w_down": w_down,
            "w_up": w_up,
            "msq": sreg(scratch, scr["msq"][0], "msq"),
            "msq_in": None,
            "dpart": sreg(scratch, scr["dpart"][0], "dpart"),
            "lora": sreg(scratch, scr["lora"][0], "lora"),
            "binf": sreg(scratch, scr["binf"][0], "binf"),
            "bin": scratch + fx.Int64(scr["bin"][0]),
            "inj_out": lambda t, r, vs: _put_inj(c, t, r, vs),
        }
        is_r = (bid >= R0) & (bid < RT0)
        is_g = (bid >= RT0) & (bid < RT0 + s)
        # ---- o_proj, the attention combine, the mlp HC mix, router / sgate:
        # a CTA's whole sequence in one branch, its weights in flight first
        if bid < D0:
            oops = o_loads(c)
            run_oproj(c, oops)
            if bid < hc.NRED:
                stage_attn_combine(c, bid)
            uops = hc.up_loads(c, bid)
            hc.load_rrms(c)
            hc.stage_up(c, bid, uops)
        if (bid >= D0) & (bid < OTASKS):
            oops = o_loads(c)
            dops = hc.down_loads(c, bid - D0)
            run_oproj(c, oops)
            uops = hc.up_loads(c, bid)
            hc.load_rrms(c)
            hc.stage_down(c, bid - D0, dops)
            hc.stage_up(c, bid, uops)
        if is_r:
            dops = hc.down_loads(c, bid - D0)
            rops = router_loads(c, bid - R0)
            hc.load_rrms(c)
            hc.stage_down(c, bid - D0, dops)
            hc.bin_to_lds(c, XROW)
            stage_router(c, bid - R0, rops)
        if bid >= RT0:
            dops = hc.down_loads(c, bid - D0)
            hc.load_rrms(c)
            hc.stage_down(c, bid - D0, dops)
            if is_g:
                hc.bin_to_lds(c, XROW)
                stage_sgate(c, bid - RT0)
        # ---- the routed experts: x in and quantized while the logits come
        # in; the fused shared expert's tasks need no route
        groups = lay.RI // UG_COLS
        is_ps = (bid >= PS0) & (bid < PS0 + groups)
        if const_expr(key.ps):
            sops = ug_loads(c, fx.min(fx.max(bid - PS0, 0), groups - 1), is_ps, True)
        if ~(is_r | is_g):
            hc.bin_to_lds(c, XROW)
        fp4.quant_x_fp4(c, H)
        if const_expr(key.ps):  # noqa: SIM102 (traced: `and` does not combine traced values)
            if is_ps:
                stage_ug(c, bid - PS0, sops, True)
        local_route(c)
        nu = load_route(c)
        total = (nu - int(key.ps)) * groups
        last = total - 1
        # reversed placement: the round's leftover tasks go to the high CTAs,
        # past the down / final ones; a prefetch past the last task loads nothing
        task = BLOCKS - 1 - bid

        def task_loads(n):
            return ug_loads(c, fx.min(n, last), n <= last)

        ops = [task_loads(task + BLOCKS * j) for j in range(UG_AHEAD)]
        while task < total:
            ops_n = task_loads(task + BLOCKS * UG_AHEAD)
            stage_ug(c, task, ops[0])
            task = task + BLOCKS
            ops = ops[1:] + [ops_n]
        if bid < OTASKS:
            dops = down_loads(c, bid, nu, fx.min(c["wave"], nu - 1))
            load_inter(c)
            stage_down(c, bid, nu, dops)
        if bid < hc.NRED:
            stage_final(c, bid)

    @flyc.jit
    def launch(
        core: Int64,
        gate: Int64,
        state: Int64,
        inj_a: Int64,
        w_o: Int64,
        w_os: Int64,
        hc_w: Int64,
        w_down: Int64,
        w_up: Int64,
        w_gate: Int64,
        w_sg: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        out: Int64,
        msq_out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k2(
            core,
            gate,
            state,
            inj_a,
            w_o,
            w_os,
            hc_w,
            w_down,
            w_up,
            w_gate,
            w_sg,
            w13,
            w13s,
            w2,
            w2s,
            out,
            msq_out,
            scratch,
            peers,
            rank,
            epoch,
            layer,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(k2, launch)
    _BUILDS[key] = launch
    return launch


def _put_inj(c, t, r, vs):
    """The mlp injection rows r, r + 1 of token t -> INJ (fp32 of bf16)."""
    c["put"](c["inj"], t * HC + r, vs[0])
    c["put"](c["inj"], t * HC + r + 1, vs[1])


def layer_post(
    key: K2Build,
    *,
    core,
    state,
    inj_a,
    w_o,
    hc_w,
    w_down,
    w_up,
    w_gate,
    w_sg,
    w13,
    w13s,
    w2,
    w2s,
    out,
    msq_out,
    scratch,
    peers,
    rank,
    epoch,
    layer,
    w_os=None,
    gate=None,
):
    """Host launcher; every weight as vLLM holds it after loading, the experts'
    shuffled (``layout``) with the shared expert fused as expert 512. ``state``
    / ``out``: the multi-stream state [S, 4 H] before the layer's attention
    combine and after its mlp combine; ``inj_a`` [S, 4] bf16: the attention
    HC's injection logits; ``msq_out`` [20, S, 4] fp32: ``out``'s sums of
    squares for the next layer's front. ``w_os``: an fp4 o_proj's scales
    (``key.oq``); ``gate``: the attention output gate (``key.og``), core's
    shape. ``peers``: the int64 table of every rank's peer buffer (at least
    ``peer_bytes(MAX_TOKENS, tp)`` each)."""
    s = key.tokens
    lay = L.Layout(key.tp)
    bf = torch.bfloat16
    fp4s = (torch.uint8, torch.float4_e2m1fn_x2)
    ne = E + 1
    k = lay.CORE
    need(core.shape == (s, k), f"core: [{s}, {k}]")
    need(state.shape == (s, HH) and out.shape == (s, HH), "state / out: [S, 4 H]")
    need(inj_a.shape == (s, HC) and inj_a.dtype == bf, "inj_a: bf16 [S, 4]")
    need(
        msq_out.numel() == hc.NRED * s * HC and msq_out.dtype == torch.float32,
        "msq_out: fp32 [20, S, 4]",
    )
    need(
        (gate is not None) == key.og and (gate is None or gate.shape == core.shape),
        "gate: core's shape, with og only",
    )
    if key.oq:
        need(
            w_o.shape == (H, k // 2) and w_o.dtype in fp4s,
            f"o_proj: fp4x2 [{H}, {k // 2}]",
        )
        need(
            w_os is not None
            and w_os.numel() == H * k // 32
            and w_os.element_size() == 1,
            "o_proj scales: e8m0, one a 32-block",
        )
    else:
        need(w_o.shape == (H, k) and w_o.dtype == bf, f"o_proj: bf16 [{H}, {k}]")
    need(hc_w.shape == (HH,), "mlp HC norm weight: [4 H]")
    need(w_down.shape == (L.DOWN_ROWS, HH), "mlp HC down: [336, 4 H]")
    need(w_up.shape == (HH, L.LORA), "mlp HC up: [4 H, 320]")
    need(w_gate.shape == (E, H) and w_sg.numel() == H, "router shape")
    need(
        w13.shape == (ne, 2 * lay.RIP, H // 2) and w13.dtype in fp4s,
        f"w13: fp4x2 [{ne}, {2 * lay.RIP}, {H // 2}]",
    )
    need(
        w2.shape == (ne, H, lay.RIP // 2) and w2.dtype in fp4s,
        f"w2: fp4x2 [{ne}, {H}, {lay.RIP // 2}]",
    )
    need(
        w13s.numel() == ne * 2 * lay.RIP * L.scale_cols(H) and w13s.element_size() == 1,
        "w13 scales: e8m0, one a 32-block, shuffled",
    )
    need(
        w2s.numel() == ne * H * L.scale_cols(lay.RIP) and w2s.element_size() == 1,
        "w2 scales: e8m0, one a 32-block, shuffled (K blocks padded to 8)",
    )
    used = [w_o, w_os, w_gate, w_sg, w13, w13s, w2, w2s, gate]
    tensors = [core, state, inj_a, out, msq_out, hc_w, w_down, w_up]
    tensors += [w for w in used if w is not None]
    need(all(t.is_contiguous() for t in tensors), "a non-contiguous input")
    dense = [core, state, out, hc_w, w_down, w_up, w_gate, w_sg]
    dense += [gate] if gate is not None else []
    need(all(t.dtype == bf for t in dense), "activations and dense weights: bf16")

    def ptr(t):
        return 0 if t is None else t.data_ptr()

    f = build(key)
    f(
        *ABI.pack(
            {
                "core": core.data_ptr(),
                "gate": ptr(gate),
                "state": state.data_ptr(),
                "inj_a": inj_a.data_ptr(),
                "w_o": w_o.data_ptr(),
                "w_os": ptr(w_os) if key.oq else 0,
                "hc_w": hc_w.data_ptr(),
                "w_down": w_down.data_ptr(),
                "w_up": w_up.data_ptr(),
                "w_gate": w_gate.data_ptr(),
                "w_sg": w_sg.data_ptr(),
                "w13": w13.data_ptr(),
                "w13s": w13s.data_ptr(),
                "w2": w2.data_ptr(),
                "w2s": w2s.data_ptr(),
                "out": out.data_ptr(),
                "msq_out": msq_out.data_ptr(),
                "scratch": scratch.data_ptr(),
                "peers": peers.data_ptr(),
                "rank": rank,
                "epoch": epoch.data_ptr(),
                "layer": layer,
            }
        ),
        stream=torch.cuda.current_stream(),
    )
