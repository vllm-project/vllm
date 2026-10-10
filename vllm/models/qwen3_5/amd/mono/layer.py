# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2 ``layer_post``: one Qwen3.8 decoder layer from the attention's core
output (the out_proj / o_proj input, the same shape on GDN and full-attention
layers) to the layer's MoE output, every TP reduction inside the kernel.

One launch a layer and rank, ``BLOCKS`` x ``THREADS``, S <= 8 decode rows:

    o_proj   (512 x 16 rows, K 2048 a rank; every CTA, tasks bid, bid + 256):
             bf16 GEMV of the core rows -> bf16 partial -> every rank's AR
    norm     (CTAs 0 .. 63, 128 columns each): the ranks' partials in rank
             order (fp32 -> bf16), + residual (bf16) -> ``res_out``; every
             task's sum of squares -> post_attention_layernorm (Gemma, 1 + w)
             -> x rows (plain, XRDY a task)
    router   (CTAs 64 .. 79: 4 x 16 of this rank's 64 experts x K quarters):
             fp32 parts -> bf16 logits -> every rank's RLOG
    sgu      (CTAs 80 .. 111: 32 x 16 rows of the shared gate | up) -> SGU
    route    (CTAs 112 .. 112+S-1, token t): sigmoid(x . w_sg) -> SGATE; the
             512 logits' top-10 (radix select; ties: the lower expert), in
             descending order, softmax-renormalized -> ROUTE
    ug       (|U| x 16, every CTA): the union U of the step's picks; an
             expert's 16 gate + 16 up rows (fp4 -> bf16 by their e8m0 scales)
             against the x rows through MXFP4; silu(g) u (fp32) -> INTER a pick
    down     (512 x 16 hidden rows, every CTA): w2 of every expert of U
             against its picks' INTER through MXFP4, times the route weight,
             summed in top-k
             order; + the shared down rows of silu(g) u (SGU) times SGATE ->
             bf16 partial -> every rank's FINAL
    final    (CTAs 0 .. 63, 128 columns): FINAL in rank order -> bf16 ``out``

The routed experts quantize their activations as the stock decode path does
(AITER's one-stage ``fmoe_bf16_pertokenMXfp4_g1u1_flat`` below 32 rows takes bf16
in, but runs x and silu(g) u through MXFP4, 1 x 32 blocks, RCEIL scales); the
quantized values are exact in bf16, so the MFMAs stay bf16. Their weights are
``AITER_MXFP4_MXFP4``'s shuffled ones (``layout``). Peer regions are
double-buffered by the step epoch's parity.
"""

import math
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T, as_ir_value

from vllm.models.kimi_k3.amd.mono.common.debug import region_ids
from vllm.models.kimi_k3.amd.mono.common.ops import (
    CM_DEV,
    CM_NT,
    ballot,
    bf16_pair,
    bf16_round,
    bf_hi,
    bf_lo,
    block_excl_scan,
    div_rn,
    hw_exp2,
    hw_rsq,
    lanes_below,
    ld_i32,
    load_ptr64,
    mfma_bf16,
    popcount,
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
from vllm.models.kimi_k3.amd.mono.common.ranks import peer_bases, sum_partials
from vllm.models.kimi_k3.amd.mono.common.sync import Mailbox, preg, publish, shift, sreg
from vllm.models.kimi_k3.amd.mono.stages import gemv
from vllm.models.qwen3_5.amd.mono import layout as L
from vllm.models.qwen3_5.amd.mono.layout import need
from vllm.models.qwen3_5.amd.mono.sources import digest

TP, HIDDEN, ROWS = L.TP, L.HIDDEN, L.ROWS
E, TOPK, RI, SI, EPR = L.E, L.TOPK, L.RI, L.SI, L.EPR
OK = L.CORE  # o_proj K a rank: 2048
OTASKS = HIDDEN // ROWS  # 512
COLS = 128  # columns a norm / final task
NRED = HIDDEN // COLS  # 64
R_GROUPS = EPR // ROWS  # 4
R_PARTS = 4  # K quarters: the router is on the path to the route
R_K = HIDDEN // R_PARTS
R_TASKS = R_GROUPS * R_PARTS  # 16
G_TASKS = 2 * SI // ROWS  # 32
R0 = NRED  # 64: router CTAs
G0 = R0 + R_TASKS  # 80: shared gate | up CTAs
RT0 = G0 + G_TASKS  # 112: route CTAs
UG_COLS = 16  # intermediate columns an ug task
UG_GROUPS = RI // UG_COLS  # 16
W13_STEPS = HIDDEN // L.FP4_STEP  # 64
UG_STEPS = W13_STEPS // 4  # a wave's K quarter: 16 steps
W2_STEPS = RI // L.FP4_STEP  # 2
DOWN_BATCH = 4  # a wave's experts whose w2 is in flight together
MAX_U = L.MAX_U
LOG2E = 1.0 / math.log(2.0)
TAG_SLOTS = 256

assert RT0 + L.MAX_TOKENS <= BLOCKS and OTASKS == 2 * BLOCKS and L.BLOCKS == BLOCKS

REGIONS = (
    "msq",
    "xrdy",
    "rpart",
    "sgu",
    "sgate",
    "route",
    "inter",
    "ar",
    "rlog",
    "final",
)

_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class K2Build:
    tokens: int
    eps: float = 1e-6  # post_attention_layernorm
    diag: bool = False  # bounded waits recording into ``diag`` (debug)
    # debug: run phases < stop only (1 o_proj, 2 norm, 3 router / route,
    # 4 ug, 5 down)
    stop: int = 99


ABI = KernelAbi(
    (
        "core",
        "residual",
        "w_o",
        "ln_w",
        "w_gate",
        "w_sg",
        "w_sgu",
        "w_sd",
        "w13",
        "w13s",
        "w2",
        "w2s",
        "out",
        "res_out",
        "scratch",
        "peers",
        "rank",
        "epoch",
        "layer",
        "diag",
    )
)


def peer_pairs(s):
    """One parity's peer regions, pairs: name -> count."""
    return {
        "ar": TP * s * HIDDEN // 2,
        "rlog": TP * s * EPR,
        "final": TP * s * HIDDEN // 2,
    }


def peer_layout(s):
    return pair_layout(peer_pairs(s).items())


def peer_bytes(s):
    lay = peer_layout(s)
    return 2 * max(o + n for o, n in lay.values())


def scratch_layout(key: K2Build) -> dict:
    s = key.tokens
    pairs = {
        "msq": NRED * s,
        "xrdy": NRED,
        "rpart": R_TASKS * s * ROWS,
        "sgu": s * SI,  # 2 SI bf16 a token, two a pair
        "sgate": s,
        "route": s * 2 * TOPK,
        "inter": s * TOPK * RI,
    }
    lay = pair_layout(pairs.items())
    end = max(o + n for o, n in lay.values())
    lay["xrow"] = ((end + 15) // 16 * 16, s * HIDDEN * 2)
    return lay


def scratch_bytes(key: K2Build) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def step_tag(epoch, slot):
    """This launch's mailbox tag: the step's epoch (bumped once a step, in the
    graph) and the launch's slot, so no pair of another step or launch passes."""
    e = uniform(ld_i32(epoch, 0))
    return (e % (1 << 22)) * TAG_SLOTS + slot + 1


def f32_of(w):
    return w.bitcast(fx.Float32)


def ld_bf(ptr, i):
    return fx.Float32(
        fx.BFloat16(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.bf16))
    )


def sigmoid(z):
    return fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(z * fx.Float32(-LOG2E)))


def silu_mul(g, u):
    return g * sigmoid(g) * u


def pow2(code):
    """The fp32 power of two an e8m0 code stands for."""
    return (code << 23).bitcast(fx.Float32)


def mx_scale(amax):
    """A 1 x 32 block's MXFP4 e8m0 code from its abs max: ceil_pow2(amax / 6)
    (RCEIL, AITER's activation quant), clamped to codes 1 .. 253."""
    d = div_rn(amax, fx.Float32(6.0), fx.Float32(1.0 / 6.0))
    bits = d.bitcast(fx.Int32)
    e = (bits >> 23) & 0xFF
    e = e + ((bits & 0x7FFFFF) != 0).select(fx.Int32(1), fx.Int32(0))
    return fx.min(fx.max(e, fx.Int32(1)), fx.Int32(253))


def mx_qdq(v, code):
    """An fp32 through e2m1 at scale ``code`` (nearest even, saturating at 6)
    and back: a power of two times an e2m1 value, exact in bf16."""
    a = fx.min(fx.Float32(fmath.absf(v * pow2(fx.Int32(254) - code))), fx.Float32(6.0))
    q = (a < fx.Float32(2.0)).select(
        fx.Float32(fmath.roundeven(a * fx.Float32(2.0))) * fx.Float32(0.5),
        (a < fx.Float32(4.0)).select(
            fx.Float32(fmath.roundeven(a)),
            fx.Float32(fmath.roundeven(a * fx.Float32(0.5))) * fx.Float32(2.0),
        ),
    )
    return (v < fx.Float32(0.0)).select(-q, q) * pow2(code)


def fp4x8_bf16(word, scale):
    """The 8 fp4 of ``word`` times ``scale`` -> an MFMA operand (8 bf16 as 4 i32,
    nibble order: element 2 j, 2 j + 1 from byte j)."""
    out = []
    for j in range(4):
        v = rocdl.cvt_scalef32_pk_bf16_fp4(
            T.vec(2, T.bf16),
            as_ir_value(fx.Int32(word)),
            as_ir_value(fx.Float32(scale)),
            j,
        )
        out.append(fx.Vector(v).bitcast(fx.Int32)[0])
    return fx.Vector.from_elements(out, fx.Int32)


def fp4_mfmas(tile, scale, xl, xoff, acc):
    """One 128-K step: a lane's 16 B of fp4 (4 dwords, K 32 (lane / 16) + 8 q ..
    for dword q) against the bf16 B rows at LDS ``xl`` + ``xoff`` (the same K)."""
    for q in range_constexpr(4):
        xb = fx.Vector(
            fx.ptr_load(
                xl + (xoff + 8 * q) // 2,
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
        )
        a = fp4x8_bf16(tile[q], scale)
        acc = mfma_bf16(a.bitcast(fx.BFloat16), xb.bitcast(fx.BFloat16), acc)
    return acc


def fp4_tile(r_w, at):
    return fx.Vector(
        bo.buffer_load(r_w, at, vec_width=4, dtype=T.i32, cache_modifier=CM_NT)
    )


def scale_word(r_s, byte_index):
    return fx.Int32(bo.buffer_load(r_s, byte_index // 4, vec_width=1, dtype=T.i32))


# ---------------------------------------------------------------- o_proj
@traced
def stage_o(c, rg, ops):
    """Rows 16 rg ..: GEMV of the LDS core rows -> bf16 partials -> every rank's
    AR region, this rank's slot (thread: token, row pair)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    acc = gemv.mfmas(lane, wave, c["xl"], OK + gemv.LDS_PAD, OK, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < s * (ROWS // 2):
        tt = tid // (ROWS // 2)
        rp = tid % (ROWS // 2)
        v0 = bf16_round(row_sum(red, 2 * rp, tt))
        v1 = bf16_round(row_sum(red, 2 * rp + 1, tt))
        at = (c["rank"] * s + tt) * HIDDEN + rg * ROWS + 2 * rp
        for p in range_constexpr(TP):
            c["put_bf"](c["peer"]["ar"][p], at, [v0, v1])
    gpu.barrier()


def o_loads(c, j):
    """o_proj task bid + 256 j's weights for this wave."""
    return gemv.loads(
        c["lane"], c["wave"], c["args"]["w_o"], (c["bid"] + BLOCKS * j) * ROWS, OK
    )


@traced
def run_oproj(c, oops):
    """This CTA's o_proj tasks (bid, bid + 256) on the core rows (plain loads:
    the previous launch wrote them), their weights ``oops`` already in flight."""
    core = c["args"]["core"]
    gemv.rows_to_lds(c["tid"], core, OK, c["S"], c["xl"], OK + gemv.LDS_PAD)
    for j in range_constexpr(OTASKS // BLOCKS):
        stage_o(c, c["bid"] + BLOCKS * j, oops[j])


# ---------------------------------------------------------------- norm
@traced
def stage_norm(c, j):
    """Columns 128 j ..: the AR sum (rank order, fp32 -> bf16) + residual (fp32;
    bf16 -> ``res_out``); every task's sum of squares -> rstd -> bf16(v rstd
    (1 + w)) -> the x rows, XRDY j. The norm reads the unrounded sum, as
    ``fused_add_rms_norm`` does."""
    s, tid, lane, red = c["S"], c["tid"], c["lane"], c["red"]
    a = c["args"]
    mine = tid < s * 64
    t = fx.min(tid // 64, s - 1)
    cp = tid % 64
    col = j * COLS + 2 * cp
    lo, hi = sum_partials(
        c["poll"],
        c["peer"]["ar_own"],
        lambda src: ((src * s + t) * HIDDEN + col) // 2,
        TP,
    )
    rw = fx.Int32(
        bo.buffer_load(
            rsrc(a["residual"]), (t * HIDDEN + col) // 2, vec_width=1, dtype=T.i32
        )
    )
    z0 = bf16_round(lo) + bf_lo(rw)
    z1 = bf16_round(hi) + bf_hi(rw)
    zw = (
        fx.Vector.from_elements([z0, z1], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
    )
    if mine:
        bo.buffer_store(zw, rsrc(a["res_out"]), (t * HIDDEN + col) // 2)
    sq = wave_sum(z0 * z0 + z1 * z1)
    if mine & (lane == 0):
        c["put"](c["msq"], j * s + t, sq)
    gpu.barrier()
    if mine:
        v = c["poll"]([(c["msq"], lane * s + t, 1)])[0][0]
        fx.ptr_store(f32_of(v), red + (t * 64 + lane))
    gpu.barrier()
    tot = fx.ptr_load(red + t * 64)
    for q in range_constexpr(1, NRED):
        tot = tot + fx.ptr_load(red + (t * 64 + q))
    r = hw_rsq(tot * fx.Float32(1.0 / HIDDEN) + fx.Float32(c["eps"]))
    w0 = fx.Float32(1.0) + ld_bf(a["ln_w"], col)
    w1 = fx.Float32(1.0) + ld_bf(a["ln_w"], col + 1)
    y = (
        fx.Vector.from_elements([z0 * r * w0, z1 * r * w1], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
    )
    if mine:
        at = (t * HIDDEN + col) // 2
        bo.buffer_store(y, rsrc(c["xbuf"]), at, cache_modifier=CM_DEV)
    publish(c["put"], c["xrdy"], j, fx.Int32(1), tid == 0)
    gpu.barrier()


@traced
def wait_x(c, klen, k0):
    """Every norm task's x columns ready -> x columns k0 .. k0 + klen into LDS."""
    tid = c["tid"]
    if tid < NRED:
        c["poll"]([(c["xrdy"], tid, 1)])
    gpu.barrier()
    gemv.rows_to_lds(
        tid,
        c["xbuf"],
        klen,
        c["S"],
        c["xl"],
        klen + gemv.LDS_PAD,
        CM_DEV,
        ldk=HIDDEN,
        k0=k0,
    )


@traced
def quant_x(c):
    """The x rows in LDS through MXFP4 in place, a thread a 1 x 32 block (RCEIL
    scale): the routed experts' activations, as the stock decode kernel takes
    them. The router and the shared expert read x unquantized, before this."""
    s, tid, xl = c["S"], c["tid"], c["xl"]
    xrow = HIDDEN + gemv.LDS_PAD
    nb = s * HIDDEN // 32
    for i in range_constexpr((nb + THREADS - 1) // THREADS):
        b = tid + THREADS * i
        if b < nb:
            base = ((b // (HIDDEN // 32)) * xrow + 32 * (b % (HIDDEN // 32))) // 2
            ws = [fx.ptr_load(xl + (base + j)) for j in range_constexpr(16)]
            vs = []
            for j in range_constexpr(16):
                vs += [bf_lo(ws[j]), bf_hi(ws[j])]
            amax = fx.Float32(0.0)
            for j in range_constexpr(32):
                amax = fx.max(amax, fx.Float32(fmath.absf(vs[j])))
            e = mx_scale(amax)
            for j in range_constexpr(16):
                w = bf16_pair(mx_qdq(vs[2 * j], e), mx_qdq(vs[2 * j + 1], e))
                fx.ptr_store(w.bitcast(fx.Int32), xl + (base + j))
    gpu.barrier()


# ---------------------------------------------------------------- router / sgu
def router_loads(c, i, q):
    a = c["args"]
    return gemv.loads(
        c["lane"],
        c["wave"],
        a["w_gate"],
        c["rank"] * EPR + i * ROWS,
        R_K,
        ldk=HIDDEN,
        k0=q * R_K,
    )


@traced
def stage_router(c, i, q, ops):
    """Router task (i, q): this rank's experts 16 i .., K quarter q. Parts
    q > 0 publish fp32 partials; part 0 adds them in part order, rounds to bf16
    (GateLinear's output) and puts the logits to every rank's RLOG."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    wait_x(c, R_K, q * R_K)
    acc = gemv.mfmas(lane, wave, c["xl"], R_K + gemv.LDS_PAD, R_K, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS * s:
        r = tid % ROWS
        t = tid // ROWS
        v = row_sum(red, r, t)
        at = lambda qq: ((i * R_PARTS + qq) * s + t) * ROWS + r  # noqa: E731
        if q > 0:
            c["put"](c["rpart"], at(q), v)
        if q == 0:
            got = c["poll"]([(c["rpart"], at(qq), 1) for qq in range(1, R_PARTS)])
            for qq in range_constexpr(R_PARTS - 1):
                v = v + f32_of(got[qq][0])
            v = bf16_round(v)
            for p in range_constexpr(TP):
                c["put"](
                    c["peer"]["rlog"][p], (c["rank"] * s + t) * EPR + i * ROWS + r, v
                )
    gpu.barrier()


@traced
def stage_sgu(c, g, ops):
    """Shared gate | up rows 16 g .. (gate 0 .. SI, up SI .. 2 SI) -> bf16 SGU."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    wait_x(c, HIDDEN, 0)
    acc = gemv.mfmas(lane, wave, c["xl"], HIDDEN + gemv.LDS_PAD, HIDDEN, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < (ROWS // 2) * s:
        rp = tid % (ROWS // 2)
        t = tid // (ROWS // 2)
        v0 = bf16_round(row_sum(red, 2 * rp, t))
        v1 = bf16_round(row_sum(red, 2 * rp + 1, t))
        c["put_bf"](c["sgu"], t * 2 * SI + g * ROWS + 2 * rp, [v0, v1])
    gpu.barrier()


# ---------------------------------------------------------------- route
@traced
def stage_sgate(c, t):
    """Token t's shared-expert gate: sigmoid(bf16(x . w_sg)) as bf16 -> SGATE
    (ReplicatedLinear then torch.sigmoid, both bf16 out)."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    if tid < NRED:
        c["poll"]([(c["xrdy"], tid, 1)])
    gpu.barrier()
    per = HIDDEN // THREADS  # 16: two 16 B loads of x and of w a thread
    acc = fx.Float32(0.0)
    for h in range_constexpr(per // 8):
        k = tid * per + 8 * h
        xw = fx.Vector(
            bo.buffer_load(
                rsrc(c["xbuf"]),
                (t * HIDDEN + k) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        ww = fx.Vector(
            bo.buffer_load(rsrc(a["w_sg"]), k // 2, vec_width=4, dtype=T.i32)
        )
        for d in range_constexpr(4):
            acc = acc + bf_lo(xw[d]) * bf_lo(ww[d]) + bf_hi(xw[d]) * bf_hi(ww[d])
    acc = wave_sum(acc)
    if lane == 0:
        fx.ptr_store(acc, red + wave)
    gpu.barrier()
    if tid == 0:
        tot = fx.ptr_load(red + 0)
        for w in range_constexpr(1, WAVES):
            tot = tot + fx.ptr_load(red + w)
        c["put"](c["sgate"], t, bf16_round(sigmoid(bf16_round(tot))))
    gpu.barrier()


def _okey(v):
    """An order-preserving uint32 key of an f32 (larger float, larger key)."""
    b = v.bitcast(fx.Int32)
    neg = b < fx.Int32(0)
    return fx.Uint32(neg.select(~b, b ^ fx.Int32(-2147483648)))


@traced
def stage_route(c, t):
    """Token t (wave 0): the 512 bf16 logits, 8 a lane (expert lane + 64 i) ->
    the top-10 by a radix select on their keys (the 10th largest key by a
    32-step bit search of ballot counts; equal keys: the lower experts) -> the
    10 ranked by logit, descending -> exp(l - l_max) renormalized over the 10
    (the renormalized softmax) -> ROUTE (every rank alike)."""
    lane, wave, red = c["lane"], c["wave"], c["red"]
    if wave == 0:
        s = c["S"]
        per = E // 64
        specs = []
        for i in range_constexpr(per):
            e = lane + 64 * i
            src = e // EPR
            specs.append((c["peer"]["rlog_own"], (src * s + t) * EPR + e % EPR, 1))
        got = c["poll"](specs)
        keys, lgs = [], []
        for i in range_constexpr(per):
            lg = f32_of(got[i][0])
            # a NaN logit (a warmup's garbage row) ranks last
            keys.append(_okey((lg == lg).select(lg, fx.Float32(-3.0e38))))
            lgs.append((lg == lg).select(lg, fx.Float32(-3.0e38)))

        def count_ge(th):
            n = fx.Int32(0)
            for i in range_constexpr(per):
                n = n + popcount(ballot(keys[i] >= th))
            return n

        th = fx.Uint32(0)
        for bit in range_constexpr(31, -1, -1):
            cand = th | fx.Uint32(1 << bit)
            th = (count_ge(cand) >= TOPK).select(cand, th)
        n_gt = fx.Int32(0)
        for i in range_constexpr(per):
            n_gt = n_gt + popcount(ballot(keys[i] > th))
        need = fx.Int32(TOPK) - n_gt
        eq_before = fx.Int32(0)
        sel_before = fx.Int32(0)
        for i in range_constexpr(per):
            eqb = ballot(keys[i] == th)
            eq_rank = eq_before + lanes_below(eqb, lane)
            sel = (keys[i] > th) | ((keys[i] == th) & (eq_rank < need))
            selb = ballot(sel)
            pos = sel_before + lanes_below(selb, lane)
            if sel:
                fx.ptr_store(keys[i].bitcast(fx.Float32), red + pos * 3)
                fx.ptr_store(
                    fx.Int32(lane + 64 * i).bitcast(fx.Float32), red + (pos * 3 + 1)
                )
                fx.ptr_store(lgs[i], red + (pos * 3 + 2))
            eq_before = eq_before + popcount(eqb)
            sel_before = sel_before + popcount(selb)
        q = fx.min(lane, TOPK - 1)
        mk = fx.Uint32(fx.ptr_load(red + q * 3).bitcast(fx.Int32))
        me = fx.ptr_load(red + (q * 3 + 1)).bitcast(fx.Int32)
        ml = fx.ptr_load(red + (q * 3 + 2))
        rank = fx.Int32(0)
        for j in range_constexpr(TOPK):
            ok = fx.Uint32(fx.ptr_load(red + j * 3).bitcast(fx.Int32))
            oe = fx.ptr_load(red + (j * 3 + 1)).bitcast(fx.Int32)
            before = (ok > mk) | ((ok == mk) & (oe < me))
            rank = rank + before.select(fx.Int32(1), fx.Int32(0))
        if lane < TOPK:
            fx.ptr_store(ml, red + (64 + rank))
        top = fx.ptr_load(red + 64)
        ex = hw_exp2((ml - top) * fx.Float32(LOG2E))
        if lane < TOPK:
            fx.ptr_store(ex, red + (96 + rank))
        total = fx.Float32(0.0)
        for j in range_constexpr(TOPK):
            total = total + fx.ptr_load(red + (96 + j))
        inv = fx.Float32(1.0) / total
        if lane < TOPK:
            c["put"](
                c["route"], t * 2 * TOPK + rank, fx.min(fx.max(me, 0), fx.Int32(E - 1))
            )
            c["put"](c["route"], t * 2 * TOPK + TOPK + rank, ex * inv)
    gpu.barrier()


@traced
def load_route(c):
    """ROUTE -> LDS (ids, weights); the union U in expert order: flag[e] -> slot,
    expert[slot], kof[slot][t] (the token's top-k index, -1 if not picked);
    returns |U|."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rt = c["rt"]
    n = s * 2 * TOPK
    if tid < n:
        v = c["poll"]([(c["route"], tid, 1)])[0][0]
        fx.ptr_store(v, rt["route"] + tid)
    for i in range_constexpr((E + THREADS - 1) // THREADS):
        e = tid + THREADS * i
        if e < E:
            fx.ptr_store(fx.Int32(0), rt["flag"] + e)
    for i in range_constexpr((MAX_U * 8 + THREADS - 1) // THREADS):
        j = tid + THREADS * i
        if j < MAX_U * 8:
            fx.ptr_store(fx.Int32(-1), rt["kof"] + j)
    gpu.barrier()
    if tid < s * TOPK:
        t = tid // TOPK
        k = tid % TOPK
        e = fx.max(fx.min(fx.ptr_load(rt["route"] + (t * 2 * TOPK + k)), E - 1), 0)
        fx.ptr_store(fx.Int32(1), rt["flag"] + e)
    gpu.barrier()
    # E = 512 = THREADS: an expert a thread
    f = fx.ptr_load(rt["flag"] + tid)
    off, tot = block_excl_scan(f, lane, wave, rt["scan"])
    gpu.barrier()
    if f == 1:
        fx.ptr_store(off, rt["flag"] + tid)
        fx.ptr_store(fx.Int32(tid), rt["expert"] + off)
    gpu.barrier()
    if tid < s * TOPK:
        t = tid // TOPK
        k = tid % TOPK
        e = fx.max(fx.min(fx.ptr_load(rt["route"] + (t * 2 * TOPK + k)), E - 1), 0)
        slot = fx.ptr_load(rt["flag"] + e)
        fx.ptr_store(fx.Int32(k), rt["kof"] + (slot * 8 + t))
    gpu.barrier()
    return uniform(tot)


# ---------------------------------------------------------------- ug
def ug_loads(c, task):
    """Task ``task``'s w13 operands for this wave (gate or up by wave % 2, K
    quarter wave / 2: 16 steps) and its scale words (a word: two steps)."""
    lane, wave = c["lane"], c["wave"]
    a, rt = c["args"], c["rt"]
    slot = task // UG_GROUPS
    g = task % UG_GROUPS
    e = uniform(fx.ptr_load(rt["expert"] + slot))
    gu = wave % 2
    kq = wave // 2
    rg = L.w13_group(gu, g)
    r_w = rsrc(a["w13"])
    base = L.w13_base(e)
    tiles = [
        fp4_tile(r_w, L.fp4_tile_dword(base, rg, HIDDEN, kq * UG_STEPS + i, lane))
        for i in range(UG_STEPS)
    ]
    r_s = rsrc(a["w13s"])
    row = rg * ROWS + lane % ROWS
    words = [
        scale_word(
            r_s, L.w13_scale_index(e, row, 4 * (kq * UG_STEPS + 2 * i) + lane // 16)
        )
        for i in range(UG_STEPS // 2)
    ]
    return tiles + words


@traced
def stage_ug(c, task, ops):
    """Slot task / 16, columns 16 (task % 16) ..: the gate and up rows against
    the quantized x rows in LDS (``ops``: ``ug_loads``); silu(g) u -> fp32 ->
    INTER of each picking token."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    rt = c["rt"]
    slot = task // UG_GROUPS
    g = task % UG_GROUPS
    kq = wave // 2
    rg = L.w13_group(wave % 2, g)
    tiles = ops[:UG_STEPS]
    words = ops[UG_STEPS:]
    t = fx.min(lane % 16, s - 1)
    xrow = HIDDEN + gemv.LDS_PAD
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(UG_STEPS):
        st = kq * UG_STEPS + i
        # the code is byte (st % 2) 2 + rg % 2 of its word (e8m0_shuffle)
        code = (words[i // 2] >> ((st % 2) * 16 + (rg % 2) * 8)) & 0xFF
        xoff = t * xrow + st * L.FP4_STEP + 32 * (lane // 16)
        acc = fp4_mfmas(tiles[i], pow2(code), c["xl"], xoff, acc)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    # thread (token, column pair): g, u summed over the K quarters
    if tid < s * (UG_COLS // 2):
        tt = tid // (UG_COLS // 2)
        cp = tid % (UG_COLS // 2)
        k = fx.ptr_load(rt["kof"] + (slot * 8 + tt))
        gw = (0, 2, 4, 6)
        uw = (1, 3, 5, 7)
        v = [
            silu_mul(
                row_sum(red, 2 * cp + d, tt, waves=gw),
                row_sum(red, 2 * cp + d, tt, waves=uw),
            ).bitcast(fx.Int32)
            for d in range(2)
        ]
        if k >= 0:
            pick = tt * TOPK + k
            c["put_words"](c["inter"], pick * RI + g * UG_COLS + 2 * cp, v)
    gpu.barrier()


# ---------------------------------------------------------------- down
@traced
def load_inter(c):
    """Every pick's INTER (fp32) through MXFP4 by 32 columns (RCEIL scale, the
    block's 16 column pairs on 16 lanes of a row), as the stock decode kernel
    quantizes its intermediate -> LDS (bf16 rows of RI, padded); one poll batch."""
    s, tid = c["S"], c["tid"]
    n = s * TOPK * RI // 2
    per = (n + THREADS - 1) // THREADS
    irow = RI + gemv.LDS_PAD
    got = c["poll"](
        [(c["inter"], 2 * fx.min(tid + THREADS * i, n - 1), 2) for i in range(per)]
    )
    for i in range_constexpr(per):
        p = fx.min(tid + THREADS * i, n - 1)
        pick = p // (RI // 2)
        cp = p % (RI // 2)
        h0, h1 = f32_of(got[i][0]), f32_of(got[i][1])
        amax = fx.max(fx.Float32(fmath.absf(h0)), fx.Float32(fmath.absf(h1)))
        for off in range_constexpr(4):
            amax = xred(amax, 1 << off, fx.max)
        e = mx_scale(amax)
        w = bf16_pair(mx_qdq(h0, e), mx_qdq(h1, e)).bitcast(fx.Int32)
        if tid + THREADS * i < n:
            fx.ptr_store(w, c["il"] + (pick * irow + 2 * cp) // 2)
    gpu.barrier()


@traced
def load_h(c):
    """SGU -> h = bf16(silu(g) u) (SiluAndMul on bf16) -> LDS rows of SI."""
    s, tid = c["S"], c["tid"]
    n = s * SI // 2
    per = (n + THREADS - 1) // THREADS
    hrow = SI + gemv.LDS_PAD
    specs = []
    for i in range_constexpr(per):
        p = fx.min(tid + THREADS * i, n - 1)
        t = p // (SI // 2)
        cp = p % (SI // 2)
        specs += [
            (c["sgu"], (t * 2 * SI) // 2 + cp, 1),
            (c["sgu"], (t * 2 * SI + SI) // 2 + cp, 1),
        ]
    got = c["poll"](specs)
    for i in range_constexpr(per):
        p = fx.min(tid + THREADS * i, n - 1)
        t = p // (SI // 2)
        cp = p % (SI // 2)
        g, u = got[2 * i][0], got[2 * i + 1][0]
        h0 = silu_mul(bf_lo(g), bf_lo(u))
        h1 = silu_mul(bf_hi(g), bf_hi(u))
        hw = (
            fx.Vector.from_elements([h0, h1], fx.Float32)
            .to(fx.BFloat16)
            .bitcast(fx.Int32)[0]
        )
        if tid + THREADS * i < n:
            fx.ptr_store(hw, c["hl"] + (t * hrow + 2 * cp) // 2)
    gpu.barrier()


def down_loads(c, rgd, nu, b0):
    """The w2 tiles and scale words of a wave's batch from slot b0 (slots b0,
    b0 + 8, ...; past |U| the last slot's)."""
    lane = c["lane"]
    a, rt = c["args"], c["rt"]
    r_w = rsrc(a["w2"])
    r_s = rsrc(a["w2s"])
    row = rgd * ROWS + lane % ROWS
    tiles, words = [], []
    for j in range(DOWN_BATCH):
        sl = fx.min(b0 + WAVES * j, nu - 1)
        e = uniform(fx.ptr_load(rt["expert"] + sl))
        base = L.w2_base(e)
        tiles += [
            fp4_tile(r_w, L.fp4_tile_dword(base, rgd, RI, st, lane))
            for st in range(W2_STEPS)
        ]
        words.append(scale_word(r_s, L.w2_scale_index(e, row, lane // 16)))
    return tiles + words


@traced
def stage_down(c, rgd, nu, ops0):
    """Hidden rows 16 rgd ..: every slot's w2 against its picks' INTER (wave w
    the slots w, w + 8, ...; a batch in flight under the last one's MFMAs)
    -> fp32 (t, k) contributions times the route weight in LDS -> summed in
    top-k order -> bf16; + the shared down rows against h, times SGATE ->
    bf16 partial -> every rank's FINAL."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    rt = c["rt"]
    t = fx.min(lane % 16, s - 1)
    irow = RI + gemv.LDS_PAD
    nt = DOWN_BATCH * W2_STEPS
    # the shared down's operands first: K 256 is two chunks (waves 0, 1)
    sops = gemv.loads(lane, wave, c["args"]["w_sd"], rgd * ROWS, SI)
    for i in range_constexpr((s * TOPK * ROWS + THREADS - 1) // THREADS):
        j = tid + THREADS * i
        if j < s * TOPK * ROWS:
            fx.ptr_store(fx.Float32(0.0), c["contrib"] + j)
    gpu.barrier()
    b0 = wave
    ops = ops0
    while b0 < nu:
        nb = b0 + WAVES * DOWN_BATCH
        ops_n = down_loads(c, rgd, nu, fx.min(nb, nu - 1))
        for j in range_constexpr(DOWN_BATCH):
            sl = fx.min(b0 + WAVES * j, nu - 1)
            live = (b0 + WAVES * j) < nu
            k = fx.ptr_load(rt["kof"] + (sl * 8 + t))
            picked = (k >= 0) & (lane % 16 < s)
            pick = t * TOPK + fx.max(k, 0)
            word = ops[nt + j]
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for st in range_constexpr(W2_STEPS):
                code = (word >> (st * 16 + (rgd % 2) * 8)) & 0xFF
                xoff = pick * irow + st * L.FP4_STEP + 32 * (lane // 16)
                tile = ops[j * W2_STEPS + st]
                for q in range_constexpr(4):
                    xb = fx.Vector(
                        fx.ptr_load(
                            c["il"] + (xoff + 8 * q) // 2,
                            result_type=fx.Vector.make_type(4, fx.Int32),
                        )
                    )
                    xb = fx.Vector.from_elements(
                        [picked.select(xb[d], fx.Int32(0)) for d in range(4)],
                        fx.Int32,
                    )
                    a = fp4x8_bf16(tile[q], pow2(code))
                    acc = mfma_bf16(
                        a.bitcast(fx.BFloat16), xb.bitcast(fx.BFloat16), acc
                    )
            if live & picked:
                wt = f32_of(
                    fx.ptr_load(rt["route"] + (t * 2 * TOPK + TOPK + fx.max(k, 0)))
                )
                for i in range_constexpr(4):
                    row = 4 * (lane // 16) + i
                    fx.ptr_store(
                        acc[i] * wt,
                        c["contrib"] + ((t * TOPK + fx.max(k, 0)) * ROWS + row),
                    )
        b0 = nb
        ops = ops_n
    # the shared down: rows 16 rgd .. against h
    acc = gemv.mfmas(lane, wave, c["hl"], SI + gemv.LDS_PAD, SI, s, sops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < (ROWS // 2) * s:
        rp = tid % (ROWS // 2)
        tt = tid // (ROWS // 2)
        sg = f32_of(c["poll"]([(c["sgate"], tt, 1)])[0][0])
        ys = []
        for d in range_constexpr(2):
            r = 2 * rp + d
            y = fx.Float32(0.0)
            for k in range_constexpr(TOPK):
                y = y + fx.ptr_load(c["contrib"] + ((tt * TOPK + k) * ROWS + r))
            sd = bf16_round(bf16_round(row_sum(red, r, tt)) * sg)
            ys.append(bf16_round(bf16_round(y) + sd))
        for p in range_constexpr(TP):
            c["put_bf"](
                c["peer"]["final"][p],
                (c["rank"] * s + tt) * HIDDEN + rgd * ROWS + 2 * rp,
                ys,
            )
    gpu.barrier()


# ---------------------------------------------------------------- final
@traced
def stage_final(c, j):
    """Columns 128 j ..: FINAL in rank order -> bf16 out."""
    s, tid = c["S"], c["tid"]
    if tid < s * 64:
        t = tid // 64
        cp = tid % 64
        col = j * COLS + 2 * cp
        lo, hi = sum_partials(
            c["poll"],
            c["peer"]["final_own"],
            lambda src: ((src * s + t) * HIDDEN + col) // 2,
            TP,
        )
        w = (
            fx.Vector.from_elements([lo, hi], fx.Float32)
            .to(fx.BFloat16)
            .bitcast(fx.Int32)[0]
        )
        bo.buffer_store(w, rsrc(c["args"]["out"]), (t * HIDDEN + col) // 2)
    gpu.barrier()


# ---------------------------------------------------------------- kernel
_BUILDS: dict = {}


def build(key: K2Build):
    if key in _BUILDS:
        return _BUILDS[key]
    s = key.tokens
    need(1 <= s <= L.MAX_TOKENS, f"{s} rows, the kernels take 1..{L.MAX_TOKENS}")
    lay = scratch_layout(key)
    # every width at the widest's offsets: a step's parity half never overlaps
    # the other half of a step of another width
    play = peer_layout(L.MAX_TOKENS)
    half = peer_bytes(L.MAX_TOKENS) // 2

    @fx.struct
    class RouteLds:
        route: fx.Array[fx.Int32, s * 2 * TOPK, 16]
        flag: fx.Array[fx.Int32, E, 16]
        expert: fx.Array[fx.Int32, MAX_U, 16]
        kof: fx.Array[fx.Int32, MAX_U * 8, 16]
        scan: fx.Array[fx.Int32, WAVES, 16]

    @fx.struct
    class XLds:
        xl: fx.Array[fx.Int32, s * (HIDDEN + gemv.LDS_PAD) // 2, 16]

    @fx.struct
    class DownLds:
        il: fx.Array[fx.Int32, s * TOPK * (RI + gemv.LDS_PAD) // 2, 16]
        hl: fx.Array[fx.Int32, s * (SI + gemv.LDS_PAD) // 2, 16]
        contrib: fx.Array[fx.Float32, s * TOPK * ROWS, 16]

    @fx.union
    class StageLds:
        x: XLds
        down: DownLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]

    keyed = key_tuple(key, digest())
    name = f"qwen38_mono_k2_s{s}_g{int(key.diag)}_p{key.stop}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k2(
        core: Int64,
        residual: Int64,
        w_o: Int64,
        ln_w: Int64,
        w_gate: Int64,
        w_sg: Int64,
        w_sgu: Int64,
        w_sd: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        out: Int64,
        res_out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        diag: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        rl = alloc.allocate(RouteLds).peek()
        st_lds = alloc.allocate(StageLds)
        xl, dl = st_lds.x.peek(), st_lds.down.peek()
        if const_expr(key.diag):
            mb = Mailbox(step_tag(epoch, layer) - 1, diag, region_ids(REGIONS))
        else:
            mb = Mailbox(step_tag(epoch, layer) - 1)
        par = uniform(ld_i32(epoch, 0)) % 2
        bases = peer_bases(peers, TP)
        own = load_ptr64(peers, rank)
        peer = {}
        for region in ("ar", "rlog", "final"):
            off = play[region][0]
            peer[region] = [
                shift(preg(b, off, region), fx.Int64(par) * half) for b in bases
            ]
            peer[region + "_own"] = shift(preg(own, off, region), fx.Int64(par) * half)
        c = {
            "S": s,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "rank": rank,
            "eps": key.eps,
            "put": mb.put,
            "put_bf": mb.put_bf,
            "put_words": mb.put_words,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "peer": peer,
            "xl": xl.xl.ptr,
            "il": dl.il.ptr,
            "hl": dl.hl.ptr,
            "contrib": dl.contrib.ptr,
            "rt": {
                "route": rl.route.ptr,
                "flag": rl.flag.ptr,
                "expert": rl.expert.ptr,
                "kof": rl.kof.ptr,
                "scan": rl.scan.ptr,
            },
            "xbuf": scratch + fx.Int64(lay["xrow"][0]),
            "args": {
                "core": core,
                "residual": residual,
                "w_o": w_o,
                "ln_w": ln_w,
                "w_gate": w_gate,
                "w_sg": w_sg,
                "w_sgu": w_sgu,
                "w_sd": w_sd,
                "w13": w13,
                "w13s": w13s,
                "w2": w2,
                "w2s": w2s,
                "out": out,
                "res_out": res_out,
            },
        }
        for region in ("msq", "xrdy", "rpart", "sgu", "sgate", "route", "inter"):
            c[region] = sreg(scratch, lay[region][0], region)
        is_r = (bid >= R0) & (bid < R0 + R_TASKS)
        is_g = (bid >= G0) & (bid < G0 + G_TASKS)
        nj = OTASKS // BLOCKS
        # ---- o_proj (both tasks' weights in flight first; a router CTA's
        # router weights behind them, they land before x does), then the
        # post-attention add + norm, router, shared gate | up
        if is_r:
            oops = [o_loads(c, j) for j in range(nj)]
            ri = bid - R0
            rops = router_loads(c, ri // R_PARTS, ri % R_PARTS)
            run_oproj(c, oops)
            if const_expr(key.stop > 2):
                stage_router(c, ri // R_PARTS, ri % R_PARTS, rops)
        if ~is_r:
            run_oproj(c, [o_loads(c, j) for j in range(nj)])
        if const_expr(key.stop > 2):
            if bid < NRED:
                stage_norm(c, bid)
            if is_g:
                gi = bid - G0
                gops = gemv.loads(c["lane"], c["wave"], w_sgu, gi * ROWS, HIDDEN)
                stage_sgu(c, gi, gops)
        if const_expr(key.stop > 3):  # noqa: SIM102 (traced: `and` does not combine traced values)
            if (bid >= RT0) & (bid < RT0 + s):
                stage_sgate(c, bid - RT0)
                stage_route(c, bid - RT0)
        # ---- the routed experts
        if const_expr(key.stop > 4):
            nu = load_route(c)
            wait_x(c, HIDDEN, 0)
            quant_x(c)
            total = nu * UG_GROUPS
            last = total - 1
            # reversed placement: the round's leftover tasks go to the high
            # CTAs, past the norm / router / route ones
            task = BLOCKS - 1 - bid
            ops = ug_loads(c, fx.min(task, last))
            while task < total:
                nxt = task + BLOCKS
                ops_n = ug_loads(c, fx.min(nxt, last))
                stage_ug(c, task, ops)
                task = nxt
                ops = ops_n
            if const_expr(key.stop > 5):
                dops = down_loads(c, bid, nu, fx.min(c["wave"], nu - 1))
                load_h(c)
                load_inter(c)
                stage_down(c, bid, nu, dops)
                dops = down_loads(c, bid + BLOCKS, nu, fx.min(c["wave"], nu - 1))
                stage_down(c, bid + BLOCKS, nu, dops)
                if bid < NRED:
                    stage_final(c, bid)

    @flyc.jit
    def launch(
        core: Int64,
        residual: Int64,
        w_o: Int64,
        ln_w: Int64,
        w_gate: Int64,
        w_sg: Int64,
        w_sgu: Int64,
        w_sd: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        out: Int64,
        res_out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        diag: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k2(
            core,
            residual,
            w_o,
            ln_w,
            w_gate,
            w_sg,
            w_sgu,
            w_sd,
            w13,
            w13s,
            w2,
            w2s,
            out,
            res_out,
            scratch,
            peers,
            rank,
            epoch,
            layer,
            diag,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(k2, launch)
    _BUILDS[key] = launch
    return launch


def layer_post(
    key: K2Build,
    *,
    core,
    residual,
    w_o,
    ln_w,
    w_gate,
    w_sg,
    w_sgu,
    w_sd,
    w13,
    w13s,
    w2,
    w2s,
    out,
    res_out,
    scratch,
    peers,
    rank,
    epoch,
    layer,
    diag=None,
):
    """Host launcher; every weight as vLLM holds it after loading, the experts'
    shuffled (``layout``). ``peers``: the int64 table of every rank's peer buffer
    (at least ``peer_bytes(S)`` each)."""
    s = key.tokens
    bf, u8 = torch.bfloat16, torch.uint8
    need(
        core.shape == (s, OK) and residual.shape == (s, HIDDEN),
        "core / residual shape",
    )
    need(out.shape == (s, HIDDEN) and res_out.shape == (s, HIDDEN), "output shape")
    need(w_o.shape == (HIDDEN, OK) and ln_w.shape == (HIDDEN,), "o_proj / norm shape")
    need(w_gate.shape == (E, HIDDEN) and w_sg.numel() == HIDDEN, "router shape")
    need(
        w_sgu.shape == (2 * SI, HIDDEN) and w_sd.shape == (HIDDEN, SI),
        "shared expert shape",
    )
    fp4 = (u8, torch.float4_e2m1fn_x2)
    need(
        w13.shape == (E, 2 * RI, HIDDEN // 2) and w13.dtype in fp4,
        "w13: fp4x2 [512, 512, 4096]",
    )
    need(
        w2.shape == (E, HIDDEN, RI // 2) and w2.dtype in fp4,
        "w2: fp4x2 [512, 8192, 128]",
    )
    need(
        w13s.numel() == E * 2 * RI * HIDDEN // 32 and w13s.element_size() == 1,
        "w13 scales: e8m0, one a 32-block",
    )
    need(
        w2s.numel() == E * HIDDEN * RI // 32 and w2s.element_size() == 1,
        "w2 scales: e8m0, one a 32-block",
    )
    weights = (w_o, w_gate, w_sgu, w_sd, w13, w13s, w2, w2s)
    tensors = (core, residual, out, res_out, *weights)
    need(all(t.is_contiguous() for t in tensors), "a non-contiguous input")
    dense = (core, residual, out, res_out, w_o, ln_w, w_gate, w_sg, w_sgu, w_sd)
    need(all(t.dtype == bf for t in dense), "activations and dense weights: bf16")
    f = build(key)
    f(
        *ABI.pack(
            {
                "core": core.data_ptr(),
                "residual": residual.data_ptr(),
                "w_o": w_o.data_ptr(),
                "ln_w": ln_w.data_ptr(),
                "w_gate": w_gate.data_ptr(),
                "w_sg": w_sg.data_ptr(),
                "w_sgu": w_sgu.data_ptr(),
                "w_sd": w_sd.data_ptr(),
                "w13": w13.data_ptr(),
                "w13s": w13s.data_ptr(),
                "w2": w2.data_ptr(),
                "w2s": w2s.data_ptr(),
                "out": out.data_ptr(),
                "res_out": res_out.data_ptr(),
                "scratch": scratch.data_ptr(),
                "peers": peers.data_ptr(),
                "rank": rank,
                "epoch": epoch.data_ptr(),
                "layer": layer,
                "diag": 0 if diag is None else diag.data_ptr(),
            }
        ),
        stream=torch.cuda.current_stream(),
    )
