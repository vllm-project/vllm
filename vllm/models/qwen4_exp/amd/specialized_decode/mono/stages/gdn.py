# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The gated delta rule of a GDN (linear-attention) block, on the in_proj
outputs the block's earlier stages put to PROJ / BAPART (``in_proj``):

    gdn  (S x NV value heads x 8 chunks of 16 state rows, task n on CTA
         (n + goff) % 256; a CTA runs its tasks phase by phase, so the rounds'
         hand-offs overlap): the conv update of the owned channels (a key
         head's q / k by its first value head's tasks, out through QK), the
         gates, q / k l2norm, the recurrence on 16 state rows (fp32, in
         place), partial norms (ONRM), RMSNormGated -> ``core``

Rounding follows the stock decode path (AITER's Triton conv update and fused
gated delta rule, RMSNormGated) where it is elementwise: every conv product
rounded to bf16 as Triton's bf16 multiply does, the bf16 conv output, the bf16
core before the gated norm. Split-K reductions differ only in their order.

A row whose state index is <= 0 (vLLM's null block, what the GDN metadata pads
a decode batch with) writes no state and a zero core row, as stock (its query
length is 0 there).

A GDN round is 256 consecutive tasks on distinct CTAs; a task waits only on
tasks of its own round (a key head's 64 tasks, a head's 8 chunks), so no CTA
waits on work queued behind it.
"""

import math

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from vllm.models.qwen4_exp.amd.specialized_decode.mono import layout as L
from vllm.models.qwen4_exp.amd.specialized_decode.mono.common.ops import (
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    hw_exp2,
    hw_rsq,
    ld_i32,
    row_sum,
    rsrc,
    traced,
    uniform,
    wave_sum,
)
from vllm.models.qwen4_exp.amd.specialized_decode.mono.common.plan import (
    BLOCKS,
)
from vllm.models.qwen4_exp.amd.specialized_decode.mono.stages import gemv
from vllm.models.qwen4_exp.amd.specialized_decode.mono.stages.moe import (
    f32_of,
    rows_at,
    sigmoid,
    tiled,
    tok,
)

ROWS, HD = L.ROWS, L.HD
VCH = 16  # state rows a GDN task
NVCH = HD // VCH  # 8
SL = L.CONV_W - 1  # conv state entries (no spec decode)
SOFTPLUS_THRESHOLD = 20.0
LOG2E = 1.0 / math.log(2.0)
LN2 = math.log(2.0)


def ld_bf(ptr, i):
    return fx.Float32(
        fx.BFloat16(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.bf16))
    )


def ld_f32(ptr, i):
    return fx.Float32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.f32))


def hw_log2(x):
    return fx.Float32(
        llvm.call_intrinsic(
            T.f32, "llvm.amdgcn.log.f32", [fx.Float32(x).ir_value()], [], []
        )
    )


def hw_exp(x):
    return hw_exp2(x * fx.Float32(LOG2E))


def silu(z):
    return z / (fx.Float32(1.0) + hw_exp(-z))


# ---------------------------------------------------------------- in_proj_ba
def ba_loads(c, i, parts, k):
    """in_proj_ba task i's weights for this wave (rows 16 (i / parts) .., K
    columns ``k`` (i % parts) .. of ``parts k``)."""
    g, q = i // parts, i % parts
    return gemv.loads(
        c["lane"], c["wave"], c["args"]["w_ba"], g * ROWS, k, ldk=parts * k, k0=q * k
    )


def stage_ba(c, i, ops, parts, k):
    """in_proj_ba task i (``ba_loads``) on the unquantized x rows (LDS rows of
    ``parts k`` + pad) -> fp32 BAPART."""
    lane, wave = c["lane"], c["wave"]
    xl = c["xl"] + (i % parts) * k // 2
    xrow = parts * k + gemv.LDS_PAD
    tiled(
        c,
        lambda t0, n: gemv.mfmas(lane, wave, rows_at(xl, xrow, t0), xrow, k, n, ops),
        lambda t0, n: _ba_out(c, i, t0, n),
    )


@traced
def _ba_out(c, i, t0, n):
    s, tid = c["S"], c["tid"]
    if tid < ROWS * n:
        r = tid % ROWS
        j = tid // ROWS
        c["put"](c["bapart"], (i * s + tok(t0, j)) * ROWS + r, row_sum(c["red"], r, j))
    gpu.barrier()


def ba_sums(c, t, h, parts):
    """Head h's b and a of token t (in_proj_ba rows h, nv + h): the tasks'
    K parts summed in part order, rounded to bf16."""
    s, nv = c["S"], c["gdn"].nv
    got = c["poll"](
        [
            (c["bapart"], (((ro // ROWS) * parts + q) * s + t) * ROWS + ro % ROWS, 1)
            for ro in (h, nv + h)
            for q in range(parts)
        ]
    )
    ba = []
    for e in range_constexpr(2):
        v = f32_of(got[e * parts][0])
        for q in range_constexpr(1, parts):
            v = v + f32_of(got[e * parts + q][0])
        ba.append(bf16_round(v))
    return ba[0], ba[1]


# ---------------------------------------------------------------- gdn
def gdn_stage(c, t, h, ch):
    """The task's inputs that need no hand-off, issued before its first poll:
    the slot, the owned channels' conv window and weights, the state rows and
    the head's constants."""
    tid = c["tid"]
    a, g = c["args"], c["gdn"]
    k = {}
    k["slot"] = uniform(ld_i32(a["st_idx"], t))
    kh = h // (g.nv // g.nk)
    k["owner"] = (h % (g.nv // g.nk)) == 0
    # conv: thread < 48 owns channel i of part (q, k, v)
    ct = fx.min(tid, 3 * VCH - 1)
    part = ct // VCH
    i = ct % VCH
    qk_chan = part * (g.nk * HD) + kh * HD + ch * VCH + i
    v_chan = 2 * g.nk * HD + h * HD + ch * VCH + i
    chan = (part < 2).select(qk_chan, v_chan)
    k["part"], k["i"], k["chan"] = part, i, chan
    cs = a["conv_st"] + fx.Int64(fx.max(k["slot"], 0)) * fx.Int64(a["cs_seq"]) * 2
    co = chan * a["cs_dim"]
    k["cs"], k["co"] = cs, co
    k["win"] = [ld_bf(cs, co + j * a["cs_tok"]) for j in range(SL)]
    cw = fx.Vector(
        bo.buffer_load(rsrc(a["conv_w"]), chan * 2, vec_width=2, dtype=T.i32)
    )
    k["cw"] = [bf_lo(cw[0]), bf_hi(cw[0]), bf_lo(cw[1]), bf_hi(cw[1])]
    # recurrence: thread -> state row ch 16 + tid / 32, K columns (tid % 32) 4 ..
    vr = ch * VCH + tid // 32
    k["head_off"] = (h * HD + vr) * HD + (tid % 32) * 4
    k["sbase"] = (
        a["rstate"] + fx.Int64(fx.max(k["slot"], 0)) * fx.Int64(a["rs_seq"]) * 4
    )
    k["st"] = fx.Vector(
        bo.buffer_load(rsrc(k["sbase"]), k["head_off"], vec_width=4, dtype=T.f32)
    )
    k["a_log"] = ld_f32(a["a_log"], h)
    k["dtb"] = ld_bf(a["dt_bias"], h)
    return k


@traced
def gdn_conv(c, k, t, h, ch):
    """The owned channels' conv update: the window shifted (in place), the
    width-4 conv (each product rounded to bf16, fp32 sum), SiLU -> bf16; q / k
    out through QK (the key head's owner tasks), v to LDS."""
    tid = c["tid"]
    a, g = c["args"], c["gdn"]
    part, owner = k["part"], k["owner"]
    if tid < 3 * VCH:
        x = f32_of(c["poll"]([(c["proj"], t * g.proj + k["chan"], 1)])[0][0])
        owned = (part == 2) | owner
        if (k["slot"] > 0) & owned:
            rs = rsrc(k["cs"])
            for j in range_constexpr(SL):
                nv = k["win"][j + 1] if j < SL - 1 else x
                bo.buffer_store(
                    fx.Float32(nv).to(fx.BFloat16), rs, k["co"] + j * a["cs_tok"]
                )
        w, win = k["cw"], k["win"]
        acc = bf16_round(w[0] * win[0]) + bf16_round(w[1] * win[1])
        acc = acc + bf16_round(w[2] * win[2])
        acc = acc + bf16_round(w[3] * x)
        o = bf16_round(silu(acc))
        if (part < 2) & owner:
            kh = h // (g.nv // g.nk)
            at = ((t * g.nk + kh) * 2 + part) * HD + ch * VCH + k["i"]
            c["put"](c["qk"], at, o)
        if part == 2:
            fx.ptr_store(o, c["vloc"] + k["i"])
    gpu.barrier()


@traced
def gdn_gates(c, k, t, h):
    """The key head's q and k (QK) l2-normed (q scaled by HD^-1/2); the head's
    decay exp(-exp(A_log) softplus(a + dt_bias)) and beta = sigmoid(b), b and a
    as the model's ``ba_of`` hands them over."""
    tid, lane, wave, g = c["tid"], c["lane"], c["wave"], c["gdn"]
    kh = h // (g.nv // g.nk)
    if tid < 2 * HD:
        v = c["poll"]([(c["qk"], (t * g.nk + kh) * 2 * HD + tid, 1)])[0][0]
        fx.ptr_store(f32_of(v), c["qkl"] + tid)
    if tid == 2 * HD:
        b, a = c["ba_of"](c, t, h)
        x = a + k["dtb"]
        sp = hw_log2(fx.Float32(1.0) + hw_exp(x)) * fx.Float32(LN2)
        sp = (x <= fx.Float32(SOFTPLUS_THRESHOLD)).select(sp, x)
        gl = -hw_exp(k["a_log"]) * sp
        fx.ptr_store(hw_exp(gl), c["scal"] + 0)
        fx.ptr_store(sigmoid(b), c["scal"] + 1)
    gpu.barrier()
    if wave < 2:
        x0 = fx.ptr_load(c["qkl"] + (wave * HD + lane))
        x1 = fx.ptr_load(c["qkl"] + (wave * HD + 64 + lane))
        sc = hw_rsq(wave_sum(x0 * x0 + x1 * x1) + fx.Float32(1e-6))
        y0, y1 = x0 * sc, x1 * sc
        qs = fx.Float32(HD**-0.5)
        y0 = (wave == 0).select(y0 * qs, y0)
        y1 = (wave == 0).select(y1 * qs, y1)
        fx.ptr_store(y0, c["qkl"] + (wave * HD + lane))
        fx.ptr_store(y1, c["qkl"] + (wave * HD + 64 + lane))
    gpu.barrier()


@traced
def gdn_recur(c, k, t, h, ch):
    """State rows 16 ch .. of head h, one token: decay, the delta rule, the new
    state (in place), o = bf16(h q) -> LDS ``ol``, the chunk's sum of squares
    -> ONRM."""
    tid, lane = c["tid"], c["lane"]
    v = tid // 32
    k0 = (tid % 32) * 4
    live = k["slot"] > 0
    decay = fx.ptr_load(c["scal"] + 0)
    beta = fx.ptr_load(c["scal"] + 1)
    qv = [fx.ptr_load(c["qkl"] + (k0 + e)) for e in range(4)]
    kv = [fx.ptr_load(c["qkl"] + (HD + k0 + e)) for e in range(4)]
    stv = [live.select(k["st"][e], fx.Float32(0.0)) * decay for e in range(4)]
    p = stv[0] * kv[0] + stv[1] * kv[1] + stv[2] * kv[2] + stv[3] * kv[3]
    p = butterfly(p, (16, 8, 4, 2, 1))
    vn = (fx.ptr_load(c["vloc"] + v) - p) * beta
    stv = [stv[e] + vn * kv[e] for e in range(4)]
    o = stv[0] * qv[0] + stv[1] * qv[1] + stv[2] * qv[2] + stv[3] * qv[3]
    o = bf16_round(butterfly(o, (16, 8, 4, 2, 1)))
    o = live.select(o, fx.Float32(0.0))
    if live:
        bo.buffer_store(
            fx.Vector.from_elements(stv, fx.Float32),
            rsrc(k["sbase"]),
            k["head_off"],
        )
    if lane % 32 == 0:
        fx.ptr_store(o, c["ol"] + v)
    gpu.barrier()
    if tid == 0:
        ss = fx.Float32(0.0)
        for e in range_constexpr(VCH):
            x = fx.ptr_load(c["ol"] + e)
            ss = ss + x * x
        c["put"](c["onrm"], (t * c["gdn"].nv + h) * NVCH + ch, ss)
    gpu.barrier()


@traced
def gdn_norm(c, k, t, h, ch):
    """Rows 16 ch ..: RMSNormGated over the head's 128 (every chunk's partial):
    bf16(o rstd w gate(z)) -> ``core``, the gate the model's ``z_gate``."""
    tid = c["tid"]
    a, g = c["args"], c["gdn"]
    if tid < VCH:
        vr = ch * VCH + tid
        z0 = 2 * g.nk * HD + g.nv * HD
        got = c["poll"](
            [(c["onrm"], (t * g.nv + h) * NVCH + q, 1) for q in range(NVCH)]
            + [(c["proj"], t * g.proj + z0 + h * HD + vr, 1)]
        )
        ss = f32_of(got[0][0])
        for q in range_constexpr(1, NVCH):
            ss = ss + f32_of(got[q][0])
        r = hw_rsq(ss * fx.Float32(1.0 / HD) + fx.Float32(c["norm_eps"]))
        y = fx.ptr_load(c["ol"] + tid) * r * ld_bf(a["norm_w"], vr)
        y = y * c["z_gate"](f32_of(got[NVCH][0]))
        y = (k["slot"] > 0).select(y, fx.Float32(0.0))
        bo.buffer_store(
            y.to(fx.BFloat16), rsrc(a["core"]), t * (g.nv * HD) + h * HD + vr
        )
    gpu.barrier()


def gdn_round(c, r):
    """Round r's task of this CTA (``bid - goff`` cyclically + ``BLOCKS r``,
    clamped into range), its (token, head, chunk) and the round's LDS."""
    nv = c["gdn"].nv
    n = c["S"] * nv * NVCH
    task = (c["bid"] + (BLOCKS - c["goff"])) % BLOCKS + BLOCKS * r
    tc = fx.min(task, n - 1)
    cr = dict(c)
    for name, m in (("qkl", 2 * HD), ("vloc", VCH), ("ol", VCH), ("scal", 4)):
        cr[name] = c[name] + r * m
    return cr, task < n, tc // (nv * NVCH), (tc // NVCH) % nv, tc % NVCH


@traced
def gdn_tasks(c, rounds):
    """This CTA's GDN tasks (``rounds`` of them) phase by phase: a task waits
    only on earlier phases of other CTAs' tasks, and every CTA finishes a
    phase's work before it waits in the next."""
    tasks = [gdn_round(c, r) for r in range(rounds)]
    ks = [gdn_stage(cr, t, h, ch) for cr, _, t, h, ch in tasks]
    phases = (
        lambda cr, k, t, h, ch: gdn_conv(cr, k, t, h, ch),
        lambda cr, k, t, h, ch: gdn_gates(cr, k, t, h),
        lambda cr, k, t, h, ch: gdn_recur(cr, k, t, h, ch),
        lambda cr, k, t, h, ch: gdn_norm(cr, k, t, h, ch),
    )
    for phase in phases:
        for (cr, live, t, h, ch), k in zip(tasks, ks):
            if live:
                phase(cr, k, t, h, ch)


# ---------------------------------------------------------------- kernel
