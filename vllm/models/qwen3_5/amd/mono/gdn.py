# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K1 ``gdn_pre``: one Qwen3.8 GDN (linear-attention) layer of a decode step from
the layer's input to the gated core output (the out_proj input, K2's), S <= 8
rows, one token a request.

One launch a layer and rank, ``BLOCKS`` x ``THREADS``:

    front    (CTAs 0 .. 63, 128 columns each): hidden + residual (bf16) ->
             ``res_out``; the columns' sums of squares -> MSQ
    in_proj  (290 x 16 rows: 288 of in_proj_qkvz, 2 of in_proj_ba; task j on
             CTA 255 - j, the round's 34 leftovers past the front CTAs): every
             in_proj CTA builds the x rows itself (input_layernorm, Gemma,
             1 + w, from MSQ), bf16 GEMV against them -> bf16 PROJ
    gdn      (S x 16 value heads x 8 chunks of 16 state rows, on CTAs 64 ..
             when a single round fits there, else 0 ..; a CTA runs its tasks
             phase by phase, so the rounds' hand-offs overlap):
             the conv update of the owned channels (a key head's q / k by its
             first value head's tasks, out through QK), the gates, q / k
             l2norm, the recurrence on 16 state rows (fp32, in place), partial
             norms (ONRM), RMSNormGated -> ``core``

With ``qq`` (an online-MXFP4 in_proj_qkvz on the ASM FP4 GEMM, its weight and
scales shuffled as K2's fp4 o_proj's), the in_proj_qkvz tasks read the x rows
through MXFP4 (RCEIL), as ``per_1x32_f4_quant_hip`` quantizes the GEMM's input;
in_proj_ba stays bf16 on the unquantized rows, as 16 tasks (2 row groups x 8
K parts) on CTAs 0 .. 15 whose weights go out at the start; the gates sum the
parts.

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
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.kimi_k3.amd.mono.common.ops import (
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
from vllm.models.kimi_k3.amd.mono.common.plan import (
    BLOCKS,
    THREADS,
    KernelAbi,
    key_tuple,
    pair_layout,
)
from vllm.models.kimi_k3.amd.mono.common.sync import Mailbox, sreg
from vllm.models.kimi_k3.amd.mono.stages import gemv
from vllm.models.qwen3_5.amd.mono import layout as L
from vllm.models.qwen3_5.amd.mono.layout import need
from vllm.models.qwen3_5.amd.mono.moe import (
    f32_of,
    fp4_gemv,
    fp4_gemv_loads,
    fp6_gemv,
    fp6_gemv_loads,
    quant_x,
    sigmoid,
    step_tag,
)
from vllm.models.qwen3_5.amd.mono.sources import digest

HIDDEN, ROWS, HD = L.HIDDEN, L.ROWS, L.HD
NK, NV, KEY = L.NK, L.NV, L.KEY
NPROJ = L.QKVZ + L.BA  # 4640 PROJ entries a token: [q | k | v | z | b | a]
PTASKS = NPROJ // ROWS  # 290
QKVZ_TASKS = L.QKVZ // ROWS  # 288
COLS = 128  # columns a front task
NRED = HIDDEN // COLS  # 64
Z0 = L.CONV  # PROJ entry of z
B0 = L.QKVZ  # PROJ entry of b
A0 = L.QKVZ + NV  # PROJ entry of a
VCH = 16  # state rows a GDN task
NVCH = HD // VCH  # 8
HPK = NV // NK  # value heads a key head: 8
GTASKS = NV * NVCH  # GDN tasks a token: 128
SL = L.CONV_W - 1  # conv state entries (no spec decode)
SOFTPLUS_THRESHOLD = 20.0
BA_PARTS = 8  # ``qq``: in_proj_ba's K split
BA_K = HIDDEN // BA_PARTS
BA_TASKS = L.BA // ROWS * BA_PARTS  # 16, on CTAs 0 ..
LOG2E = 1.0 / math.log(2.0)
LN2 = math.log(2.0)

# the CTAs with a second in_proj task are past the front ones
assert BLOCKS < PTASKS <= 2 * BLOCKS and NRED <= 2 * BLOCKS - PTASKS
assert BLOCKS % (HPK * NVCH) == 0, "a key head's GDN tasks must share a round"
assert BA_TASKS <= 2 * BLOCKS - QKVZ_TASKS
# The MXFP6 in_proj (q6) puts the q, k and v tasks, those before z, on the
# CTAs between the front ones and the ones with two tasks.
assert Z0 % ROWS == 0 and Z0 // ROWS == 2 * BLOCKS - QKVZ_TASKS - NRED
# It also moves the GDN tasks of the owner heads 0 and 8 to CTAs 64 .. 95.
assert NV == 16 and HPK == 8 and NVCH == 8 and NRED == 64

REGIONS = ("msq", "proj", "qk", "onrm", "bapart")

_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class K1Build:
    tokens: int
    first: bool = False  # the model's first layer: no residual (residual := hidden)
    eps: float = 1e-6  # input_layernorm
    norm_eps: float = 1e-6  # the gated norm
    qq: bool = False  # in_proj_qkvz MXFP4 (ASM FP4 GEMM layout)
    # (``qq``) in_proj_qkvz in MXFP6 E2M3, in the layout of moe.fp6_pack,
    # multiplied with the bf16 x rows (W6A16). quant_x does not run. The
    # in_proj tasks are placed by what waits for them (``qp``).
    q6: bool = False

    @property
    def qp(self) -> bool:
        """The q, k and v tasks run on CTAs with one in_proj task and no
        front. The CTAs with two in_proj tasks run z tasks and issue their
        second task's weights after the x polls. When the GDN tasks fill
        one round of BLOCKS, the owner tasks of the key heads run on CTAs
        64 .. 95. The outputs are those of the build without it, bit for
        bit."""
        return self.q6


ABI = KernelAbi(
    (
        "hidden",
        "residual",
        "res_out",
        "ln_w",
        "w_qkvz",
        "w_qkvzs",
        "w_ba",
        "conv_w",
        "conv_st",
        "cs_seq",
        "cs_dim",
        "cs_tok",
        "a_log",
        "dt_bias",
        "norm_w",
        "rstate",
        "rs_seq",
        "st_idx",
        "core",
        "scratch",
        "epoch",
        "layer",
    )
)


ROUNDS = -(-L.MAX_TOKENS * GTASKS // BLOCKS)  # 4: GDN rounds at the widest


def scratch_layout(key: K1Build) -> dict:
    s = key.tokens
    pairs = {
        "msq": NRED * s,
        "proj": s * NPROJ,
        "qk": s * NK * 2 * HD,
        "onrm": s * NV * NVCH,
        "bapart": BA_TASKS * s * ROWS if key.qq else 0,
    }
    return pair_layout(pairs.items())


def scratch_bytes(key: K1Build) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def ld_i32w(ptr, i):
    return fx.Int32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.i32))


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


def pack_bf(v0, v1):
    return (
        fx.Vector.from_elements([v0, v1], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
    )


# ---------------------------------------------------------------- front
@traced
def stage_front(c, j):
    """Columns 128 j ..: hidden + residual (fp32; bf16 -> ``res_out``); the
    columns' sums of squares -> MSQ (the in_proj CTAs build x: ``local_x``)."""
    s, tid, lane = c["S"], c["tid"], c["lane"]
    a = c["args"]
    mine = tid < s * 64
    t = fx.min(tid // 64, s - 1)
    col = j * COLS + 2 * (tid % 64)
    at = (t * HIDDEN + col) // 2
    hw = ld_i32w(a["hidden"], at)
    if const_expr(c["first"]):
        z0, z1 = bf_lo(hw), bf_hi(hw)
    else:
        rw = ld_i32w(a["residual"], at)
        z0 = bf_lo(hw) + bf_lo(rw)
        z1 = bf_hi(hw) + bf_hi(rw)
    if mine:
        bo.buffer_store(pack_bf(z0, z1), rsrc(a["res_out"]), at)
    sq = wave_sum(z0 * z0 + z1 * z1)
    if mine & (lane == 0):
        c["put"](c["msq"], j * s + t, sq)


# ---------------------------------------------------------------- in_proj
def proj_loads(c, rg):
    """in_proj task ``rg``'s weights for this wave: in_proj_qkvz rows 16 rg ..,
    past them in_proj_ba's."""
    a = c["args"]
    is_ba = rg >= QKVZ_TASKS
    w = is_ba.select(a["w_ba"], a["w_qkvz"])
    row0 = (rg - is_ba.select(fx.Int32(QKVZ_TASKS), fx.Int32(0))) * ROWS
    return gemv.loads(c["lane"], c["wave"], w, row0, HIDDEN)


def qkvz_fp4_loads(c, rg):
    """An fp4 (``q6``: fp6) in_proj_qkvz task's (rg < 288) operands for this
    wave."""
    a = c["args"]
    if const_expr(c.get("q6", False)):
        return fp6_gemv_loads(
            c["lane"], c["wave"], a["w_qkvz"], a["w_qkvzs"], rg, HIDDEN
        )
    return fp4_gemv_loads(c["lane"], c["wave"], a["w_qkvz"], a["w_qkvzs"], rg, HIDDEN)


@traced
def local_x(c):
    """Every front task's sums of squares -> each token's rstd (fp32 sum in task
    order) -> x = bf16((hidden + residual) rstd (1 + w)) rows into LDS, 8
    columns a lane. The norm reads the unrounded sum, as ``fused_add_rms_norm``
    does."""
    s, tid, lane, red = c["S"], c["tid"], c["lane"], c["red"]
    a = c["args"]
    if tid < 64 * s:
        t = tid // 64
        v = c["poll"]([(c["msq"], lane * s + t, 1)])[0][0]
        fx.ptr_store(f32_of(v), red + (t * 64 + lane))
    gpu.barrier()
    if tid < s:
        tot = fx.ptr_load(red + tid * 64)
        for q in range_constexpr(1, NRED):
            tot = tot + fx.ptr_load(red + (tid * 64 + q))
        r = hw_rsq(tot * fx.Float32(1.0 / HIDDEN) + fx.Float32(c["eps"]))
        fx.ptr_store(r, c["rs"] + tid)
    gpu.barrier()
    xrow = HIDDEN + gemv.LDS_PAD
    words = s * HIDDEN // 8
    for i in range_constexpr((words + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, words - 1)
        t = e // (HIDDEN // 8)
        kk = e % (HIDDEN // 8) * 8
        g = (t * HIDDEN + kk) // 2
        hv = fx.Vector(bo.buffer_load(rsrc(a["hidden"]), g, vec_width=4, dtype=T.i32))
        if const_expr(not c["first"]):
            rv = fx.Vector(
                bo.buffer_load(rsrc(a["residual"]), g, vec_width=4, dtype=T.i32)
            )
        wv = fx.Vector(
            bo.buffer_load(rsrc(a["ln_w"]), kk // 2, vec_width=4, dtype=T.i32)
        )
        r = fx.ptr_load(c["rs"] + t)
        ys = []
        for d in range_constexpr(4):
            if const_expr(c["first"]):
                z0, z1 = bf_lo(hv[d]), bf_hi(hv[d])
            else:
                z0 = bf_lo(hv[d]) + bf_lo(rv[d])
                z1 = bf_hi(hv[d]) + bf_hi(rv[d])
            w0 = fx.Float32(1.0) + bf_lo(wv[d])
            w1 = fx.Float32(1.0) + bf_hi(wv[d])
            ys.append(pack_bf(z0 * r * w0, z1 * r * w1))
        if tid + THREADS * i < words:
            fx.ptr_store(
                fx.Vector.from_elements(ys, fx.Int32), c["xl"] + (t * xrow + kk) // 2
            )
    gpu.barrier()


def ba_loads(c, i, parts, k):
    """in_proj_ba task i's weights for this wave (rows 16 (i / parts) .., K
    columns ``k`` (i % parts) .. of ``parts k``)."""
    g, q = i // parts, i % parts
    return gemv.loads(
        c["lane"], c["wave"], c["args"]["w_ba"], g * ROWS, k, ldk=parts * k, k0=q * k
    )


@traced
def stage_ba(c, i, ops, parts, k):
    """in_proj_ba task i (``ba_loads``) on the unquantized x rows (LDS rows of
    ``parts k`` + pad) -> fp32 BAPART."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    xl = c["xl"] + (i % parts) * k // 2
    acc = gemv.mfmas(lane, wave, xl, parts * k + gemv.LDS_PAD, k, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS * s:
        r = tid % ROWS
        t = tid // ROWS
        c["put"](c["bapart"], (i * s + t) * ROWS + r, row_sum(red, r, t))
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


@traced
def stage_proj(c, rg, ops, fp4=False):
    """Rows 16 rg .. of [in_proj_qkvz; in_proj_ba] -> bf16 PROJ (``fp4``: an
    in_proj_qkvz task on the quantized rows, with ``q6`` an MXFP6 one on the
    bf16 rows)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    xrow = HIDDEN + gemv.LDS_PAD
    if const_expr(fp4 and c.get("q6", False)):
        acc = fp6_gemv(lane, wave, c["xl"], xrow, HIDDEN, s, ops)
    elif const_expr(fp4):
        acc = fp4_gemv(lane, wave, c["xl"], xrow, HIDDEN, s, rg, ops)
    else:
        acc = gemv.mfmas(lane, wave, c["xl"], xrow, HIDDEN, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS * s:
        r = tid % ROWS
        t = tid // ROWS
        v = bf16_round(row_sum(red, r, t))
        c["put"](c["proj"], t * NPROJ + rg * ROWS + r, v)
    gpu.barrier()


def _ba_of(c, t, h):
    """Head h's b and a of token t: in_proj_ba's K parts summed and rounded to
    bf16 (``qq``), else its PROJ entries."""
    if const_expr(c["qq"]):
        return ba_sums(c, t, h, BA_PARTS)
    got = c["poll"](
        [(c["proj"], t * NPROJ + B0 + h, 1), (c["proj"], t * NPROJ + A0 + h, 1)]
    )
    return f32_of(got[0][0]), f32_of(got[1][0])


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


def gdn_offset(key) -> int:
    """The first GDN CTA: past the front CTAs (whose in_proj weights go out
    late) and short of the two-task in_proj ones, if a single round fits."""
    fits = key.tokens * GTASKS <= 2 * BLOCKS - QKVZ_TASKS - NRED
    return NRED if fits else 0


def gdn_round(c, r):
    """Round r's task of this CTA (``bid - goff`` cyclically + ``BLOCKS r``,
    clamped into range), its (token, head, chunk) and the round's LDS."""
    nv = c["gdn"].nv
    n = c["S"] * nv * NVCH
    task = (c["bid"] + (BLOCKS - c["goff"])) % BLOCKS + BLOCKS * r
    if c.get("qp", False) and n == BLOCKS and c["goff"] == 0:
        # ``qp``: a block of NVCH consecutive tasks is one head's chunks, so
        # block t NV + h is token t's head h. The owner heads 0 and 8 of
        # tokens 0 and 1 are blocks 0, 8, 16 and 24. Blocks 0, 16 and 24
        # swap places with blocks 9, 10 and 11, so that all owner tasks
        # run on CTAs 64 .. 95, whose in_proj tasks are v rows. The front
        # CTAs and the CTAs with two in_proj tasks then run no owner task.
        q = task // NVCH
        for qa, qb in ((0, 9), (16, 10), (24, 11)):
            q = (q == qa).select(fx.Int32(qb), (q == qb).select(fx.Int32(qa), q))
        task = q * NVCH + task % NVCH
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
_BUILDS: dict = {}


def build(key: K1Build):
    if key in _BUILDS:
        return _BUILDS[key]
    s = key.tokens
    need(1 <= s <= L.MAX_TOKENS, f"{s} rows, the kernels take 1..{L.MAX_TOKENS}")
    lay = scratch_layout(key)
    nr = -(-s * GTASKS // BLOCKS)  # GDN rounds: each its own LDS

    @fx.struct
    class XLds:
        xl: fx.Array[fx.Int32, s * (HIDDEN + gemv.LDS_PAD) // 2, 16]

    @fx.struct
    class GdnLds:
        qkl: fx.Array[fx.Float32, nr * 2 * HD, 16]
        vloc: fx.Array[fx.Float32, nr * VCH, 16]
        ol: fx.Array[fx.Float32, nr * VCH, 16]
        scal: fx.Array[fx.Float32, nr * 4, 16]

    @fx.union
    class StageLds:
        x: XLds
        gdn: GdnLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, THREADS * 4, 16]
        rs: fx.Array[fx.Float32, 16, 16]

    keyed = key_tuple(key, digest())
    name = (
        f"qwen38_mono_k1_s{s}_f{int(key.first)}_q{int(key.qq)}{int(key.q6)}"
        f"_p{int(key.qp)}"
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k1(
        hidden: Int64,
        residual: Int64,
        res_out: Int64,
        ln_w: Int64,
        w_qkvz: Int64,
        w_qkvzs: Int64,
        w_ba: Int64,
        conv_w: Int64,
        conv_st: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        a_log: Int64,
        dt_bias: Int64,
        norm_w: Int64,
        rstate: Int64,
        rs_seq: Int32,
        st_idx: Int64,
        core: Int64,
        scratch: Int64,
        epoch: Int64,
        layer: Int32,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        st_lds = alloc.allocate(StageLds)
        xl, gl = st_lds.x.peek(), st_lds.gdn.peek()
        mb = Mailbox(step_tag(epoch, layer) - 1)
        c = {
            "qq": key.qq,
            "q6": key.q6,
            "qp": key.qp,
            "gdn": L.GDN,
            "ba_of": _ba_of,
            "z_gate": silu,
            "goff": gdn_offset(key),
            "rs": lds.rs.ptr,
            "S": s,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "first": key.first,
            "eps": key.eps,
            "norm_eps": key.norm_eps,
            "put": mb.put,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "xl": xl.xl.ptr,
            "qkl": gl.qkl.ptr,
            "vloc": gl.vloc.ptr,
            "ol": gl.ol.ptr,
            "scal": gl.scal.ptr,
            "args": {
                "hidden": hidden,
                "residual": residual,
                "res_out": res_out,
                "ln_w": ln_w,
                "w_qkvz": w_qkvz,
                "w_qkvzs": w_qkvzs,
                "w_ba": w_ba,
                "conv_w": conv_w,
                "conv_st": conv_st,
                "cs_seq": cs_seq,
                "cs_dim": cs_dim,
                "cs_tok": cs_tok,
                "a_log": a_log,
                "dt_bias": dt_bias,
                "norm_w": norm_w,
                "rstate": rstate,
                "rs_seq": rs_seq,
                "st_idx": st_idx,
                "core": core,
            },
        }
        for region in REGIONS:
            c[region] = sreg(scratch, lay[region][0], region)
        # ---- front; in_proj (a front CTA's weights only after its polls: a
        # poll's load retires behind every earlier load)
        if bid < NRED:
            stage_front(c, bid)
        if const_expr(not key.qq):
            rg = BLOCKS - 1 - bid
            ops = proj_loads(c, rg)
            local_x(c)
            stage_proj(c, rg, ops)
            if bid >= 2 * BLOCKS - PTASKS:
                rg2 = 2 * BLOCKS - 1 - bid
                stage_proj(c, rg2, proj_loads(c, rg2))
        elif const_expr(key.qp):
            # ``qp``: the front CTAs 0 .. 63 keep z tasks 255 .. 192 and load
            # their weights after the front, as without qp. CTAs 64 .. 223
            # run tasks 159 .. 0, so the q, k and v rows, which the conv of
            # every GDN task polls, come from CTAs with one in_proj task.
            # CTAs 224 .. 255 run z tasks 191 .. 160 and then 287 .. 256.
            # Only the gated norm at the end reads z. Each group of CTAs
            # issues all its loads in its own branch, in this order. A load
            # issued before the branch whose value only the branch uses is
            # moved into the branch by the compiler, behind the loads that
            # are already there. x is not quantized (q6).
            nb = 2 * BLOCKS - QKVZ_TASKS
            if bid < NRED:
                rga = BLOCKS - 1 - bid
                opsa = qkvz_fp4_loads(c, rga)
                if bid < BA_TASKS:
                    bops = ba_loads(c, bid, BA_PARTS, BA_K)
                    local_x(c)
                    stage_ba(c, bid, bops, BA_PARTS, BA_K)
                    stage_proj(c, rga, opsa, fp4=True)
                if bid >= BA_TASKS:
                    local_x(c)
                    stage_proj(c, rga, opsa, fp4=True)
            if (bid >= NRED) & (bid < nb):
                rgb = nb - 1 - bid
                opsb = qkvz_fp4_loads(c, rgb)
                local_x(c)
                stage_proj(c, rgb, opsb, fp4=True)
            if bid >= nb:
                # The x polls go out behind the first task's weights only.
                # The second task's weights go out after them, and the first
                # task's MFMAs run while they are on their way.
                rgc = Z0 // ROWS + BLOCKS - 1 - bid
                rgc2 = 2 * BLOCKS - 1 - bid
                opsc = qkvz_fp4_loads(c, rgc)
                local_x(c)
                opsc2 = qkvz_fp4_loads(c, rgc2)
                stage_proj(c, rgc, opsc, fp4=True)
                stage_proj(c, rgc2, opsc2, fp4=True)
        else:

            def quant():
                # The MXFP6 in_proj (q6) multiplies the bf16 rows, so x is
                # not quantized.
                if const_expr(not key.q6):
                    quant_x(c, HIDDEN)

            # rg < 256 is in_proj_qkvz; the second tasks 256 .. 287 on CTAs
            # 224 .. 255, their weights with the first's; in_proj_ba's parts
            # on CTAs 0 .. 15 (unquantized x: before quant_x)
            rg = BLOCKS - 1 - bid
            ops = qkvz_fp4_loads(c, rg)
            if bid >= 2 * BLOCKS - QKVZ_TASKS:
                rg2 = 2 * BLOCKS - 1 - bid
                ops2 = qkvz_fp4_loads(c, rg2)
                local_x(c)
                quant()
                stage_proj(c, rg, ops, fp4=True)
                stage_proj(c, rg2, ops2, fp4=True)
            if bid < BA_TASKS:
                bops = ba_loads(c, bid, BA_PARTS, BA_K)
                local_x(c)
                stage_ba(c, bid, bops, BA_PARTS, BA_K)
                quant()
                stage_proj(c, rg, ops, fp4=True)
            if (bid >= BA_TASKS) & (bid < 2 * BLOCKS - QKVZ_TASKS):
                local_x(c)
                quant()
                stage_proj(c, rg, ops, fp4=True)
        gdn_tasks(c, nr)

    @flyc.jit
    def launch(
        hidden: Int64,
        residual: Int64,
        res_out: Int64,
        ln_w: Int64,
        w_qkvz: Int64,
        w_qkvzs: Int64,
        w_ba: Int64,
        conv_w: Int64,
        conv_st: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        a_log: Int64,
        dt_bias: Int64,
        norm_w: Int64,
        rstate: Int64,
        rs_seq: Int32,
        st_idx: Int64,
        core: Int64,
        scratch: Int64,
        epoch: Int64,
        layer: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k1(
            hidden,
            residual,
            res_out,
            ln_w,
            w_qkvz,
            w_qkvzs,
            w_ba,
            conv_w,
            conv_st,
            cs_seq,
            cs_dim,
            cs_tok,
            a_log,
            dt_bias,
            norm_w,
            rstate,
            rs_seq,
            st_idx,
            core,
            scratch,
            epoch,
            layer,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(k1, launch)
    _BUILDS[key] = launch
    return launch


def gdn_pre(
    key: K1Build,
    *,
    hidden,
    residual,
    res_out,
    ln_w,
    w_qkvz,
    w_ba,
    conv_w,
    conv_state,
    a_log,
    dt_bias,
    norm_w,
    rstate,
    st_idx,
    core,
    scratch,
    epoch,
    layer,
    w_qkvzs=None,
):
    """Host launcher; every weight as vLLM holds it after loading (``w_qkvzs``:
    an fp4 in_proj_qkvz's scales, ``key.qq``). ``residual``
    None on the first layer (``key.first``). ``conv_state`` [slots, dim, 3] or a
    transposed view of [slots, 3, dim] (strides read); ``rstate`` [slots, 16,
    128, 128] fp32, a slot dense, slots strided; ``st_idx`` [>= S] int32, <= 0 a
    pad row; ``epoch`` a device int32 bumped once a step, ``layer`` this
    launch's slot below ``TAG_SLOTS``."""
    s = key.tokens
    bf = torch.bfloat16
    need((residual is None) == key.first, "residual given iff not the first layer")
    need(
        hidden.shape == (s, HIDDEN) and res_out.shape == (s, HIDDEN),
        "hidden / res_out shape",
    )
    need(core.shape == (s, L.CORE) and ln_w.shape == (HIDDEN,), "core / ln_w shape")
    need(key.qq or not key.q6, "q6 is a weight format of the qq path")
    if key.q6:
        need(
            w_qkvz.dtype == torch.uint8 and w_qkvz.numel() == L.QKVZ * HIDDEN * 3 // 4,
            "in_proj_qkvz: fp6_pack's MXFP6 weight, 4608 x 8192 x 3/4 bytes",
        )
    if key.qq and not key.q6:
        need(
            w_qkvz.shape == (L.QKVZ, HIDDEN // 2)
            and w_qkvz.dtype in (torch.uint8, torch.float4_e2m1fn_x2),
            "in_proj_qkvz: fp4x2 [4608, 4096]",
        )
    if key.qq:
        need(
            w_qkvzs is not None
            and w_qkvzs.numel() == L.QKVZ * HIDDEN // 32
            and w_qkvzs.element_size() == 1
            and w_qkvzs.is_contiguous(),
            "in_proj_qkvz scales: e8m0, one a 32-block",
        )
    else:
        need(
            w_qkvz.shape == (L.QKVZ, HIDDEN) and w_qkvz.dtype == bf,
            "in_proj_qkvz: bf16 [4608, 8192]",
        )
    need(w_ba.shape == (L.BA, HIDDEN), "in_proj_ba shape")
    need(conv_w.numel() == L.CONV * L.CONV_W and conv_w.dtype == bf, "conv1d weight")
    need(
        conv_state.dtype == bf and conv_state.shape[1:] == (L.CONV, SL),
        "conv state: bf16 [slots, 2560, 3]",
    )
    need(
        rstate.dtype == torch.float32 and rstate.shape[1:] == (NV, HD, HD),
        "recurrent state: fp32 [slots, 16, 128, 128]",
    )
    need(rstate[0].is_contiguous(), "a recurrent state slot must be dense")
    need(a_log.dtype == torch.float32 and dt_bias.dtype == bf, "A_log / dt_bias dtype")
    need(norm_w.dtype == bf and norm_w.numel() == HD, "gated norm weight")
    need(st_idx.dtype == torch.int32 and st_idx.numel() >= s, "state indices")
    weights = (hidden, res_out, w_qkvz, w_ba, conv_w, core)
    need(all(t.is_contiguous() for t in weights), "a non-contiguous input")
    need(
        all(t.dtype == bf for t in (hidden, res_out, ln_w, w_ba, core)),
        "activations and dense weights must be bf16",
    )
    f = build(key)
    f(
        *ABI.pack(
            {
                "hidden": hidden.data_ptr(),
                "residual": hidden.data_ptr()
                if residual is None
                else residual.data_ptr(),
                "res_out": res_out.data_ptr(),
                "ln_w": ln_w.data_ptr(),
                "w_qkvz": w_qkvz.data_ptr(),
                "w_qkvzs": w_qkvzs.data_ptr() if key.qq else 0,
                "w_ba": w_ba.data_ptr(),
                "conv_w": conv_w.data_ptr(),
                "conv_st": conv_state.data_ptr(),
                "cs_seq": conv_state.stride(0),
                "cs_dim": conv_state.stride(1),
                "cs_tok": conv_state.stride(2),
                "a_log": a_log.data_ptr(),
                "dt_bias": dt_bias.data_ptr(),
                "norm_w": norm_w.data_ptr(),
                "rstate": rstate.data_ptr(),
                "rs_seq": rstate.stride(0),
                "st_idx": st_idx.data_ptr(),
                "core": core.data_ptr(),
                "scratch": scratch.data_ptr(),
                "epoch": epoch.data_ptr(),
                "layer": layer,
            }
        ),
        stream=torch.cuda.current_stream(),
    )
