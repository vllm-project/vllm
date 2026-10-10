# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K1 ``gdn_pre``: one Qwen3.8 GDN (linear-attention) layer of a decode step from
the layer's input to the gated core output (the out_proj input, K2's), S <= 8
rows, one token a request.

One launch a layer and rank, ``BLOCKS`` x ``THREADS``:

    front    (CTAs 0 .. 63, 128 columns each): hidden + residual (bf16) ->
             ``res_out``; every task's sum of squares -> input_layernorm
             (Gemma, 1 + w) -> x rows (plain, XRDY a task)
    in_proj  (290 x 16 rows: 288 of in_proj_qkvz, 2 of in_proj_ba; task j on
             CTA 255 - j, the round's 34 leftovers past the front CTAs): bf16
             GEMV against the x rows -> bf16 PROJ
    gdn      (S x 16 value heads x 8 chunks of 16 state rows, on CTAs 0 ..):
             the conv update of the owned channels (a key head's q / k by its
             first value head's tasks, out through QK), the gates, q / k
             l2norm, the recurrence on 16 state rows (fp32, in place), partial
             norms (ONRM), RMSNormGated -> ``core``

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

from vllm.models.kimi_k3.amd.mono.common.debug import region_ids
from vllm.models.kimi_k3.amd.mono.common.ops import (
    CM_DEV,
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
    first_task,
    key_tuple,
    pair_layout,
)
from vllm.models.kimi_k3.amd.mono.common.sync import Mailbox, publish, sreg
from vllm.models.kimi_k3.amd.mono.stages import gemv
from vllm.models.qwen3_5.amd.mono import layout as L
from vllm.models.qwen3_5.amd.mono.layout import need
from vllm.models.qwen3_5.amd.mono.sources import digest

HIDDEN, ROWS, HD = L.HIDDEN, L.ROWS, L.HD
NK, NV, KEY = L.NK, L.NV, L.KEY
NPROJ = L.QKVZ + L.BA  # 4640 PROJ entries a token: [q | k | v | z | b | a]
PTASKS = NPROJ // ROWS  # 290
QKVZ_TASKS = L.QKVZ // ROWS  # 288
COLS = 128  # columns a front task
NRED = HIDDEN // COLS  # 64
V0 = 2 * KEY  # PROJ / conv channel of v
Z0 = L.CONV  # PROJ entry of z
B0 = L.QKVZ  # PROJ entry of b
A0 = L.QKVZ + NV  # PROJ entry of a
VCH = 16  # state rows a GDN task
NVCH = HD // VCH  # 8
HPK = NV // NK  # value heads a key head: 8
GTASKS = NV * NVCH  # GDN tasks a token: 128
SL = L.CONV_W - 1  # conv state entries (no spec decode)
SOFTPLUS_THRESHOLD = 20.0
LOG2E = 1.0 / math.log(2.0)
LN2 = math.log(2.0)
TAG_SLOTS = 256

# the CTAs with a second in_proj task are past the front ones
assert BLOCKS < PTASKS <= 2 * BLOCKS and NRED <= 2 * BLOCKS - PTASKS
assert BLOCKS % (HPK * NVCH) == 0, "a key head's GDN tasks must share a round"

REGIONS = ("msq", "xrdy", "proj", "qk", "onrm")

_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class K1Build:
    tokens: int
    first: bool = False  # the model's first layer: no residual (residual := hidden)
    eps: float = 1e-6  # input_layernorm
    norm_eps: float = 1e-6  # the gated norm
    diag: bool = False  # bounded waits recording into ``diag`` (debug)
    # debug: run phases < stop only (1 front, 2 in_proj, 3 gdn)
    stop: int = 99


ABI = KernelAbi(
    (
        "hidden",
        "residual",
        "res_out",
        "ln_w",
        "w_qkvz",
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
        "diag",
    )
)


def scratch_layout(key: K1Build) -> dict:
    s = key.tokens
    pairs = {
        "msq": NRED * s,
        "xrdy": NRED,
        "proj": s * NPROJ,
        "qk": s * NK * 2 * HD,
        "onrm": s * NV * NVCH,
    }
    lay = pair_layout(pairs.items())
    end = max(o + n for o, n in lay.values())
    lay["xrow"] = ((end + 15) // 16 * 16, s * HIDDEN * 2)
    return lay


def scratch_bytes(key: K1Build) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def step_tag(epoch, slot):
    """This launch's mailbox tag: the step's epoch and the launch's slot."""
    e = uniform(ld_i32(epoch, 0))
    return (e % (1 << 22)) * TAG_SLOTS + slot + 1


def f32_of(w):
    return w.bitcast(fx.Float32)


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


def sigmoid(z):
    return fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp(-z))


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
    """Columns 128 j ..: hidden + residual (fp32; bf16 -> ``res_out``); every
    task's sum of squares -> rstd -> bf16(v rstd (1 + w)) -> the x rows, XRDY j.
    The norm reads the unrounded sum, as ``fused_add_rms_norm`` does."""
    s, tid, lane, red = c["S"], c["tid"], c["lane"], c["red"]
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
    y = pack_bf(z0 * r * w0, z1 * r * w1)
    if mine:
        bo.buffer_store(y, rsrc(c["xbuf"]), at, cache_modifier=CM_DEV)
    publish(c["put"], c["xrdy"], j, fx.Int32(1), tid == 0)
    gpu.barrier()


# ---------------------------------------------------------------- in_proj
def proj_loads(c, rg):
    """in_proj task ``rg``'s weights for this wave: in_proj_qkvz rows 16 rg ..,
    past them in_proj_ba's."""
    a = c["args"]
    is_ba = rg >= QKVZ_TASKS
    w = is_ba.select(a["w_ba"], a["w_qkvz"])
    row0 = (rg - is_ba.select(fx.Int32(QKVZ_TASKS), fx.Int32(0))) * ROWS
    return gemv.loads(c["lane"], c["wave"], w, row0, HIDDEN)


@traced
def wait_x(c):
    """Every front task's x columns ready -> the x rows into LDS."""
    tid = c["tid"]
    if tid < NRED:
        c["poll"]([(c["xrdy"], tid, 1)])
    gpu.barrier()
    gemv.rows_to_lds(
        tid, c["xbuf"], HIDDEN, c["S"], c["xl"], HIDDEN + gemv.LDS_PAD, CM_DEV
    )


@traced
def stage_proj(c, rg, ops):
    """Rows 16 rg .. of [in_proj_qkvz; in_proj_ba] -> bf16 PROJ."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    xrow = HIDDEN + gemv.LDS_PAD
    acc = gemv.mfmas(lane, wave, c["xl"], xrow, HIDDEN, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS * s:
        r = tid % ROWS
        t = tid // ROWS
        v = bf16_round(row_sum(red, r, t))
        c["put"](c["proj"], t * NPROJ + rg * ROWS + r, v)
    gpu.barrier()


# ---------------------------------------------------------------- gdn
def gdn_stage(c, t, h, ch):
    """The task's inputs that need no hand-off, issued before its first poll:
    the slot, the owned channels' conv window and weights, the state rows and
    the head's constants."""
    tid = c["tid"]
    a = c["args"]
    k = {}
    k["slot"] = uniform(ld_i32(a["st_idx"], t))
    kh = h // HPK
    k["owner"] = (h % HPK) == 0
    # conv: thread < 48 owns channel i of part (q, k, v)
    ct = fx.min(tid, 3 * VCH - 1)
    part = ct // VCH
    i = ct % VCH
    qk_chan = part * KEY + kh * HD + ch * VCH + i
    v_chan = V0 + h * HD + ch * VCH + i
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
    a = c["args"]
    part, owner = k["part"], k["owner"]
    if tid < 3 * VCH:
        x = f32_of(c["poll"]([(c["proj"], t * NPROJ + k["chan"], 1)])[0][0])
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
            kh = h // HPK
            at = ((t * NK + kh) * 2 + part) * HD + ch * VCH + k["i"]
            c["put"](c["qk"], at, o)
        if part == 2:
            fx.ptr_store(o, c["vloc"] + k["i"])
    gpu.barrier()


@traced
def gdn_gates(c, k, t, h):
    """The key head's q and k (QK) l2-normed (q scaled by HD^-1/2); the head's
    decay exp(-exp(A_log) softplus(a + dt_bias)) and beta = sigmoid(b)."""
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    kh = h // HPK
    if tid < 2 * HD:
        v = c["poll"]([(c["qk"], (t * NK + kh) * 2 * HD + tid, 1)])[0][0]
        fx.ptr_store(f32_of(v), c["qkl"] + tid)
    if tid == 2 * HD:
        got = c["poll"](
            [(c["proj"], t * NPROJ + B0 + h, 1), (c["proj"], t * NPROJ + A0 + h, 1)]
        )
        x = f32_of(got[1][0]) + k["dtb"]
        sp = hw_log2(fx.Float32(1.0) + hw_exp(x)) * fx.Float32(LN2)
        sp = (x <= fx.Float32(SOFTPLUS_THRESHOLD)).select(sp, x)
        g = -hw_exp(k["a_log"]) * sp
        fx.ptr_store(hw_exp(g), c["scal"] + 0)
        fx.ptr_store(sigmoid(f32_of(got[0][0])), c["scal"] + 1)
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
        c["put"](c["onrm"], (t * NV + h) * NVCH + ch, ss)
    gpu.barrier()


@traced
def gdn_norm(c, k, t, h, ch):
    """Rows 16 ch ..: RMSNormGated over the head's 128 (every chunk's partial):
    bf16(o rstd w silu(z)) -> ``core``."""
    tid = c["tid"]
    a = c["args"]
    if tid < VCH:
        vr = ch * VCH + tid
        got = c["poll"](
            [(c["onrm"], (t * NV + h) * NVCH + q, 1) for q in range(NVCH)]
            + [(c["proj"], t * NPROJ + Z0 + h * HD + vr, 1)]
        )
        ss = f32_of(got[0][0])
        for q in range_constexpr(1, NVCH):
            ss = ss + f32_of(got[q][0])
        r = hw_rsq(ss * fx.Float32(1.0 / HD) + fx.Float32(c["norm_eps"]))
        y = fx.ptr_load(c["ol"] + tid) * r * ld_bf(a["norm_w"], vr)
        y = y * silu(f32_of(got[NVCH][0]))
        y = (k["slot"] > 0).select(y, fx.Float32(0.0))
        bo.buffer_store(y.to(fx.BFloat16), rsrc(a["core"]), t * L.CORE + h * HD + vr)
    gpu.barrier()


@traced
def gdn_tasks(c):
    s, bid = c["S"], c["bid"]
    for task in range(first_task(bid, 0), s * GTASKS, BLOCKS):
        t = task // GTASKS
        h = (task // NVCH) % NV
        ch = task % NVCH
        k = gdn_stage(c, t, h, ch)
        gdn_conv(c, k, t, h, ch)
        gdn_gates(c, k, t, h)
        gdn_recur(c, k, t, h, ch)
        gdn_norm(c, k, t, h, ch)


# ---------------------------------------------------------------- kernel
_BUILDS: dict = {}


def build(key: K1Build):
    if key in _BUILDS:
        return _BUILDS[key]
    s = key.tokens
    need(1 <= s <= L.MAX_TOKENS, f"{s} rows, the kernels take 1..{L.MAX_TOKENS}")
    lay = scratch_layout(key)

    @fx.struct
    class XLds:
        xl: fx.Array[fx.Int32, s * (HIDDEN + gemv.LDS_PAD) // 2, 16]

    @fx.struct
    class GdnLds:
        qkl: fx.Array[fx.Float32, 2 * HD, 16]
        vloc: fx.Array[fx.Float32, VCH, 16]
        ol: fx.Array[fx.Float32, VCH, 16]
        scal: fx.Array[fx.Float32, 4, 16]

    @fx.union
    class StageLds:
        x: XLds
        gdn: GdnLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, THREADS * 4, 16]

    keyed = key_tuple(key, digest())
    name = f"qwen38_mono_k1_s{s}_f{int(key.first)}_g{int(key.diag)}_p{key.stop}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k1(
        hidden: Int64,
        residual: Int64,
        res_out: Int64,
        ln_w: Int64,
        w_qkvz: Int64,
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
        diag: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        st_lds = alloc.allocate(StageLds)
        xl, gl = st_lds.x.peek(), st_lds.gdn.peek()
        if const_expr(key.diag):
            mb = Mailbox(step_tag(epoch, layer) - 1, diag, region_ids(REGIONS))
        else:
            mb = Mailbox(step_tag(epoch, layer) - 1)
        c = {
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
            "xbuf": scratch + fx.Int64(lay["xrow"][0]),
            "args": {
                "hidden": hidden,
                "residual": residual,
                "res_out": res_out,
                "ln_w": ln_w,
                "w_qkvz": w_qkvz,
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
        if const_expr(key.stop > 2):
            rg = BLOCKS - 1 - bid
            ops = proj_loads(c, rg)
            wait_x(c)
            stage_proj(c, rg, ops)
            if bid >= 2 * BLOCKS - PTASKS:
                rg2 = 2 * BLOCKS - 1 - bid
                stage_proj(c, rg2, proj_loads(c, rg2))
        if const_expr(key.stop > 3):
            gdn_tasks(c)

    @flyc.jit
    def launch(
        hidden: Int64,
        residual: Int64,
        res_out: Int64,
        ln_w: Int64,
        w_qkvz: Int64,
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
        diag: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k1(
            hidden,
            residual,
            res_out,
            ln_w,
            w_qkvz,
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
            diag,
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
    diag=None,
):
    """Host launcher; every weight as vLLM holds it after loading. ``residual``
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
    need(
        w_qkvz.shape == (L.QKVZ, HIDDEN) and w_ba.shape == (L.BA, HIDDEN),
        "in_proj_qkvz / in_proj_ba shape",
    )
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
        all(t.dtype == bf for t in (hidden, res_out, ln_w, w_qkvz, w_ba, core)),
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
                "diag": 0 if diag is None else diag.data_ptr(),
            }
        ),
        stream=torch.cuda.current_stream(),
    )
