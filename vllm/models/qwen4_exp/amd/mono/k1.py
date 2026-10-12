# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K1 ``gdn_pre``: the front of one Qwen3.8-Flash-Next layer of a decode step,
S <= 8 rows, one token a request: the attention HC's norm and mix of the
(materialized) multi-stream state, then, on a GDN layer, the linear attention
up to the gated core output (K2's out_proj input).

One launch a layer and rank, ``BLOCKS`` x ``THREADS``:

    msq      (``self_msq``, CTAs 0 .. 19): the state's sums of squares (the
             model's first layer and the one after the PLE; past them K2 hands
             them over as ``msq_in``)
    down     (``hc.DOWN_TASKS``, CTAs 88 ..): xn of the state, [down; inject]
             -> LORA and the injection logits -> ``inj_out`` (bf16)
    up       (``hc.UP_TASKS``, CTAs 0 .. 159): the gate-mix -> block input BIN
             (``front_only``: ``bin_out``, the QSA layers' attention input; the
             launch ends there)
    in_proj  (every CTA: in_proj_qkvz row groups bid, bid + 256 through MXFP4,
             their weights in flight from the start; in_proj_ba as 3 row
             groups x 2 K halves on the unquantized rows, CTAs 160 ..)
    gdn      (S x NV value heads x 8 chunks of 16 state rows, task n on CTA
             (n + goff) % 256): as the Qwen3.8 mono's K1 (``qwen3_5.amd.mono.
             gdn``), at this model's head counts

Every CTA runs its GDN tasks phase by phase (conv, gates, recurrence, norm) and
a phase polls only work of an earlier phase, so the task placement needs no
round structure. A row whose state index is <= 0 writes no state and a zero
core row, as stock.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.kimi_k3.amd.mono.common.ops import (
    bf16_pair,
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
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
    WAVES,
    KernelAbi,
    key_tuple,
    pair_layout,
)
from vllm.models.kimi_k3.amd.mono.common.sync import Mailbox, sreg
from vllm.models.kimi_k3.amd.mono.stages import gemv
from vllm.models.qwen3_5.amd.mono.gdn import (
    LN2,
    SOFTPLUS_THRESHOLD,
    hw_exp,
    hw_log2,
    ld_bf,
    ld_f32,
    silu,
)
from vllm.models.qwen3_5.amd.mono.layer import quant_x, sigmoid, step_tag
from vllm.models.qwen4_exp.amd.mono import fp4, hc
from vllm.models.qwen4_exp.amd.mono import layout as L
from vllm.models.qwen4_exp.amd.mono.layout import need
from vllm.models.qwen4_exp.amd.mono.sources import digest

H, HH, HC, ROWS, HD = L.HIDDEN, L.HH, L.HC, L.ROWS, L.HD
D0 = BLOCKS - hc.DOWN_TASKS  # 88: the down CTAs, to the last
VCH = 16  # state rows a GDN task
NVCH = HD // VCH  # 8
SL = L.CONV_W - 1  # conv state entries (no spec decode)
BA_PARTS = 2
BA_K = H // BA_PARTS
BA0 = hc.UP_TASKS  # 160: in_proj_ba's CTAs (down only)
XROW = H + gemv.LDS_PAD

assert hc.UP_TASKS <= BLOCKS and D0 >= 0 and hc.NRED <= hc.UP_TASKS

_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class K1Build:
    tokens: int
    tp: int = 2
    self_msq: bool = False  # the state's sums of squares computed here
    front_only: bool = False  # the HC front alone (a QSA layer)
    eps: float = 1e-6  # the HC norm
    norm_eps: float = 1e-6  # the GDN gated norm


ABI = KernelAbi(
    (
        "state",
        "msq_in",
        "hc_w",
        "w_down",
        "w_up",
        "inj_out",
        "bin_out",
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


def ba_tasks(lay):
    return lay.BA // ROWS * BA_PARTS


def gdn_tasks_of(lay):
    """GDN tasks a token."""
    return lay.NV * NVCH


def scratch_layout(key: K1Build) -> dict:
    s = key.tokens
    lay = L.Layout(key.tp)
    pairs = hc.regions(s)
    if not key.front_only:
        pairs = pairs | {
            "proj": s * lay.QKVZ,
            "qk": s * lay.NK * 2 * HD,
            "onrm": s * lay.NV * NVCH,
            "bapart": ba_tasks(lay) * s * ROWS,
        }
    out = pair_layout(pairs.items())
    if not key.front_only:
        end = (max(o + n for o, n in out.values()) + 15) // 16 * 16
        out["bin"] = (end, hc.bin_bytes(s))
    return out


def scratch_bytes(key: K1Build) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def f32_of(w):
    return w.bitcast(fx.Float32)


def gdn_offset(key: K1Build) -> int:
    """The first GDN CTA: the high (down only) CTAs while a single round fits."""
    n = key.tokens * gdn_tasks_of(L.Layout(key.tp))
    return BLOCKS - n if n < BLOCKS else 0


# ---------------------------------------------------------------- in_proj
def qkvz_loads(c, rg):
    a = c["args"]
    return fp4.fp4_gemv_loads(c["lane"], c["wave"], a["w_qkvz"], a["w_qkvzs"], rg, H)


@traced
def stage_qkvz(c, rg, ops):
    """in_proj_qkvz rows 16 rg .. on the MXFP4-exact x rows -> bf16 PROJ."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    acc = fp4.fp4_gemv(lane, wave, c["xl"], XROW, H, s, rg, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS * s:
        r = tid % ROWS
        t = tid // ROWS
        v = bf16_round(row_sum(red, r, t))
        c["put"](c["proj"], t * c["lay"].QKVZ + rg * ROWS + r, v)
    gpu.barrier()


def ba_loads(c, i):
    """in_proj_ba task i's weights (rows 16 (i / 2) .., K half i % 2)."""
    g, q = i // BA_PARTS, i % BA_PARTS
    return gemv.loads(
        c["lane"], c["wave"], c["args"]["w_ba"], g * ROWS, BA_K, ldk=H, k0=q * BA_K
    )


@traced
def stage_ba(c, i, ops):
    """in_proj_ba task i on the unquantized x rows -> fp32 BAPART."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    xl = c["xl"] + (i % BA_PARTS) * BA_K // 2
    acc = gemv.mfmas(lane, wave, xl, XROW, BA_K, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS * s:
        r = tid % ROWS
        t = tid // ROWS
        c["put"](c["bapart"], (i * s + t) * ROWS + r, row_sum(red, r, t))
    gpu.barrier()


# ---------------------------------------------------------------- gdn
def gdn_stage(c, t, h, ch):
    """The task's inputs that need no hand-off, issued before its first poll:
    the slot, the owned channels' conv window and weights, the state rows and
    the head's constants."""
    tid = c["tid"]
    a = c["args"]
    lay = c["lay"]
    k = {}
    k["slot"] = uniform(ld_i32(a["st_idx"], t))
    kh = h // (lay.NV // lay.NK)
    k["owner"] = (h % (lay.NV // lay.NK)) == 0
    # conv: thread < 48 owns channel i of part (q, k, v)
    ct = fx.min(tid, 3 * VCH - 1)
    part = ct // VCH
    i = ct % VCH
    qk_chan = part * lay.KEY + kh * HD + ch * VCH + i
    v_chan = 2 * lay.KEY + h * HD + ch * VCH + i
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
    lay = c["lay"]
    part, owner = k["part"], k["owner"]
    if tid < 3 * VCH:
        x = f32_of(c["poll"]([(c["proj"], t * lay.QKVZ + k["chan"], 1)])[0][0])
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
            kh = h // (lay.NV // lay.NK)
            at = ((t * lay.NK + kh) * 2 + part) * HD + ch * VCH + k["i"]
            c["put"](c["qk"], at, o)
        if part == 2:
            fx.ptr_store(o, c["vloc"] + k["i"])
    gpu.barrier()


@traced
def gdn_gates(c, k, t, h):
    """The key head's q and k (QK) l2-normed (q scaled by HD^-1/2); the head's
    decay exp(-exp(A_log) softplus(a + dt_bias)) and beta = sigmoid(b), b and a
    the sums of in_proj_ba's K halves (bf16)."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    lay = c["lay"]
    kh = h // (lay.NV // lay.NK)
    if tid < 2 * HD:
        v = c["poll"]([(c["qk"], (t * lay.NK + kh) * 2 * HD + tid, 1)])[0][0]
        fx.ptr_store(f32_of(v), c["qkl"] + tid)
    if tid == 2 * HD:
        rows = (h, lay.NV + h)  # b, a
        parts = c["poll"](
            [
                (
                    c["bapart"],
                    (((ro // ROWS) * BA_PARTS + q) * s + t) * ROWS + ro % ROWS,
                    1,
                )
                for ro in rows
                for q in range(BA_PARTS)
            ]
        )
        ba = []
        for e in range_constexpr(2):
            v = f32_of(parts[e * BA_PARTS][0])
            for q in range_constexpr(1, BA_PARTS):
                v = v + f32_of(parts[e * BA_PARTS + q][0])
            ba.append(bf16_round(v))
        x = ba[1] + k["dtb"]
        sp = hw_log2(fx.Float32(1.0) + hw_exp(x)) * fx.Float32(LN2)
        sp = (x <= fx.Float32(SOFTPLUS_THRESHOLD)).select(sp, x)
        g = -hw_exp(k["a_log"]) * sp
        fx.ptr_store(hw_exp(g), c["scal"] + 0)
        fx.ptr_store(sigmoid(ba[0]), c["scal"] + 1)
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
    lay = c["lay"]
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
        c["put"](c["onrm"], (t * lay.NV + h) * NVCH + ch, ss)
    gpu.barrier()


@traced
def gdn_norm(c, k, t, h, ch):
    """Rows 16 ch ..: RMSNormGated over the head's 128 (every chunk's partial):
    bf16(o rstd w sigmoid(z)) -> ``core`` (the checkpoint's output_gate_type)."""
    tid = c["tid"]
    a = c["args"]
    lay = c["lay"]
    if tid < VCH:
        vr = ch * VCH + tid
        got = c["poll"](
            [(c["onrm"], (t * lay.NV + h) * NVCH + q, 1) for q in range(NVCH)]
            + [(c["proj"], t * lay.QKVZ + lay.CONV + h * HD + vr, 1)]
        )
        ss = f32_of(got[0][0])
        for q in range_constexpr(1, NVCH):
            ss = ss + f32_of(got[q][0])
        r = hw_rsq(ss * fx.Float32(1.0 / HD) + fx.Float32(c["norm_eps"]))
        y = fx.ptr_load(c["ol"] + tid) * r * ld_bf(a["norm_w"], vr)
        y = y * sigmoid(f32_of(got[NVCH][0]))
        y = (k["slot"] > 0).select(y, fx.Float32(0.0))
        bo.buffer_store(y.to(fx.BFloat16), rsrc(a["core"]), t * lay.CORE + h * HD + vr)
    gpu.barrier()


def gdn_round(c, r):
    """Round r's task of this CTA (``bid - goff`` cyclically + ``BLOCKS r``,
    clamped into range), its (token, head, chunk) and the round's LDS."""
    lay = c["lay"]
    gt = gdn_tasks_of(lay)
    n = c["S"] * gt
    task = (c["bid"] + (BLOCKS - c["goff"])) % BLOCKS + BLOCKS * r
    tc = fx.min(task, n - 1)
    cr = dict(c)
    for name, m in (("qkl", 2 * HD), ("vloc", VCH), ("ol", VCH), ("scal", 4)):
        cr[name] = c[name] + r * m
    return cr, task < n, tc // gt, (tc // NVCH) % lay.NV, tc % NVCH


@traced
def gdn_tasks(c, rounds):
    """This CTA's GDN tasks phase by phase: a task waits only on earlier phases
    of other CTAs' tasks, and every CTA finishes a phase's work before it
    waits in the next."""
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
def _put_inj(c, t, r, vs):
    """The injection rows r, r + 1 of token t -> ``inj_out`` [S, HC] bf16."""
    bo.buffer_store(
        bf16_pair(vs[0], vs[1]).bitcast(fx.Int32),
        rsrc(c["args"]["inj_out"]),
        (t * HC + r) // 2,
    )


_BUILDS: dict = {}


def build(key: K1Build):
    if key in _BUILDS:
        return _BUILDS[key]
    s = key.tokens
    need(1 <= s <= L.MAX_TOKENS, f"{s} rows, the kernels take 1..{L.MAX_TOKENS}")
    lay = L.Layout(key.tp)
    scr = scratch_layout(key)
    gt = gdn_tasks_of(lay)
    nr = -(-s * gt // BLOCKS)
    nba = ba_tasks(lay)
    need(BA0 + nba <= BLOCKS, "in_proj_ba's CTAs")
    need(lay.QKVZ // ROWS == 2 * BLOCKS, "in_proj_qkvz: two row groups a CTA")
    need(H % (BA_PARTS * gemv.KCH) == 0, "in_proj_ba's K halves")
    xwords = s * (max(H, hc.DK) + gemv.LDS_PAD) // 2

    @fx.struct
    class XLds:
        xl: fx.Array[fx.Int32, xwords, 16]

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
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        rs: fx.Array[fx.Float32, L.MAX_TOKENS * HC, 16]
        ll: fx.Array[fx.Int32, s * hc.LROW // 2, 16]

    keyed = key_tuple(key, digest())
    name = (
        f"flashnext_mono_k1_s{s}_t{key.tp}_m{int(key.self_msq)}_f{int(key.front_only)}"
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k1(
        state: Int64,
        msq_in: Int64,
        hc_w: Int64,
        w_down: Int64,
        w_up: Int64,
        inj_out: Int64,
        bin_out: Int64,
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
            "S": s,
            "lay": lay,
            "goff": gdn_offset(key),
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "eps": key.eps,
            "norm_eps": key.norm_eps,
            "put": mb.put,
            "put_bf": mb.put_bf,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "rs": lds.rs.ptr,
            "ll": lds.ll.ptr,
            "xl": xl.xl.ptr,
            "qkl": gl.qkl.ptr,
            "vloc": gl.vloc.ptr,
            "ol": gl.ol.ptr,
            "scal": gl.scal.ptr,
            "args": {
                "inj_out": inj_out,
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
        if const_expr(not key.front_only):
            for region in ("proj", "qk", "onrm", "bapart"):
                c[region] = sreg(scratch, scr[region][0], region)
        c["hc"] = {
            "state": state,
            "state_cm": 0,
            "w_norm": hc_w,
            "w_down": w_down,
            "w_up": w_up,
            "msq": sreg(scratch, scr["msq"][0], "msq"),
            "msq_in": None if key.self_msq else msq_in,
            "dpart": sreg(scratch, scr["dpart"][0], "dpart"),
            "lora": sreg(scratch, scr["lora"][0], "lora"),
            "binf": None if key.front_only else sreg(scratch, scr["binf"][0], "binf"),
            "bin": bin_out if key.front_only else scratch + fx.Int64(scr["bin"][0]),
            "inj_out": lambda t, r, vs: _put_inj(c, t, r, vs),
        }
        # ---- the HC front: a CTA's whole sequence in one branch, its weights
        # (and its in_proj_qkvz ones) in flight first
        if const_expr(not key.front_only):
            qops = [qkvz_loads(c, bid + BLOCKS * j) for j in range(2)]
        if const_expr(key.self_msq):  # noqa: SIM102 (traced: `and` does not combine traced values)
            if bid < hc.NRED:
                hc.stage_msq(c, bid)
        if bid < D0:
            uops = hc.up_loads(c, bid)
            hc.load_rrms(c)
            hc.stage_up(c, bid, uops)
        if (bid >= D0) & (bid < hc.UP_TASKS):
            dops = hc.down_loads(c, bid - D0)
            uops = hc.up_loads(c, bid)
            hc.load_rrms(c)
            hc.stage_down(c, bid - D0, dops)
            hc.stage_up(c, bid, uops)
        if const_expr(key.front_only):
            if bid >= hc.UP_TASKS:
                dops = hc.down_loads(c, bid - D0)
                hc.load_rrms(c)
                hc.stage_down(c, bid - D0, dops)
        else:
            is_ba = (bid >= BA0) & (bid < BA0 + nba)
            if is_ba:
                dops = hc.down_loads(c, bid - D0)
                bops = ba_loads(c, bid - BA0)
                hc.load_rrms(c)
                hc.stage_down(c, bid - D0, dops)
                hc.bin_to_lds(c, XROW)
                stage_ba(c, bid - BA0, bops)
            if bid >= BA0 + nba:
                dops = hc.down_loads(c, bid - D0)
                hc.load_rrms(c)
                hc.stage_down(c, bid - D0, dops)
            # ---- in_proj on the block input, then the GDN tasks
            if ~is_ba:
                hc.bin_to_lds(c, XROW)
            quant_x(c, H)
            for j in range_constexpr(2):
                stage_qkvz(c, bid + BLOCKS * j, qops[j])
            gdn_tasks(c, nr)

    @flyc.jit
    def launch(
        state: Int64,
        msq_in: Int64,
        hc_w: Int64,
        w_down: Int64,
        w_up: Int64,
        inj_out: Int64,
        bin_out: Int64,
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
            state,
            msq_in,
            hc_w,
            w_down,
            w_up,
            inj_out,
            bin_out,
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


def _hc_checks(key, s, state, msq_in, hc_w, w_down, w_up, inj_out):
    bf = torch.bfloat16
    need(state.shape == (s, HH) and state.dtype == bf, "state: bf16 [S, 4 H]")
    need(
        (msq_in is None) == key.self_msq
        and (
            msq_in is None
            or (msq_in.numel() == hc.NRED * s * HC and msq_in.dtype == torch.float32)
        ),
        "msq_in: fp32 [20, S, 4], absent with self_msq only",
    )
    need(hc_w.shape == (HH,) and hc_w.dtype == bf, "attention HC norm weight: [4 H]")
    need(
        w_down.shape == (L.DOWN_ROWS, HH) and w_down.dtype == bf, "HC down: [336, 4 H]"
    )
    need(w_up.shape == (HH, L.LORA) and w_up.dtype == bf, "HC up: [4 H, 320]")
    need(inj_out.shape == (s, HC) and inj_out.dtype == bf, "inj_out: bf16 [S, 4]")
    tensors = [state, hc_w, w_down, w_up, inj_out]
    tensors += [msq_in] if msq_in is not None else []
    need(all(t.is_contiguous() for t in tensors), "a non-contiguous HC input")


def hc_front(
    key: K1Build,
    *,
    state,
    msq_in,
    hc_w,
    w_down,
    w_up,
    inj_out,
    bin_out,
    scratch,
    epoch,
    layer,
):
    """Host launcher of the HC front alone (``key.front_only``, a QSA layer):
    the attention HC's norm and mix of ``state`` [S, 4 H] -> ``bin_out`` [S, H]
    bf16 (the attention's input) and ``inj_out`` [S, 4] bf16. ``msq_in``:
    ``state``'s sums of squares as K2 wrote them, None with ``key.self_msq``."""
    s = key.tokens
    need(key.front_only, "hc_front takes a front_only build")
    _hc_checks(key, s, state, msq_in, hc_w, w_down, w_up, inj_out)
    need(
        bin_out.shape == (s, H)
        and bin_out.dtype == torch.bfloat16
        and bin_out.is_contiguous(),
        "bin_out: bf16 [S, H]",
    )
    vals = dict.fromkeys(ABI.names, 0)
    vals |= {
        "state": state.data_ptr(),
        "msq_in": 0 if msq_in is None else msq_in.data_ptr(),
        "hc_w": hc_w.data_ptr(),
        "w_down": w_down.data_ptr(),
        "w_up": w_up.data_ptr(),
        "inj_out": inj_out.data_ptr(),
        "bin_out": bin_out.data_ptr(),
        "scratch": scratch.data_ptr(),
        "epoch": epoch.data_ptr(),
        "layer": layer,
    }
    build(key)(*ABI.pack(vals), stream=torch.cuda.current_stream())


def gdn_pre(
    key: K1Build,
    *,
    state,
    msq_in,
    hc_w,
    w_down,
    w_up,
    inj_out,
    w_qkvz,
    w_qkvzs,
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
):
    """Host launcher of a GDN layer's K1; every weight as vLLM holds it after
    loading (in_proj_qkvz online MXFP4: the ASM FP4 GEMM's shuffled weight and
    e8m0 scales). ``conv_state`` [slots, CONV, 3] or a transposed view of
    [slots, 3, CONV] (strides read); ``rstate`` [slots, NV, 128, 128] fp32, a
    slot dense, slots strided; ``st_idx`` [>= S] int32, <= 0 a pad row;
    ``epoch`` a device int32 bumped once a step, ``layer`` this launch's tag
    slot."""
    s = key.tokens
    lay = L.Layout(key.tp)
    bf = torch.bfloat16
    need(not key.front_only, "gdn_pre takes a GDN build")
    _hc_checks(key, s, state, msq_in, hc_w, w_down, w_up, inj_out)
    need(
        core.shape == (s, lay.CORE) and core.dtype == bf, f"core: bf16 [S, {lay.CORE}]"
    )
    need(
        w_qkvz.shape == (lay.QKVZ, H // 2)
        and w_qkvz.dtype in (torch.uint8, torch.float4_e2m1fn_x2),
        f"in_proj_qkvz: fp4x2 [{lay.QKVZ}, {H // 2}]",
    )
    need(
        w_qkvzs.numel() == lay.QKVZ * H // 32 and w_qkvzs.element_size() == 1,
        "in_proj_qkvz scales: e8m0, one a 32-block",
    )
    need(
        w_ba.shape == (lay.BA, H) and w_ba.dtype == bf,
        f"in_proj_ba: bf16 [{lay.BA}, {H}]",
    )
    need(conv_w.numel() == lay.CONV * L.CONV_W and conv_w.dtype == bf, "conv1d weight")
    need(
        conv_state.dtype == bf and conv_state.shape[1:] == (lay.CONV, SL),
        f"conv state: bf16 [slots, {lay.CONV}, 3]",
    )
    need(
        rstate.dtype == torch.float32 and rstate.shape[1:] == (lay.NV, HD, HD),
        f"recurrent state: fp32 [slots, {lay.NV}, 128, 128]",
    )
    need(rstate[0].is_contiguous(), "a recurrent state slot must be dense")
    need(a_log.dtype == torch.float32 and dt_bias.dtype == bf, "A_log / dt_bias dtype")
    need(norm_w.dtype == bf and norm_w.numel() == HD, "gated norm weight")
    need(st_idx.dtype == torch.int32 and st_idx.numel() >= s, "state indices")
    dense = (w_qkvz, w_qkvzs, w_ba, conv_w, core)
    need(all(t.is_contiguous() for t in dense), "a non-contiguous GDN input")
    vals = {
        "state": state.data_ptr(),
        "msq_in": 0 if msq_in is None else msq_in.data_ptr(),
        "hc_w": hc_w.data_ptr(),
        "w_down": w_down.data_ptr(),
        "w_up": w_up.data_ptr(),
        "inj_out": inj_out.data_ptr(),
        "bin_out": 0,
        "w_qkvz": w_qkvz.data_ptr(),
        "w_qkvzs": w_qkvzs.data_ptr(),
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
    build(key)(*ABI.pack(vals), stream=torch.cuda.current_stream())
