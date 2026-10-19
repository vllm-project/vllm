# SPDX-License-Identifier: Apache-2.0
"""K1 for Qwen3.8-27B at TP1 or TP2 (``dense_gdn_pre``): one GDN layer of a
decode step from the layer's input (the previous layer's FFN output + residual;
at TP2 after its all-reduce) to the gated core (out_proj's input, ``dense_ffn``'s
with ``oproj``), S <= 8 rows, one token a request. At TP2 a rank runs its half
of the heads (``Geo``) on the full input; nothing crosses ranks. One launch,
``BLOCKS`` x ``THREADS`` (sizes at TP1, TP2 in brackets):

    front    (CTAs 0 .. 159, 32 hidden columns each): h + residual -> ``res_out``;
             sum-of-squares partials -> rstd -> input_layernorm (Gemma, 1 + w)
             -> bf16 x -> MXFP4 blocks put for every CTA -> the LDS x rows
             (``dense_ffn``'s distributed norm). With ``normed`` (the stock
             input norm ran) every CTA quantizes the normed rows itself
    in_proj  in_proj_qkvz: 1024 [512] groups of 16 rows, CTA b groups 4 b ..
             [2 b ..]; wave w group w / 2, K half w % 2 -> bf16 PROJ entries.
             in_proj_ba: 6 [3] groups x 8 K eighths on CTAs 0 .. 47 [23] -> fp32
             partials (BAP)
    gdn      tasks of 8 state rows (48 [24] value heads x 16 chunks a token:
             3 S [ceil(1.5 S)] a CTA), phase by phase over the CTA's tasks so
             each hand-off is waited on once:
               conv   the task's 8 v channels; the key head's first value head
                      also its 8 q and 8 k channels (out through QK). Window
                      updated in place, products rounded to bf16, SiLU -> bf16
               gates  q / k of the key head (QK) l2-normed, q * HD^-1/2; b, a
                      (the BAP partials in order, bf16) -> beta, decay
               recur  the delta rule on the 8 rows (fp32 state in place) ->
                      bf16 o -> OV
               norm   RMSNormGated over the head (all 128 o from OV) * silu(z)
                      -> the task's 8 core rows
    out_proj (``oproj``, TP2 from ``normed``): every CTA's core rows ready (one
             flag a CTA) -> CTAs 0 .. 159 qdq the S core rows and run their 32
             hidden columns of out_proj's K share (``dense_ffn``'s o_proj stage)
             in the stock GEMM's K splits (``o_splits``), each chained in order,
             the splits summed in order -> bf16 ``out``, the rank's partial

Bit-exact with the stock decode path (the Triton conv update,
``fused_recurrent_gated_delta_rule_packed_decode``, RMSNormGated): the same
rounding, lane layouts, reduction orders, FMA contractions and transcendental
lowerings as their gfx950 code. A row whose state slot is -1 (a pad row) writes
no state and a zero core row, as stock.

Weights: in_proj_qkvz [q | k | v | z] and in_proj_ba [b | a] (the rank's) as
MXFP4 in the (16, 16)-shuffled order (``dense_ffn.shuffle_w``), e8m0 scales [N, K / 32].
"""

import struct
from dataclasses import dataclass
from typing import NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.qwen3_5.amd.mono.common.ops import (
    CM_DEV,
    CM_NT,
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    hw_rsq,
    row_sum,
    rsrc,
    traced,
    uniform,
    xred,
)
from vllm.models.qwen3_5.amd.mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    KernelAbi,
    key_tuple,
    pair_layout,
)
from vllm.models.qwen3_5.amd.mono.common.sync import Mailbox, publish, sreg
from vllm.models.qwen3_5.amd.mono import dense_ffn as D
from vllm.models.qwen3_5.amd.mono.helpers import step_tag
from vllm.models.qwen3_5.amd.mono.sources import digest

HIDDEN, ROWS = D.HIDDEN, D.ROWS
HD = 128
NK = 16
NV = 48
HPK = NV // NK  # 3 value heads a key head
KEY = NK * HD  # 2048
VAL = NV * HD  # 6144
QKVZ = 2 * KEY + 2 * VAL  # 16384
BA = 2 * NV  # 96
CONV = 2 * KEY + VAL  # 10240 conv channels: q | k | v
CORE = VAL
CONV_W = 4
SL = CONV_W - 1

STEPS = HIDDEN // D.STEP  # 40
Q_HALF = STEPS // 2  # 20
BA_E = 8  # K eighths a ba group
BA_STEPS = STEPS // BA_E  # 5
W_DW = HIDDEN // 8  # 640 dwords a weight row
WS_DW = HIDDEN // 32 // 4  # 40 scale dwords a row

VCH = 8  # state rows a task
NVCH = HD // VCH  # 16
CI = 3 * VCH  # conv items a task: 8 v, 8 q, 8 k
SOFTPLUS_THRESHOLD = 20.0
LOG2E = 1.4426950408889634
# Lane-reduction orders of the stock Triton kernels (num_warps=1) on gfx950: a
# 128-vector two elements a lane (DPP row_shr 8..1, row_bcast 15 / 31), and a
# state row of 32 lanes x 4 columns.
VEC_ORDER = D.WAVE_ORDER
ROW_ORDER = (16, 8, 4, 2, 1)

assert VCH == WAVES and BA_STEPS * BA_E == STEPS
TP_SIZES = (1, 2)
# The stock out_proj GEMM's split-K at TP2 (aiter get_splitk, K 3072): fp32
# sums are not associative, so the fold keeps its partition. This image's
# DEFAULT config gives 4 splits up to M 8, and 1 at M 9..16.
O_SPLITS = 4


def o_splits(s: int) -> int:
    return O_SPLITS if s <= 8 else 1


class Geo(NamedTuple):
    """A rank's GDN shapes at tensor parallel size ``tp``: 1 / tp of the key and
    value heads, each of in_proj_qkvz's q | k | v | z, in_proj_ba's b | a and
    the conv's q | k | v sliced contiguously (the stock column-parallel split)."""

    tp: int

    @property
    def nk(self) -> int:
        return NK // self.tp

    @property
    def nv(self) -> int:
        return NV // self.tp

    @property
    def key(self) -> int:
        return self.nk * HD

    @property
    def val(self) -> int:
        return self.nv * HD

    @property
    def qkvz(self) -> int:
        return 2 * self.key + 2 * self.val

    @property
    def ba(self) -> int:
        return 2 * self.nv

    @property
    def conv(self) -> int:
        return 2 * self.key + self.val

    @property
    def qg_cta(self) -> int:
        """qkvz groups of 16 rows a CTA (4 at TP1, 2 at TP2): wave w group w / 2."""
        return self.qkvz // ROWS // BLOCKS

    @property
    def ba_ctas(self) -> int:
        """CTAs of in_proj_ba: its groups x K eighths (48 at TP1, 24 at TP2)."""
        return self.ba // ROWS * BA_E

    @property
    def tpt(self) -> int:
        """Tasks a token (768 at TP1, 384 at TP2)."""
        return self.nv * NVCH


for _g in map(Geo, TP_SIZES):
    assert _g.qg_cta * ROWS * BLOCKS == _g.qkvz and 2 * _g.qg_cta <= WAVES
    assert _g.ba % ROWS == 0 and _g.ba_ctas <= BLOCKS and _g.nv % _g.nk == 0


def n_tasks(geo: Geo, s: int) -> int:
    """Tasks a CTA at S rows: task k of CTA b is ``k BLOCKS + b`` of the S x tpt
    (3 S at TP1; at TP2 ceil(1.5 S), the last one dead on half the CTAs at odd S)."""
    return (geo.tpt * s + BLOCKS - 1) // BLOCKS


_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class K1Build:
    tokens: int
    first: bool = False  # the model's first layer: no residual (residual := hidden)
    eps: float = 1e-6  # input_layernorm
    norm_eps: float = 1e-6  # the gated norm
    tp: int = 1  # this rank's heads (``Geo``)
    # hidden is input_layernorm's output (the stock norm ran): no add, no norm
    normed: bool = False
    # out_proj folded in (``normed`` only): ``out`` gets its bf16 [S, HIDDEN]
    # output (the rank's partial at TP2), ``w_o`` / ``w_os`` its shuffled weights
    oproj: bool = False
    stamps: bool = False  # debug: per-CTA s_memrealtime stamps -> scratch "stamps"
    stop: int = 99  # debug: phases < stop (1 front, 2 in_proj, 3 gdn, 4 out_proj)


ABI = KernelAbi(
    (
        "hidden",
        "residual",
        "res_out",
        "ln_w",
        "w_qkvz",
        "w_qkvzs",
        "w_ba",
        "w_bas",
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
        "w_o",
        "w_os",
        "out",
        "scratch",
        "epoch",
        "layer",
    )
)


def scratch_layout(key: K1Build) -> dict:
    s, geo = key.tokens, Geo(key.tp)
    lay = pair_layout(
        (
            ("nsum", D.CHUNKS * s),
            ("xmb", s * D.O_CTAS * D.X_PAIRS),
            ("proj", s * geo.qkvz),
            ("bap", geo.ba_ctas * s * ROWS),
            ("qk", s * geo.nk * 2 * HD),
            ("ov", s * geo.nv * HD),
            ("ordy", BLOCKS),
        )
    )
    end = max(o + n for o, n in lay.values())
    lay["stamps"] = ((end + 255) // 256 * 256, BLOCKS * D.STAMPS * 8)
    return lay


def scratch_bytes(key: K1Build) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def f32_of(w):
    return w.bitcast(fx.Float32)


def ld_bf(ptr, i):
    return fx.Float32(
        fx.BFloat16(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.bf16))
    )


def ld_f32(ptr, i):
    return fx.Float32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.f32))


def _llvm_f32(name, x):
    return fx.Float32(
        llvm.call_intrinsic(T.f32, name, [fx.Float32(x).ir_value()], [], [])
    )


def _f32_bits(u):
    return struct.unpack("<f", struct.pack("<I", u))[0]


LN2_HI, LN2_LO, LN2_32 = (
    _f32_bits(0x3F317217),
    _f32_bits(0x3377D1CF),
    _f32_bits(0x41B17218),
)
F32_TINY = _f32_bits(0x00800000)


def _ldexp(x, n):
    return fx.Float32(
        llvm.call_intrinsic(
            T.f32,
            "llvm.ldexp.f32.i32",
            [fx.Float32(x).ir_value(), fx.Int32(n).ir_value()],
            [],
            [],
        )
    )


def tl_exp(x):
    """``tl.exp`` as RMSNormGated's code computes it: llvm.exp2(x * log2(e))."""
    return _llvm_f32("llvm.exp2.f32", x * fx.Float32(LOG2E))


def conv_silu(z):
    """The stock conv update's SiLU: z / (1 + tl.exp(-z))."""
    return z / (fx.Float32(1.0) + tl_exp(-z))


def tl_exp_fma(x):
    """``tl.exp`` as the stock decode kernel's code computes it: llvm.exp2's
    range scaling (x < -126: +64, then 2^-64) contracted with x * log2(e)."""
    small = x * fx.Float32(LOG2E) < fx.Float32(-126.0)
    arg = fmath.fma(
        x, fx.Float32(LOG2E), small.select(fx.Float32(64.0), fx.Float32(0.0))
    )
    return _ldexp(
        _llvm_f32("llvm.amdgcn.exp2.f32", arg), small.select(fx.Int32(-64), fx.Int32(0))
    )


def tl_log_fma(y):
    """``tl.log`` as the stock decode kernel's code computes it: llvm.log's
    expansion (v_log_f32 times an extended ln 2) with its last add contracted."""
    small = y < fx.Float32(F32_TINY)
    r = _llvm_f32(
        "llvm.amdgcn.log.f32", _ldexp(y, small.select(fx.Int32(32), fx.Int32(0)))
    )
    hi = r * fx.Float32(LN2_HI)
    lo = fx.Float32(fmath.fma(r, fx.Float32(LN2_HI), -hi))
    lo = fx.Float32(fmath.fma(r, fx.Float32(LN2_LO), lo))
    ln = fx.Float32(fmath.fma(r, fx.Float32(LN2_HI), lo))
    ln = (fmath.absf(r) < fx.Float32(float("inf"))).select(ln, r)
    return ln - small.select(fx.Float32(LN2_32), fx.Float32(0.0))


def dot4(x, y):
    """Sum of x[j] y[j] the way the stock kernel's ISA forms it from the IR's
    fmul / fadd tree (fp-contract): fma(x0, y0, x2 y2) + fma(x1, y1, x3 y3)."""
    return fx.Float32(fmath.fma(x[0], y[0], x[2] * y[2])) + fx.Float32(
        fmath.fma(x[1], y[1], x[3] * y[3])
    )


# ---------------------------------------------------------------- front
def front_loads(c):
    """The norm thread (t, j)'s hidden, residual and 1 + w words (8 columns of
    the owner's 32)."""
    a = c["args"]
    t, j = D.o_lane(c)
    col = 2 * ROWS * D.o_owner(c) + 8 * j
    hw = D.ld4(rsrc(a["hidden"]), (t * HIDDEN + col) // 2)
    rw = hw if c["first"] else D.ld4(rsrc(a["residual"]), (t * HIDDEN + col) // 2)
    return hw, rw, D.ld4(rsrc(a["ln_w"]), col // 2)


@traced
def front_z(c, raw):
    """z = h + residual (fp32; h on the first layer) -> res_out; the chunk's sum
    of squares -> NSUM. Returns z, 1 + w."""
    s, tid, bid = c["S"], c["tid"], c["bid"]
    t, j = D.o_lane(c)
    hw, rw, wv = raw
    z, lnw = [], []
    for d in range_constexpr(4):
        if const_expr(c["first"]):
            z += [bf_lo(hw[d]), bf_hi(hw[d])]
        else:
            z += [bf_lo(hw[d]) + bf_lo(rw[d]), bf_hi(hw[d]) + bf_hi(rw[d])]
        lnw += [fx.Float32(1.0) + bf_lo(wv[d]), fx.Float32(1.0) + bf_hi(wv[d])]
    if (tid < s * 4) & (bid < D.O_CTAS):
        bo.buffer_store(
            fx.Vector.from_elements(D.pack_bf(z), fx.Int32),
            rsrc(c["args"]["res_out"]),
            (t * HIDDEN + 2 * ROWS * bid + 8 * j) // 2,
        )
        c["put"](c["nsum"], t * D.CHUNKS + 4 * bid + j, D.chunk_ss(z))
    return z, lnw


def x_loads(c):
    """``normed``: chunk ``tid + 512 i`` (8 columns) of the S normed rows."""
    s, tid, a = c["S"], c["tid"], c["args"]
    n = s * D.CHUNKS
    return [
        D.ld4(rsrc(a["hidden"]), fx.min(tid + THREADS * i, n - 1) * 4)
        for i in range((n + THREADS - 1) // THREADS)
    ]


@traced
def x_quant(c, raw):
    """``normed``: every CTA quantizes all S rows itself, by 32 columns (four
    lanes a block, as ``dense_ffn.run_norm``) -> the LDS x rows."""
    s, tid = c["S"], c["tid"]
    n = s * D.CHUNKS
    for i in range_constexpr(len(raw)):
        ch_raw = tid + THREADS * i
        ch = fx.min(ch_raw, n - 1)
        t = ch // D.CHUNKS
        col = ch % D.CHUNKS * 8
        x = []
        for d in range_constexpr(4):
            x += [bf_lo(raw[i][d]), bf_hi(raw[i][d])]
        amax = fx.Float32(0.0)
        for e in range_constexpr(8):
            amax = fx.max(amax, fx.Float32(fmath.absf(x[e])))
        amax = xred(xred(amax, 1, fx.max), 2, fx.max)
        code = D.mx_code(amax)
        w = D.pack8(x, code)
        if ch_raw < n:
            fx.ptr_store(w, c["xq"] + (t * D.XQROW + col // 8))
            if ch % 4 == 0:
                fx.ptr_store(code, c["xs"] + (t * D.XSROW + col // 32))
    gpu.barrier()


# ---------------------------------------------------------------- in_proj
def wloads(c, w, ws, nrows, row, steps):
    """This lane's tiles of (shuffled) weight row ``row`` at K steps ``steps``
    and their scale words; a row past ``nrows`` reads zeros, no traffic."""
    lane = c["lane"]
    r_w = rsrc(w, nrows * HIDDEN // 2)
    r_s = rsrc(ws, nrows * HIDDEN // 32)
    tiles = [
        D.ld4(r_w, row // ROWS * (ROWS * W_DW) + st * 256 + lane * 4, CM_NT)
        for st in steps
    ]
    words = [
        fx.Int32(bo.buffer_load(r_s, row * WS_DW + st, vec_width=1, dtype=T.i32))
        for st in steps
    ]
    return tiles, words


def qkvz_loads(c):
    """Wave w: group w / 2 of the CTA's, K half w % 2; past the weight (no
    traffic) when the CTA has fewer groups than 4."""
    a, lane, wave, bid, geo = c["args"], c["lane"], c["wave"], c["bid"], c["G"]
    row = (geo.qg_cta * bid + wave // 2) * ROWS + lane % ROWS
    row = (wave < 2 * geo.qg_cta).select(row, fx.Int32(geo.qkvz))
    return wloads(
        c,
        a["w_qkvz"],
        a["w_qkvzs"],
        geo.qkvz,
        row,
        [wave % 2 * Q_HALF + i for i in range(Q_HALF)],
    )


def ba_loads(c):
    """CTA b < ba_ctas: group b / 8, K eighth b % 8, wave w < 5 its step w; the
    other waves and CTAs past the weight."""
    a, lane, wave, bid, geo = c["args"], c["lane"], c["wave"], c["bid"], c["G"]
    row = (bid // BA_E) * ROWS + lane % ROWS
    row = ((bid < geo.ba_ctas) & (wave < BA_STEPS)).select(row, fx.Int32(geo.ba))
    st = (bid % BA_E) * BA_STEPS + fx.min(wave, BA_STEPS - 1)
    return wloads(c, a["w_ba"], a["w_bas"], geo.ba, row, [st])


@traced
def stage_qkvz(c, ops):
    """The CTA's groups: the two K halves in order -> bf16 PROJ."""
    s, tid, lane, wave, red, bid = (
        c["S"],
        c["tid"],
        c["lane"],
        c["wave"],
        c["red"],
        c["bid"],
    )
    acc = D.gu_mfmas(c, ops[0], ops[1], [wave % 2 * Q_HALF + i for i in range(Q_HALF)])
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    geo = c["G"]
    if tid < geo.qg_cta * ROWS * s:
        q = tid // (ROWS * s)
        rem = tid % (ROWS * s)
        t = rem // ROWS
        r = rem % ROWS
        v = bf16_round(row_sum(red, r, t, waves=(2 * q, 2 * q + 1)))
        c["put"](c["proj"], t * geo.qkvz + (geo.qg_cta * bid + q) * ROWS + r, v)
    gpu.barrier()


@traced
def stage_ba(c, ops):
    """CTAs below ba_ctas: the eighth's fp32 partial (waves 0 .. 4 in order) -> BAP."""
    s, tid, lane, wave, red, bid = (
        c["S"],
        c["tid"],
        c["lane"],
        c["wave"],
        c["red"],
        c["bid"],
    )
    gl = lane // ROWS
    t = fx.min(lane % ROWS, s - 1)
    st = (bid % BA_E) * BA_STEPS + fx.min(wave, BA_STEPS - 1)
    acc = D.scaled_step(
        c["xq"],
        c["xs"],
        t * D.XQROW,
        t * D.XSROW,
        st,
        gl,
        ops[0][0],
        ops[1][0],
        fx.Vector.filled(4, 0.0, fx.Float32),
    )
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if (bid < c["G"].ba_ctas) & (tid < ROWS * s):
        t2 = tid // ROWS
        r = tid % ROWS
        v = row_sum(red, r, t2, waves=tuple(range(BA_STEPS)))
        c["put"](c["bap"], (bid * s + t2) * ROWS + r, v)


# ---------------------------------------------------------------- gdn
def task_of(c, k):
    """The CTA's task k (``n_tasks``) -> token, value head, chunk and whether it
    exists (None: all of them do). A dead task aliases the last live one: it may
    read, never write."""
    s, geo = c["S"], c["G"]
    if geo.tpt % BLOCKS == 0:
        # A whole number of tasks a token a CTA (TP1): no division by tpt.
        tpc = geo.tpt // BLOCKS
        w = (k % tpc) * BLOCKS + c["bid"]
        return k // tpc, w // NVCH, w % NVCH, None
    n = geo.tpt * s
    g = k * BLOCKS + c["bid"]
    ok = None if n % BLOCKS == 0 else g < n
    if ok is not None:
        g = fx.min(g, n - 1)
    w = g % geo.tpt
    return g // geo.tpt, w // NVCH, w % NVCH, ok


def _live(pred, ok):
    return pred if ok is None else pred & ok


def recur_task(c, m):
    """Task pair m: waves 0 .. 3 take task 2 m, waves 4 .. 7 task 2 m + 1 (clamped)
    and whether it exists. Wave w % 4 holds rows 2 (w % 4) + lane / 32 of it."""
    nt = n_tasks(c["G"], c["S"])
    kk = 2 * m + c["wave"] // 4
    return fx.min(kk, nt - 1), kk < nt


def gdn_loads(c):
    """Every task pair's slot and state rows (the stock layout: 4 columns
    4 (lane % 32) .. a lane), issued before the first GDN hand-off; per conv
    item its window and weights."""
    s, tid, a, geo = c["S"], c["tid"], c["args"], c["G"]
    nt = n_tasks(geo, s)
    g = {"valid": [], "live": [], "sbase": [], "st": [], "soff": []}
    for m in range((nt + 1) // 2):
        k, valid = recur_task(c, m)
        t, h, ch, ok = task_of(c, k)
        valid = _live(valid, ok)
        slot = uniform(
            fx.Int32(bo.buffer_load(rsrc(a["st_idx"]), t, vec_width=1, dtype=T.i32))
        )
        sbase = a["rstate"] + fx.Int64(fx.max(slot, 0)) * fx.Int64(a["rs_seq"]) * 4
        row = ch * VCH + 2 * (c["wave"] % 4) + c["lane"] // 32
        soff = (h * HD + row) * HD + 4 * (c["lane"] % 32)
        g["valid"].append(valid)
        g["live"].append(valid & (slot >= 0))
        g["sbase"].append(sbase)
        g["soff"].append(soff)
        g["st"].append(
            fx.Vector(bo.buffer_load(rsrc(sbase), soff, vec_width=4, dtype=T.f32))
        )
    # conv items: item i = tid + 512 r -> task i / 24, channel i % 24 (8 v, 8 q, 8 k)
    ni = nt * CI
    items = []
    for r in range((ni + THREADS - 1) // THREADS):
        i = fx.min(tid + THREADS * r, ni - 1)
        k = i // CI
        cc = i % CI
        t, h, ch, ok = task_of(c, k)
        kh = h // HPK
        part = cc // VCH  # 0 v, 1 q, 2 k
        d = cc % VCH
        chan = (part == 0).select(
            2 * geo.key + h * HD + ch * VCH + d,
            (part - 1) * geo.key + kh * HD + ch * VCH + d,
        )
        slot = fx.Int32(bo.buffer_load(rsrc(a["st_idx"]), t, vec_width=1, dtype=T.i32))
        cs = a["conv_st"] + fx.Int64(fx.max(slot, 0)) * fx.Int64(a["cs_seq"]) * 2
        co = chan * a["cs_dim"]
        win = [ld_bf(cs, co + j * a["cs_tok"]) for j in range(SL)]
        cw = fx.Vector(
            bo.buffer_load(rsrc(a["conv_w"]), chan * 2, vec_width=2, dtype=T.i32)
        )
        live = (tid + THREADS * r < ni) & ((part == 0) | (h % HPK == 0))
        items.append(
            {
                "k": k,
                "t": t,
                "h": h,
                "ch": ch,
                "kh": kh,
                "part": part,
                "d": d,
                "chan": chan,
                "slot": slot,
                "cs": cs,
                "co": co,
                "win": win,
                "cw": [bf_lo(cw[0]), bf_hi(cw[0]), bf_lo(cw[1]), bf_hi(cw[1])],
                "live": _live(live, ok),
            }
        )
    g["items"] = items
    return g


@traced
def gdn_conv(c, g):
    """Every conv item: x (PROJ) -> window shifted in place; the width-4 conv
    (products rounded to bf16, fp32 sum in tap order), SiLU -> bf16; v -> LDS,
    q / k -> QK."""
    a, geo = c["args"], c["G"]
    items = g["items"]
    got = c["poll"]([(c["proj"], it["t"] * geo.qkvz + it["chan"], 1) for it in items])
    for n in range_constexpr(len(items)):
        it = items[n]
        x = f32_of(got[n][0])
        win, w = it["win"], it["cw"]
        if it["live"] & (it["slot"] >= 0):
            rs = rsrc(it["cs"])
            bo.buffer_store(fx.Float32(win[1]).to(fx.BFloat16), rs, it["co"])
            bo.buffer_store(
                fx.Float32(win[2]).to(fx.BFloat16), rs, it["co"] + a["cs_tok"]
            )
            bo.buffer_store(
                fx.Float32(x).to(fx.BFloat16), rs, it["co"] + 2 * a["cs_tok"]
            )
        acc = bf16_round(w[0] * win[0]) + bf16_round(w[1] * win[1])
        acc = acc + bf16_round(w[2] * win[2])
        acc = acc + bf16_round(w[3] * x)
        o = bf16_round(conv_silu(acc))
        if it["live"] & (it["part"] == 0):
            fx.ptr_store(o, c["vloc"] + (it["k"] * VCH + it["d"]))
        if it["live"] & (it["part"] > 0):
            at = (
                ((it["t"] * geo.nk + it["kh"]) * 2 + it["part"] - 1) * HD
                + it["ch"] * VCH
                + it["d"]
            )
            c["put"](c["qk"], at, o)
    gpu.barrier()


@traced
def gdn_gates(c):
    """Every task's q, k (QK) -> LDS; b, a (BAP, the eighths in order) -> beta,
    decay; then q / k l2-normed (q * HD^-1/2)."""
    s, tid, lane, wave, a, geo = (
        c["S"],
        c["tid"],
        c["lane"],
        c["wave"],
        c["args"],
        c["G"],
    )
    nt = n_tasks(geo, s)
    ni = nt * 2 * HD
    specs, dst = [], []
    for r in range_constexpr((ni + THREADS - 1) // THREADS):
        i = fx.min(tid + THREADS * r, ni - 1)
        k = i // (2 * HD)
        e = i % (2 * HD)
        t, h, _, _ = task_of(c, k)
        specs.append((c["qk"], (t * geo.nk + h // HPK) * 2 * HD + e, 1))
        dst.append(i)
    got = c["poll"](specs)
    for r in range_constexpr(len(specs)):
        if tid + THREADS * r < ni:
            fx.ptr_store(f32_of(got[r][0]), c["qkl"] + dst[r])
    if tid < 2 * nt:
        k = tid // 2
        which = tid % 2  # 0 b, 1 a
        t, h, _, _ = task_of(c, k)
        row = which * geo.nv + h
        gb = row // ROWS
        r = row % ROWS
        got2 = c["poll"](
            [(c["bap"], ((gb * BA_E + e) * s + t) * ROWS + r, 1) for e in range(BA_E)]
        )
        v = f32_of(got2[0][0])
        for e in range_constexpr(1, BA_E):
            v = v + f32_of(got2[e][0])
        v = bf16_round(v)
        x = v + ld_bf(a["dt_bias"], h)
        sp = tl_log_fma(fx.Float32(1.0) + tl_exp_fma(x))
        sp = (x <= fx.Float32(SOFTPLUS_THRESHOLD)).select(sp, x)
        decay = tl_exp_fma(-tl_exp_fma(ld_f32(a["a_log"], h)) * sp)
        beta = bf16_round(fx.Float32(1.0) / (fx.Float32(1.0) + tl_exp_fma(-v)))
        fx.ptr_store((which == 0).select(beta, decay), c["scal"] + (k * 2 + which))
    gpu.barrier()
    for n in range_constexpr((2 * nt + WAVES - 1) // WAVES):
        vi = fx.min(wave + WAVES * n, 2 * nt - 1)
        base = c["qkl"] + vi * HD
        x0 = fx.ptr_load(base + 2 * lane)
        x1 = fx.ptr_load(base + (2 * lane + 1))
        nrm = D.tl_sqrt(butterfly(x0 * x0 + x1 * x1, VEC_ORDER) + fx.Float32(1e-6))
        y0, y1 = x0 / nrm, x1 / nrm
        isq = vi % 2 == 0
        y0 = isq.select(y0 * fx.Float32(HD**-0.5), y0)
        y1 = isq.select(y1 * fx.Float32(HD**-0.5), y1)
        if wave + WAVES * n < 2 * nt:
            fx.ptr_store(y0, base + 2 * lane)
            fx.ptr_store(y1, base + (2 * lane + 1))
    gpu.barrier()


@traced
def gdn_recur(c, g):
    """Task pair by pair (``recur_task``), the stock decode kernel's arithmetic:
    decay, the delta rule, the new state (in place), o = bf16(h q) -> OV."""
    lane, wave, geo = c["lane"], c["wave"], c["G"]
    nt = n_tasks(geo, c["S"])
    col = 4 * (lane % 32)
    for m in range_constexpr((nt + 1) // 2):
        k, _ = recur_task(c, m)
        t, h, ch, _ = task_of(c, k)
        row = 2 * (wave % 4) + lane // 32
        live = g["live"][m]
        beta = fx.ptr_load(c["scal"] + 2 * k)
        decay = fx.ptr_load(c["scal"] + (2 * k + 1))
        qb = c["qkl"] + k * 2 * HD
        q = [fx.ptr_load(qb + (col + j)) for j in range(4)]
        kv = [fx.ptr_load(qb + (HD + col + j)) for j in range(4)]
        st = g["st"][m]
        hd = [live.select(st[j], fx.Float32(0.0)) * decay for j in range(4)]
        p = butterfly(dot4(hd, kv), ROW_ORDER)
        vn = (fx.ptr_load(c["vloc"] + (k * VCH + row)) - p) * beta
        hn = [fx.Float32(fmath.fma(vn, kv[j], hd[j])) for j in range(4)]
        o = bf16_round(butterfly(dot4(hn, q), ROW_ORDER))
        if live:
            bo.buffer_store(
                fx.Vector.from_elements(hn, fx.Float32),
                rsrc(g["sbase"][m]),
                g["soff"][m],
            )
        if g["valid"][m] & (lane % 32 == 0):
            c["put"](
                c["ov"],
                (t * geo.nv + h) * HD + ch * VCH + row,
                live.select(o, fx.Float32(0.0)),
            )


@traced
def gdn_norm(c, g):
    """Wave by task: the head's 128 o (OV, two a lane, the stock RMSNormGated
    layout and order) -> rstd; the task's 8 rows bf16(o rstd w silu(z)) -> core."""
    lane, wave, a, geo = c["lane"], c["wave"], c["args"], c["G"]
    nt = n_tasks(geo, c["S"])
    for n in range_constexpr((nt + WAVES - 1) // WAVES):
        k = fx.min(wave + WAVES * n, nt - 1)
        t, h, ch, ok = task_of(c, k)
        e = h * HD + 2 * lane
        got = c["poll"](
            [
                (c["ov"], t * geo.nv * HD + e, 2),
                (c["proj"], t * geo.qkvz + geo.conv + e, 2),
            ]
        )
        x = [f32_of(got[0][j]) for j in range(2)]
        rstd = hw_rsq(
            butterfly(x[0] * x[0] + x[1] * x[1], VEC_ORDER) * fx.Float32(1.0 / HD)
            + fx.Float32(c["norm_eps"])
        )
        y = []
        for j in range_constexpr(2):
            z = f32_of(got[1][j])
            sig = fx.Float32(1.0) / (fx.Float32(1.0) + tl_exp(-z))
            y.append(x[j] * rstd * ld_bf(a["norm_w"], 2 * lane + j) * (sig * z))
        if _live((wave + WAVES * n < nt) & (lane // (VCH // 2) == ch), ok):
            # with oproj read by other CTAs: past the CU's cache
            bo.buffer_store(
                D.pack_bf(y)[0],
                rsrc(a["core"]),
                (t * geo.val + e) // 2,
                cache_modifier=CM_DEV if c["oproj"] else 0,
            )


# ---------------------------------------------------------------- out_proj
@traced
def o_core(c):
    """Every CTA's core rows ready -> the S core rows through MXFP4 qdq (a
    thread a 1 x 32 block) -> the LDS o rows."""
    s, tid, a, geo = c["S"], c["tid"], c["args"], c["G"]
    if tid < BLOCKS:
        c["poll"]([(c["ordy"], tid, 1)])
    gpu.barrier()
    nblk = geo.val // 32
    nb = s * nblk
    raw = []
    for i in range_constexpr((nb + THREADS - 1) // THREADS):
        b = fx.min(tid + THREADS * i, nb - 1)
        at = (b // nblk * geo.val + 32 * (b % nblk)) // 2
        raw.append([D.ld4(rsrc(a["core"]), at + 4 * k, CM_DEV) for k in range(4)])
    D.block_quant(c, raw, geo.val, c["oq"], c["os"], geo.val // 8 + 4, geo.val // 32)


@traced
def o_out(c):
    """Thread (t, j), owner CTAs: its 8 of the 32 columns summed over the K
    quarters in order -> bf16 -> ``out``."""
    s, tid, bid = c["S"], c["tid"], c["bid"]
    t = fx.min(tid // 4, s - 1)
    j = tid % 4
    y = [
        bf16_round(D.row_sum_waves(c["red"], 8 * (j % 2) + e, t, j // 2, o_splits(s)))
        for e in range(8)
    ]
    if (tid < s * 4) & (bid < D.O_CTAS):
        bo.buffer_store(
            fx.Vector.from_elements(D.pack_bf(y), fx.Int32),
            rsrc(c["args"]["out"]),
            (t * HIDDEN + 2 * ROWS * bid + 8 * j) // 2,
        )


# ---------------------------------------------------------------- kernel
_BUILDS: dict = {}


def build(key: K1Build):
    if key in _BUILDS:
        return _BUILDS[key]
    s = key.tokens
    assert key.tp in TP_SIZES and 1 <= s <= D.max_tokens(key.tp)
    assert key.normed or not key.oproj
    lay = scratch_layout(key)
    geo = Geo(key.tp)
    assert not key.oproj or (key.tp == 2 and geo.val // D.STEP % o_splits(s) == 0)
    nt = n_tasks(geo, s)

    @fx.struct
    class XLds:
        xq: fx.Array[fx.Int32, s * D.XQROW, 16]
        xs: fx.Array[fx.Int32, s * D.XSROW, 16]

    @fx.struct
    class GLds:
        qkl: fx.Array[fx.Float32, nt * 2 * HD, 16]
        vloc: fx.Array[fx.Float32, nt * VCH, 16]
        scal: fx.Array[fx.Float32, nt * 2, 16]

    @fx.struct
    class OLds:
        oq: fx.Array[fx.Int32, s * (geo.val // 8 + 4), 16]
        os: fx.Array[fx.Int32, s * (geo.val // 32), 16]

    @fx.union
    class StageLds:
        x: XLds
        g: GLds
        o: OLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        rst: fx.Array[fx.Float32, D.max_tokens(key.tp), 16]

    keyed = key_tuple(key, digest())
    name = (
        f"qwen38_27b_k1_s{s}_f{int(key.first)}_tp{key.tp}_n{int(key.normed)}"
        f"_o{int(key.oproj)}_t{int(key.stamps)}_p{key.stop}"
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
        w_bas: Int64,
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
        w_o: Int64,
        w_os: Int64,
        out: Int64,
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
        xl, gl, ol = st_lds.x.peek(), st_lds.g.peek(), st_lds.o.peek()
        mb = Mailbox(step_tag(epoch, layer) - 1)
        c = {
            "S": s,
            "G": geo,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "first": key.first,
            "oproj": key.oproj,
            "eps": key.eps,
            "norm_eps": key.norm_eps,
            "stamps": scratch + fx.Int64(lay["stamps"][0]) if key.stamps else None,
            "put": mb.put,
            "put_words": mb.put_words,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "rst": lds.rst.ptr,
            "xq": xl.xq.ptr,
            "xs": xl.xs.ptr,
            "qkl": gl.qkl.ptr,
            "vloc": gl.vloc.ptr,
            "scal": gl.scal.ptr,
            "oq": ol.oq.ptr,
            "os": ol.os.ptr,
            "args": {
                "hidden": hidden,
                "residual": residual,
                "res_out": res_out,
                "ln_w": ln_w,
                "w_qkvz": w_qkvz,
                "w_qkvzs": w_qkvzs,
                "w_ba": w_ba,
                "w_bas": w_bas,
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
                "w_o": w_o,
                "w_os": w_os,
                "out": out,
            },
        }
        for region in ("nsum", "xmb", "proj", "bap", "qk", "ov", "ordy"):
            c[region] = sreg(scratch, lay[region][0], region)
        D.stamp(c, 0)
        # the norm's inputs first, then the in_proj weights (vmcnt drains in order)
        if const_expr(key.normed):
            xraw = x_loads(c)
            qops = qkvz_loads(c)
            bops = ba_loads(c)
            x_quant(c, xraw)
        else:
            raw = front_loads(c)
            qops = qkvz_loads(c)
            bops = ba_loads(c)
            z, lnw = front_z(c, raw)
            D.o_rstd(c)
            D.o_x(c, z, lnw)
            D.o_gather(c)
        D.stamp(c, 1)
        if const_expr(key.stop > 2):
            stage_qkvz(c, qops)
            stage_ba(c, bops)
            D.stamp(c, 2)
        if const_expr(key.stop > 3):
            g = gdn_loads(c)
            gdn_conv(c, g)
            D.stamp(c, 3)
            gdn_gates(c)
            D.stamp(c, 4)
            gdn_recur(c, g)
            D.stamp(c, 5)
            gdn_norm(c, g)
            D.stamp(c, 6)
        if const_expr(key.stop > 4 and key.oproj):
            publish(c["put"], c["ordy"], bid, fx.Int32(1), tid == 0)
            oops = D.o_loads(c, width=geo.val, splits=o_splits(s))
            o_core(c)
            D.stamp(c, 7)
            D.o_stage(c, oops, width=geo.val, splits=o_splits(s))
            o_out(c)

    @flyc.jit
    def launch(
        hidden: Int64,
        residual: Int64,
        res_out: Int64,
        ln_w: Int64,
        w_qkvz: Int64,
        w_qkvzs: Int64,
        w_ba: Int64,
        w_bas: Int64,
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
        w_o: Int64,
        w_os: Int64,
        out: Int64,
        scratch: Int64,
        epoch: Int64,
        layer: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k1(
            hidden, residual, res_out, ln_w, w_qkvz, w_qkvzs, w_ba, w_bas, conv_w,
            conv_st, cs_seq, cs_dim, cs_tok, a_log, dt_bias, norm_w, rstate, rs_seq,
            st_idx, core, w_o, w_os, out, scratch, epoch, layer,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    ABI.check(k1, launch)
    _BUILDS[key] = launch
    return launch


def dense_gdn_pre(
    key: K1Build,
    *,
    hidden,
    residual=None,
    res_out=None,
    ln_w=None,
    w_qkvz,
    w_qkvzs,
    w_ba,
    w_bas,
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
    w_o=None,
    w_os=None,
    out=None,
):
    """Host launcher. ``hidden`` the full rows: the layer input, ``residual``
    None on the first layer (``key.first``); with ``key.normed`` the input norm's
    output, no ``residual`` / ``res_out`` / ``ln_w``. Every head-sliced operand
    this rank's share (``Geo(key.tp)``): ``w_qkvz`` / ``w_ba`` shuffled
    (``dense_ffn.shuffle_w``); ``conv_state`` [slots, conv, 3] (strides read);
    ``rstate`` [slots, nv, HD, HD] fp32, a slot dense; ``core`` [S, val].
    ``st_idx`` [>= S] int32, -1 a pad row; ``epoch`` a device int32 bumped once
    a step, ``layer`` this launch's tag slot. ``key.oproj``: out_proj's rank
    share ``w_o`` [HIDDEN, val / 2] shuffled, ``w_os``, and ``out`` [S, HIDDEN]
    bf16 (``core`` still written)."""
    s, geo = key.tokens, Geo(key.tp)
    qkvz, ba = geo.qkvz, geo.ba
    if key.normed:
        assert residual is None and res_out is None and ln_w is None
        assert not key.first
        res_out = ln_w = hidden
    else:
        assert (residual is None) == key.first
        assert res_out.shape == (s, HIDDEN) and ln_w.shape == (HIDDEN,)
    assert hidden.shape == (s, HIDDEN) and core.shape == (s, geo.val)
    assert w_qkvz.shape == (qkvz, HIDDEN // 2) and w_qkvzs.shape == (qkvz, HIDDEN // 32)
    assert w_ba.shape == (ba, HIDDEN // 2) and w_bas.shape == (ba, HIDDEN // 32)
    assert conv_w.numel() == geo.conv * CONV_W and conv_w.dtype == torch.bfloat16
    assert conv_state.dtype == torch.bfloat16
    assert conv_state.shape[1:] == (geo.conv, SL)
    assert rstate.dtype == torch.float32 and rstate.shape[1:] == (geo.nv, HD, HD)
    assert rstate[0].is_contiguous()
    assert a_log.dtype == torch.float32 and dt_bias.dtype == torch.bfloat16
    assert a_log.numel() == geo.nv and dt_bias.numel() == geo.nv
    assert norm_w.dtype == torch.bfloat16 and norm_w.numel() == HD
    assert st_idx.dtype == torch.int32 and st_idx.numel() >= s
    if key.oproj:
        assert w_o.shape == (HIDDEN, geo.val // 2)
        assert w_os.shape == (HIDDEN, geo.val // 32) and w_os.is_contiguous()
        assert out.shape == (s, HIDDEN) and out.dtype == torch.bfloat16
    else:
        w_o = w_os = out = core
    assert scratch.numel() >= scratch_bytes(key)
    for t in (hidden, res_out, w_qkvz, w_qkvzs, w_ba, w_bas, conv_w, core, w_o, out):
        assert t.is_contiguous()
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
                "w_qkvzs": w_qkvzs.data_ptr(),
                "w_ba": w_ba.data_ptr(),
                "w_bas": w_bas.data_ptr(),
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
                "w_o": w_o.data_ptr(),
                "w_os": w_os.data_ptr(),
                "out": out.data_ptr(),
                "scratch": scratch.data_ptr(),
                "epoch": epoch.data_ptr(),
                "layer": layer,
            }
        ),
        stream=torch.cuda.current_stream(),
    )
