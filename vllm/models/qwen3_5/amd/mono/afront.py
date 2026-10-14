# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The attention front of a Qwen3.8 mono decode step (``attn_front``): the
residual add, input_layernorm and qkv_proj of one full-attention layer in one
launch, for S <= 8 rows. Every row has its own position and KV cache slot,
so the rows can be one token of each request (a decode step) or several
tokens of a request (an MTP verify step).

qkv_proj is MXFP6 E2M3 in the layout of moe.fp6_pack, and fp6_gemv
multiplies it with the bf16 x rows (W6A16). Its 4608 rows a rank (q with its
gate, k and v) are 288 tasks of 16 rows. 32 CTAs run two tasks and the
others one. Every CTA issues the loads of its tasks' weights first, so that
they are in flight while it builds x.

Every CTA builds the x rows itself, so no CTA waits for another. It sums the
squares of hidden + residual in fp32 and writes
x = bf16((hidden + residual) rstd (1 + w)) into LDS, as GemmaRMSNorm with a
residual computes it. The sums run in the same order on every CTA, so every
CTA gets the same x. CTA b stores its share of res_out = bf16(hidden +
residual), the 16-byte words w of res_out with w % 256 == b.

Without the build option qk, CTA b runs task b and CTAs 0 .. 31 also task
256 + b. The kernel writes the bf16 qkv rows, and the q and k norms, RoPE,
the gate copy and the KV cache write stay with the model's kernels
(fused_qk_rmsnorm_rope_gate and reshape_and_cache_flash).

With qk, the kernel also does the work of fused_qk_rmsnorm_rope_gate and
writes q, k, v and the gate as four contiguous bf16 tensors. The tasks go to
the CTAs by kind, so that the CTAs with two tasks run gate rows, which
nothing in this kernel waits for:

    CTAs   0 ..  31   the gate rows of heads 4 and 5, then of heads 6 and 7
    CTAs  32 .. 159   the q rows of heads 0 .. 7
    CTAs 160 .. 175   the k rows
    CTAs 176 .. 191   the v rows
    CTAs 192 .. 255   the gate rows of heads 0 .. 3

A q or k task puts its bf16 rows into the mailbox. After its own task, CTA
192 + h polls the 16 tasks of q head h (CTA 200 those of the k head), sums
the squares of each token's 256 values in fp32 and writes
bf16(x rstd (w + 1)), with NeoX RoPE on the first rot values, as the Triton
kernel computes them. Only the order of the fp32 sums differs, and whether
the compiler fuses the RoPE products into FMAs. These CTAs finish while the
CTAs with two tasks still run their second task.

With kv (which needs qk), the kernel also writes k and v into the FP8 KV
cache at the slots of slot_mapping, as reshape_and_cache_flash does:
e4m3fn(bf16 value / scale), rounded to nearest even and clamped to 448. A
slot below 0 (a CUDA graph pad row) is skipped. The caller must then call
the attention without k and v, so that vLLM does not write them again.
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
    div_rn,
    fp8_pack4,
    hw_rsq,
    ld_i32,
    row_sum,
    rsrc,
    traced,
    uniform,
    wave_sum,
    xshfl,
)
from vllm.models.kimi_k3.amd.mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    KernelAbi,
    key_tuple,
)
from vllm.models.kimi_k3.amd.mono.common.sync import Mailbox, sreg
from vllm.models.kimi_k3.amd.mono.stages import gemv
from vllm.models.qwen3_5.amd.mono import layout as L
from vllm.models.qwen3_5.amd.mono.layout import need
from vllm.models.qwen3_5.amd.mono.moe import fp6_gemv, fp6_gemv_loads, step_tag
from vllm.models.qwen3_5.amd.mono.sources import digest

HIDDEN, ROWS = L.HIDDEN, L.ROWS
HEAD = 256  # the head size of the full-attention layers
NQ = L.CORE // HEAD  # the q heads that each rank keeps
# qkv_proj's rows a rank: q and its gate for 8 heads, and k and v of the one
# KV head that each rank keeps.
QKV = 2 * L.CORE + 2 * HEAD
TASKS = QKV // ROWS
EXTRA = TASKS - BLOCKS
HT = HEAD // ROWS  # the tasks of one head's rows
PAIRS = HEAD // 2  # the bf16 pairs of one head and token
XROW = HIDDEN + gemv.LDS_PAD
# A row is HIDDEN / 8 words of 16 bytes, which the threads share evenly.
WORDS = HIDDEN // 8
PER = WORDS // THREADS
# With qk: the first CTA of each kind of first task (see the module doc). The
# owners of the q heads and of the k head are CTAs G0 .. G0 + NQ.
Q0 = EXTRA
K0 = Q0 + NQ * HT
V0 = K0 + HT
G0 = V0 + HT
KQ, KK, KV, KG = 0, 1, 2, 3
FP8_MAX = 448.0  # the largest finite e4m3fn value
assert QKV == 4608 and 0 < EXTRA <= BLOCKS
assert WORDS % THREADS == 0
assert EXTRA == 2 * HT and G0 + (NQ // 2) * HT == BLOCKS and G0 + NQ < BLOCKS

ABI = KernelAbi(
    (
        "hidden",
        "residual",
        "res_out",
        "ln_w",
        "w",
        "ws",
        "qkv",
        "q_out",
        "k_out",
        "v_out",
        "gate_out",
        "qn_w",
        "kn_w",
        "cos_sin",
        "positions",
        "scratch",
        "epoch",
        "slot",
        "k_cache",
        "v_cache",
        "page_stride",
        "token_stride",
        "slot_mapping",
        "k_scale",
        "v_scale",
    )
)

_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class AFBuild:
    tokens: int
    # The epsilon of input_layernorm.
    eps: float = 1e-6
    # The kernel also runs the q and k norms and RoPE, and writes q, k, v
    # and the gate as separate tensors.
    qk: bool = False
    # The kernel also writes k and v into the FP8 KV cache. This needs qk.
    kv: bool = False
    # The epsilon of q_norm and k_norm.
    qk_eps: float = 1e-6
    # The rotary_dim of the rotary embedding.
    rot: int = 64
    # cos_sin_cache is fp32 instead of bf16.
    cs_f32: bool = False
    # The tokens of one block of the KV cache views, key_cache.shape[1]. vLLM
    # splits this model's 2112-token blocks into blocks of 64 tokens for the
    # attention kernels, and slot_mapping counts in those blocks.
    page: int = 64


def pack_bf(v0, v1):
    return (
        fx.Vector.from_elements([v0, v1], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
    )


def load_words(c, t, i):
    """Thread word i of row t: hidden, residual and the norm weight, as
    vectors of 4 bf16 pairs, and the word's offset in pairs."""
    a = c["args"]
    kk = (c["tid"] + THREADS * i) * 8
    g = (t * HIDDEN + kk) // 2
    hv = fx.Vector(bo.buffer_load(rsrc(a["hidden"]), g, vec_width=4, dtype=T.i32))
    rv = fx.Vector(bo.buffer_load(rsrc(a["residual"]), g, vec_width=4, dtype=T.i32))
    return hv, rv, kk, g


@traced
def build_x(c):
    """Builds the bf16 x rows in LDS from hidden + residual, through the
    CTA's sums of squares and each row's rstd, and stores this CTA's share
    of res_out."""
    s, tid, lane, wave, bid = c["S"], c["tid"], c["lane"], c["wave"], c["bid"]
    a = c["args"]
    # Pass 1: each thread's sum of squares of each row, then the sum of the
    # wave, then the sum of the 8 waves in wave order.
    for t in range_constexpr(s):
        acc = fx.Float32(0.0)
        for i in range_constexpr(PER):
            hv, rv, _, _ = load_words(c, t, i)
            for d in range_constexpr(4):
                z0 = bf_lo(hv[d]) + bf_lo(rv[d])
                z1 = bf_hi(hv[d]) + bf_hi(rv[d])
                acc = acc + z0 * z0 + z1 * z1
        tot = wave_sum(acc)
        if lane == 0:
            fx.ptr_store(tot, c["sq"] + (wave * 8 + t))
    gpu.barrier()
    if tid < s:
        tot = fx.ptr_load(c["sq"] + tid)
        for w in range_constexpr(1, WAVES):
            tot = tot + fx.ptr_load(c["sq"] + (w * 8 + tid))
        r = hw_rsq(tot * fx.Float32(1.0 / HIDDEN) + fx.Float32(c["eps"]))
        fx.ptr_store(r, c["rs"] + tid)
    gpu.barrier()
    # Pass 2: the words again, now from the L2, -> x into LDS and res_out.
    # Word w of res_out is thread word (w - tid) / THREADS of thread
    # w % THREADS, and THREADS is a multiple of BLOCKS, so CTA b stores the
    # words of threads b and b + BLOCKS.
    mine = tid % BLOCKS == bid
    for t in range_constexpr(s):
        r = fx.ptr_load(c["rs"] + t)
        for i in range_constexpr(PER):
            hv, rv, kk, g = load_words(c, t, i)
            wv = fx.Vector(
                bo.buffer_load(rsrc(a["ln_w"]), kk // 2, vec_width=4, dtype=T.i32)
            )
            ys, zs = [], []
            for d in range_constexpr(4):
                z0 = bf_lo(hv[d]) + bf_lo(rv[d])
                z1 = bf_hi(hv[d]) + bf_hi(rv[d])
                w0 = fx.Float32(1.0) + bf_lo(wv[d])
                w1 = fx.Float32(1.0) + bf_hi(wv[d])
                ys.append(pack_bf(z0 * r * w0, z1 * r * w1))
                zs.append(pack_bf(z0, z1))
            fx.ptr_store(
                fx.Vector.from_elements(ys, fx.Int32), c["xl"] + (t * XROW + kk) // 2
            )
            if mine:
                bo.buffer_store(
                    fx.Vector.from_elements(zs, fx.Int32), rsrc(a["res_out"]), g
                )
    gpu.barrier()


@traced
def sums(c, ops):
    """The task's 16 rows against the x rows: each wave's fp32 MFMA tile into
    LDS red, for row_sum."""
    lane, wave, red = c["lane"], c["wave"], c["red"]
    acc = fp6_gemv(lane, wave, c["xl"], XROW, HIDDEN, c["S"], ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()


@traced
def proj(c, rg, ops):
    """Task rg: qkv_proj rows 16 rg .. 16 rg + 15 against the x rows -> bf16
    qkv, summed over the waves in wave order."""
    s, tid, red = c["S"], c["tid"], c["red"]
    sums(c, ops)
    if tid < ROWS * s:
        r = tid % ROWS
        t = tid // ROWS
        v = row_sum(red, r, t)
        bo.buffer_store(
            v.to(fx.BFloat16), rsrc(c["args"]["qkv"]), t * QKV + rg * ROWS + r
        )
    gpu.barrier()


def task_of(bid):
    """With qk: CTA bid's first task -> (kind, head, task of the head, row
    group of qkv_proj). The q rows of head h are row groups 32 h .. 32 h + 15
    and its gate rows 32 h + 16 .. 32 h + 31, then come k and v."""
    b = fx.Int32(bid)
    is_q = (b >= Q0) & (b < K0)
    is_k = (b >= K0) & (b < V0)
    is_v = (b >= V0) & (b < G0)
    qi = b - Q0
    # The CTAs before Q0 run the gate rows of heads 4 and 5 first, and the
    # ones from G0 those of heads 0 .. 3.
    gi = (b < Q0).select(b + 4 * HT, b - G0)
    kind = is_q.select(
        fx.Int32(KQ), is_k.select(fx.Int32(KK), is_v.select(fx.Int32(KV), fx.Int32(KG)))
    )
    hh = is_q.select(
        qi // HT, is_k.select(fx.Int32(NQ), is_v.select(fx.Int32(NQ + 1), gi // HT))
    )
    ti = is_q.select(qi % HT, is_k.select(b - K0, is_v.select(b - V0, gi % HT)))
    rg = is_q.select(
        2 * HT * (qi // HT) + qi % HT,
        is_k.select(
            2 * NQ * HT + (b - K0),
            is_v.select(
                2 * NQ * HT + HT + (b - V0), 2 * HT * (gi // HT) + HT + gi % HT
            ),
        ),
    )
    return kind, hh, ti, rg


def kv_row(c, cache, t):
    """Token t's row in a KV cache side (block slot // page, row slot % page)
    and its slot. The slot is the low word of the int64 entry, and a
    negative slot gives an address that the caller must not use."""
    a = c["args"]
    slot = uniform(ld_i32(a["slot_mapping"], 2 * t))
    base = (
        cache
        + fx.Int64(slot // c["page"]) * a["page_stride"]
        + fx.Int64(slot % c["page"]) * a["token_stride"]
    )
    return base, slot


def fp8_in(x, d, r):
    """Returns x / scale, divided as reshape_and_cache_flash divides it, then
    clamped to the finite e4m3fn range as its saturating conversion does."""
    q = div_rn(x, d, r)
    return fx.max(fx.min(q, fx.Float32(FP8_MAX)), fx.Float32(-FP8_MAX))


@traced
def kv_v(c, ti):
    """With kv, v task ti's 16 dims of each token into the FP8 v cache. Wave
    t (t < S) writes token t, so the row address is wave-uniform, and its
    lanes 0 .. 3 write four dims each. The value is the bf16 v that the
    model's path would hand to reshape_and_cache_flash."""
    s, lane = c["S"], c["lane"]
    a = c["args"]
    t = uniform(c["wave"])
    tc = fx.min(t, s - 1)
    base, slot = kv_row(c, a["v_cache"], tc)
    if (t < s) & (slot >= 0) & (lane < ROWS // 4):
        d = ld_i32(a["v_scale"], 0).bitcast(fx.Float32)
        r = fx.Float32(1.0) / d
        vals = [
            fp8_in(bf16_round(row_sum(c["red"], 4 * lane + j, tc)), d, r)
            for j in range_constexpr(4)
        ]
        bo.buffer_store(fp8_pack4(*vals), rsrc(base), ti * (ROWS // 4) + lane)


@traced
def proj_qk(c, ops, kind, hh, ti):
    """A task of the qk build: task ti of head hh's 16 tasks of kind
    ``kind``. q and k rows go to the mailbox for the head's owner as bf16
    pairs, v rows to v_out (and with kv to the FP8 v cache), gate rows to
    gate_out. The row sums are those of proj, so the bf16 values equal the
    qkv rows that the build without qk writes."""
    s, tid, red = c["S"], c["tid"], c["red"]
    a = c["args"]
    sums(c, ops)
    if (kind == KQ) | (kind == KK):
        if tid < (ROWS // 2) * s:
            p2 = tid % (ROWS // 2)
            t = tid // (ROWS // 2)
            v0 = row_sum(red, 2 * p2, t)
            v1 = row_sum(red, 2 * p2 + 1, t)
            at = (hh * s + t) * PAIRS + ti * (ROWS // 2) + p2
            c["put_bf"](c["qk"], 2 * at, [v0, v1])
    else:
        if tid < ROWS * s:
            r = tid % ROWS
            t = tid // ROWS
            v = row_sum(red, r, t)
            is_v = kind == KV
            at = is_v.select(
                t * HEAD + ti * ROWS + r, t * L.CORE + hh * HEAD + ti * ROWS + r
            )
            bo.buffer_store(
                v.to(fx.BFloat16), rsrc(is_v.select(a["v_out"], a["gate_out"])), at
            )
        if const_expr(c["kv"]):  # noqa: SIM102 (traced: `and` does not combine traced values)
            if kind == KV:
                kv_v(c, ti)
    gpu.barrier()


def cos_sin(c, t, q2):
    """Returns the cos and sin of rotary dims 2 q2 and 2 q2 + 1 at token
    t's position, from cos_sin_cache [positions, rot] with cos in the first
    half of a row and sin in the second."""
    a = c["args"]
    rot = c["rot"]
    pos = ld_i32(a["positions"], 2 * t)
    if const_expr(c["cs_f32"]):
        cv = fx.Vector(
            bo.buffer_load(
                rsrc(a["cos_sin"]), pos * rot + 2 * q2, vec_width=2, dtype=T.i32
            )
        )
        sv = fx.Vector(
            bo.buffer_load(
                rsrc(a["cos_sin"]),
                pos * rot + rot // 2 + 2 * q2,
                vec_width=2,
                dtype=T.i32,
            )
        )
        return (
            cv[0].bitcast(fx.Float32),
            cv[1].bitcast(fx.Float32),
            sv[0].bitcast(fx.Float32),
            sv[1].bitcast(fx.Float32),
        )
    cw = ld_i32(a["cos_sin"], pos * (rot // 2) + q2)
    sw = ld_i32(a["cos_sin"], pos * (rot // 2) + rot // 4 + q2)
    return bf_lo(cw), bf_hi(cw), bf_lo(sw), bf_hi(sw)


@traced
def kv_k(c):
    """With kv, the k owner's bf16 k rows (LDS ol) into the FP8 k cache. Wave
    t (t < S) writes token t, every lane four dims."""
    s, lane = c["S"], c["lane"]
    a = c["args"]
    t = uniform(c["wave"])
    tc = fx.min(t, s - 1)
    base, slot = kv_row(c, a["k_cache"], tc)
    if (t < s) & (slot >= 0):
        d = ld_i32(a["k_scale"], 0).bitcast(fx.Float32)
        r = fx.Float32(1.0) / d
        w0 = fx.ptr_load(c["ol"] + (tc * PAIRS + 2 * lane))
        w1 = fx.ptr_load(c["ol"] + (tc * PAIRS + 2 * lane + 1))
        vals = [fp8_in(x, d, r) for x in (bf_lo(w0), bf_hi(w0), bf_lo(w1), bf_hi(w1))]
        bo.buffer_store(fp8_pack4(*vals), rsrc(base), lane)


@traced
def post(c, hh):
    """The owner of head hh (q heads 0 .. NQ - 1, the k head NQ): the head's
    256 bf16 values of each token from its 16 tasks' mailbox pairs ->
    bf16(x rstd (w + 1)) with rstd from the fp32 sum of squares -> NeoX RoPE
    on the first rot values -> bf16 q or k. Thread tid holds pair tid % 128
    of token tid / 128 (+ 4 a round), so a token's pairs are two waves."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    a = c["args"]
    is_k = hh == NQ
    out = is_k.select(a["k_out"], a["q_out"])
    row = is_k.select(fx.Int32(HEAD // 2), fx.Int32(L.CORE // 2))
    col = is_k.select(fx.Int32(0), hh * (HEAD // 2))
    rp = c["rot"] // 4
    p = tid % PAIRS
    nw = ld_i32(is_k.select(a["kn_w"], a["qn_w"]), p)
    w0 = bf_lo(nw) + fx.Float32(1.0)
    w1 = bf_hi(nw) + fx.Float32(1.0)
    q2 = p % rp
    first = p < rp
    in_rot = p < 2 * rp
    for rd in range_constexpr((s * PAIRS + THREADS - 1) // THREADS):
        t_raw = tid // PAIRS + rd * (THREADS // PAIRS)
        t = fx.min(t_raw, s - 1)
        got = c["poll"]([(c["qk"], (hh * s + t) * PAIRS + p, 1)], batch=1)
        x0 = bf_lo(got[0][0])
        x1 = bf_hi(got[0][0])
        ss = wave_sum(x0 * x0 + x1 * x1)
        if lane == 0:
            fx.ptr_store(ss, c["sq"] + wave)
        gpu.barrier()
        w2 = (wave // 2) * 2
        tot = fx.ptr_load(c["sq"] + w2) + fx.ptr_load(c["sq"] + (w2 + 1))
        inv = hw_rsq(tot * fx.Float32(1.0 / HEAD) + fx.Float32(c["qk_eps"]))
        y0 = bf16_round(x0 * inv * w0)
        y1 = bf16_round(x1 * inv * w1)
        # Pair p < rp holds dims 2 p and 2 p + 1 of the first rotary half,
        # and pair p + rp the same dims of the second half, lane p ^ rp of
        # the same wave. The first half gets y cos - z sin, the second
        # y cos + z sin, with z the partner's value.
        zw = xshfl(bf16_pair(y0, y1).bitcast(fx.Int32), rp)
        cos0, cos1, sin0, sin1 = cos_sin(c, t, q2)
        sz0 = bf_lo(zw) * sin0
        sz1 = bf_hi(zw) * sin1
        r0 = y0 * cos0 + first.select(-sz0, sz0)
        r1 = y1 * cos1 + first.select(-sz1, sz1)
        ow = bf16_pair(in_rot.select(r0, y0), in_rot.select(r1, y1)).bitcast(fx.Int32)
        if t_raw < s:
            bo.buffer_store(ow, rsrc(out), t * row + col + p)
            if const_expr(c["kv"]):
                fx.ptr_store(ow, c["ol"] + (t * PAIRS + p))
        # The next round stores into sq, which this round has just read.
        gpu.barrier()
    if const_expr(c["kv"]):  # noqa: SIM102 (traced: `and` does not combine traced values)
        if is_k:
            kv_k(c)


_BUILDS: dict = {}


def build(key: AFBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    s = key.tokens
    need(1 <= s <= L.MAX_TOKENS, f"{s} rows, the kernels take 1..{L.MAX_TOKENS}")
    need(key.qk or not key.kv, "kv takes qk")
    rp = key.rot // 4
    need(
        key.rot % 4 == 0 and rp in (1, 2, 4, 8, 16, 32),
        f"rot {key.rot}: a rotary half must be 2, 4, .. 64 pairs of one wave",
    )

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, THREADS * 4, 16]
        sq: fx.Array[fx.Float32, WAVES * 8, 16]
        rs: fx.Array[fx.Float32, 16, 16]
        xl: fx.Array[fx.Int32, s * XROW // 2, 16]
        ol: fx.Array[fx.Int32, s * PAIRS, 16]

    keyed = key_tuple(key, digest() + "af")
    name = f"qwen38_mono_af_s{s}" + (
        f"_q{int(key.kv)}r{key.rot}c{int(key.cs_f32)}p{key.page}" if key.qk else ""
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def af(
        hidden: Int64,
        residual: Int64,
        res_out: Int64,
        ln_w: Int64,
        w: Int64,
        ws: Int64,
        qkv: Int64,
        q_out: Int64,
        k_out: Int64,
        v_out: Int64,
        gate_out: Int64,
        qn_w: Int64,
        kn_w: Int64,
        cos_sin: Int64,
        positions: Int64,
        scratch: Int64,
        epoch: Int64,
        slot: Int32,
        k_cache: Int64,
        v_cache: Int64,
        page_stride: Int64,
        token_stride: Int64,
        slot_mapping: Int64,
        k_scale: Int64,
        v_scale: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(Smem).peek()
        lane, wave = tid % 64, tid // 64
        c = {
            "S": s,
            "tid": tid,
            "bid": bid,
            "lane": lane,
            "wave": wave,
            "eps": key.eps,
            "qk_eps": key.qk_eps,
            "rot": key.rot,
            "cs_f32": key.cs_f32,
            "kv": key.kv,
            "page": key.page,
            "red": lds.red.ptr,
            "sq": lds.sq.ptr,
            "rs": lds.rs.ptr,
            "xl": lds.xl.ptr,
            "ol": lds.ol.ptr,
            "args": {
                "hidden": hidden,
                "residual": residual,
                "res_out": res_out,
                "ln_w": ln_w,
                "qkv": qkv,
                "q_out": q_out,
                "k_out": k_out,
                "v_out": v_out,
                "gate_out": gate_out,
                "qn_w": qn_w,
                "kn_w": kn_w,
                "cos_sin": cos_sin,
                "positions": positions,
                "k_cache": k_cache,
                "v_cache": v_cache,
                "page_stride": page_stride,
                "token_stride": token_stride,
                "slot_mapping": slot_mapping,
                "k_scale": k_scale,
                "v_scale": v_scale,
            },
        }
        if const_expr(key.qk):
            mb = Mailbox(step_tag(epoch, slot) - 1)
            c["qk"] = sreg(scratch, 0, "qk")
            c["put_bf"] = mb.put_bf
            c["poll"] = mb.poll
            kind, hh, ti, rg = task_of(bid)
            ops = fp6_gemv_loads(lane, wave, w, ws, rg, HIDDEN)
            if bid < EXTRA:
                # The second task: the gate rows of heads 6 and 7.
                g2 = bid + (NQ - 2) * HT
                ops2 = fp6_gemv_loads(
                    lane, wave, w, ws, 2 * HT * (g2 // HT) + HT + g2 % HT, HIDDEN
                )
                build_x(c)
                proj_qk(c, ops, kind, hh, ti)
                proj_qk(c, ops2, fx.Int32(KG), fx.Int32(g2 // HT), fx.Int32(g2 % HT))
            else:
                build_x(c)
                proj_qk(c, ops, kind, hh, ti)
                if (bid >= G0) & (bid <= G0 + NQ):
                    post(c, fx.Int32(bid - G0))
        else:
            ops = fp6_gemv_loads(lane, wave, w, ws, bid, HIDDEN)
            if bid < EXTRA:
                ops2 = fp6_gemv_loads(lane, wave, w, ws, bid + BLOCKS, HIDDEN)
                build_x(c)
                proj(c, bid, ops)
                proj(c, bid + BLOCKS, ops2)
            else:
                build_x(c)
                proj(c, bid, ops)

    @flyc.jit
    def launch(
        hidden: Int64,
        residual: Int64,
        res_out: Int64,
        ln_w: Int64,
        w: Int64,
        ws: Int64,
        qkv: Int64,
        q_out: Int64,
        k_out: Int64,
        v_out: Int64,
        gate_out: Int64,
        qn_w: Int64,
        kn_w: Int64,
        cos_sin: Int64,
        positions: Int64,
        scratch: Int64,
        epoch: Int64,
        slot: Int32,
        k_cache: Int64,
        v_cache: Int64,
        page_stride: Int64,
        token_stride: Int64,
        slot_mapping: Int64,
        k_scale: Int64,
        v_scale: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        af(
            hidden,
            residual,
            res_out,
            ln_w,
            w,
            ws,
            qkv,
            q_out,
            k_out,
            v_out,
            gate_out,
            qn_w,
            kn_w,
            cos_sin,
            positions,
            scratch,
            epoch,
            slot,
            k_cache,
            v_cache,
            page_stride,
            token_stride,
            slot_mapping,
            k_scale,
            v_scale,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(af, launch)
    _BUILDS[key] = launch
    return launch


# The mailbox pairs of the qk build: one (bf16 pair, tag) of 8 bytes for
# every pair of the 8 q heads and the k head of every token.
SCRATCH_BYTES = (NQ + 1) * L.MAX_TOKENS * PAIRS * 8


def _ptr(t):
    return 0 if t is None else t.data_ptr()


def attn_front(
    key: AFBuild,
    *,
    hidden,
    residual,
    res_out,
    ln_w,
    w,
    ws,
    qkv=None,
    q_out=None,
    k_out=None,
    v_out=None,
    gate_out=None,
    qn_w=None,
    kn_w=None,
    cos_sin=None,
    positions=None,
    scratch=None,
    epoch=None,
    slot=0,
    k_cache=None,
    v_cache=None,
    slot_mapping=None,
    k_scale=None,
    v_scale=None,
):
    """Host launcher. ``w`` and ``ws`` are qkv_proj's MXFP6 weight and scale
    bytes from fp6_pack, and ``res_out`` receives hidden + residual.

    Without key.qk, ``qkv`` [S, 4608] bf16 receives q with its gate, k and v
    as qkv_proj would.

    With key.qk, ``q_out`` [S, 2048], ``k_out`` [S, 256], ``v_out`` [S, 256]
    and ``gate_out`` [S, 2048] (bf16, contiguous) receive what
    fused_qk_rmsnorm_rope_gate returns and the v of qkv_proj. ``qn_w`` and
    ``kn_w`` are q_norm's and k_norm's weights (the kernel adds 1),
    ``cos_sin`` the rotary embedding's cos_sin_cache, ``positions`` the S
    int64 positions, ``scratch`` at least SCRATCH_BYTES of mailbox, and
    ``epoch`` and ``slot`` the runner's step epoch and this launch's slot.

    With key.kv, ``k_cache`` and ``v_cache`` are the backend's two views
    [blocks, key.page, 1, 256] of the FP8 KV cache (same strides), and
    ``slot_mapping`` holds exactly the S int64 slots. ``k_scale`` and
    ``v_scale`` are the layer's fp32 scales."""
    s = key.tokens
    bf = torch.bfloat16
    rows = (hidden, residual, res_out)
    need(all(t.shape == (s, HIDDEN) for t in rows), "hidden / residual / res_out shape")
    need(ln_w.shape == (HIDDEN,), "ln_w shape")
    need(
        w.dtype == torch.uint8 and w.numel() == QKV * HIDDEN * 3 // 4,
        "qkv_proj: fp6_pack's MXFP6 weight, 4608 x 8192 x 3/4 bytes",
    )
    need(
        ws.dtype == torch.uint8 and ws.numel() == QKV * HIDDEN // 32,
        "qkv_proj scales: e8m0, one a 32-block",
    )
    need(all(t.dtype == bf for t in (*rows, ln_w)), "activations and ln_w must be bf16")
    need(all(t.is_contiguous() for t in (*rows, ln_w, w, ws)), "a non-contiguous input")
    page_stride = token_stride = 0
    if key.qk:
        outs = (q_out, k_out, v_out, gate_out)
        need(
            q_out.shape == (s, L.CORE)
            and gate_out.shape == (s, L.CORE)
            and k_out.shape == (s, HEAD)
            and v_out.shape == (s, HEAD),
            "q_out / k_out / v_out / gate_out shape",
        )
        need(
            all(t.dtype == bf and t.is_contiguous() for t in outs),
            "outputs: contiguous bf16",
        )
        need(
            qn_w.shape == (HEAD,)
            and kn_w.shape == (HEAD,)
            and qn_w.dtype == bf
            and kn_w.dtype == bf,
            "q_norm / k_norm weights: 256 bf16",
        )
        cs_dtype = torch.float32 if key.cs_f32 else bf
        need(
            cos_sin.dtype == cs_dtype
            and cos_sin.dim() == 2
            and cos_sin.shape[1] == key.rot
            and cos_sin.is_contiguous(),
            f"cos_sin_cache: contiguous [positions, {key.rot}] {cs_dtype}",
        )
        need(
            positions.dtype == torch.int64
            and positions.numel() >= s
            and positions.is_contiguous(),
            "positions: int64, one a row",
        )
        need(
            scratch.numel() * scratch.element_size() >= SCRATCH_BYTES,
            "scratch too small",
        )
        need(epoch.dtype == torch.int32, "epoch: the runner's int32 step epoch")
    else:
        need(qkv.shape == (s, QKV) and qkv.dtype == bf and qkv.is_contiguous(), "qkv")
    if key.kv:
        need(
            k_cache.shape[1] == key.page
            and k_cache.shape[2:] == (1, HEAD)
            and v_cache.shape == k_cache.shape
            and v_cache.stride() == k_cache.stride()
            and k_cache.element_size() == 1
            and k_cache.stride(3) == 1,
            f"KV cache views: [blocks, {key.page}, 1, {HEAD}] of one byte, same "
            "strides",
        )
        # vLLM can pass a slot_mapping without the pad rows of the batch. The
        # kernel would then read slots past its end, so it takes only one of
        # exactly S rows.
        need(
            slot_mapping.dtype == torch.int64 and slot_mapping.shape == (s,),
            "slot_mapping: int64, exactly one a row",
        )
        need(
            k_scale.dtype == torch.float32 and v_scale.dtype == torch.float32,
            "k_scale / v_scale: fp32",
        )
        page_stride, token_stride = k_cache.stride(0), k_cache.stride(1)
    f = build(key)
    f(
        *ABI.pack(
            {
                "hidden": hidden.data_ptr(),
                "residual": residual.data_ptr(),
                "res_out": res_out.data_ptr(),
                "ln_w": ln_w.data_ptr(),
                "w": w.data_ptr(),
                "ws": ws.data_ptr(),
                "qkv": _ptr(qkv),
                "q_out": _ptr(q_out),
                "k_out": _ptr(k_out),
                "v_out": _ptr(v_out),
                "gate_out": _ptr(gate_out),
                "qn_w": _ptr(qn_w),
                "kn_w": _ptr(kn_w),
                "cos_sin": _ptr(cos_sin),
                "positions": _ptr(positions),
                "scratch": _ptr(scratch),
                "epoch": _ptr(epoch),
                "slot": int(slot),
                "k_cache": _ptr(k_cache),
                "v_cache": _ptr(v_cache),
                "page_stride": page_stride,
                "token_stride": token_stride,
                "slot_mapping": _ptr(slot_mapping),
                "k_scale": _ptr(k_scale),
                "v_scale": _ptr(v_scale),
            }
        ),
        stream=torch.cuda.current_stream(),
    )
