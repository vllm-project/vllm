# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2 ``layer_post``: one Qwen3.8 decoder layer from the attention's core
output (the out_proj / o_proj input, the same shape on GDN and full-attention
layers) to the layer's MoE output, every TP reduction inside the kernel.

One launch a layer and rank, ``BLOCKS`` x ``THREADS``, S <= 8 decode rows:

    o_proj   (512 x 16 rows, K 2048 a rank; every CTA, tasks bid, bid + 256):
             bf16 GEMV of the core rows -> bf16 partial -> every rank's AR
    norm     (CTAs 0 .. 63, 128 columns each): the ranks' partials in rank
             order (fp32 -> bf16), + residual (bf16) -> ``res_out``; the AR
             sums (bf16) and the columns' sums of squares (MSQ), from which
             every consumer builds x: post_attention_layernorm (Gemma, 1 + w)
    router   (CTAs 64 .. 79: 4 x 16 of this rank's 64 experts x K quarters):
             fp32 parts -> bf16 logits -> every rank's RLOG
    sgu      (CTAs 80 .. 111: 32 x 16 rows of the shared gate | up) -> SGU
    sgate    (CTAs 112 .. 112+S-1, token t): sigmoid(x . w_sg) -> SGATE
    route    (every CTA, wave t token t): the 512 logits' top-10 (ties: the
             lower expert), in descending order, softmax-renormalized; x is
             built by every consumer from the AR sums and the norm tasks'
             sums of squares, so no x or route hand-off
    ug       (|U| x 16, every CTA): the union U of the step's picks; an
             expert's 16 gate + 16 up rows against the x rows' MXFP4 codes on
             the scaled fp4 MFMA; silu(g) u (fp32) -> INTER a pick
    down     (512 x 16 hidden rows, every CTA): w2 of every expert of U
             against its picks' INTER through MXFP4 (fp4 MFMA), times the route
             weight, summed in top-k order; + the shared down rows of silu(g) u
             (SGU) times SGATE -> bf16 partial -> every rank's FINAL
    final    (CTAs 0 .. 63, 128 columns): FINAL in rank order -> bf16 ``out``

The routed experts quantize their activations as the stock decode path does
(AITER's one-stage ``fmoe_bf16_pertokenMXfp4_g1u1_flat`` below 32 rows takes bf16
in, but runs x and silu(g) u through MXFP4, 1 x 32 blocks, RCEIL scales). Their
weights are ``AITER_MXFP4_MXFP4``'s shuffled ones (``layout``). Peer regions are
double-buffered by the step epoch's parity.

Two schedule choices follow the width: at one row the union U is the token's
picks in top-k order (no union pass); with the fused shared expert and at
least 4 rows, its gate | up tasks run before the route (they don't depend
on it).

Two variants follow the stock flags:

- ``fse`` (``VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS``): the shared expert is
  expert 512 of w13 / w2, a pick of every token at top-k index 10 with weight
  sigmoid(bf16(x . w_sg)) (fp32, the top-k kernel's fused sigmoid); no sgu
  stage, no shared down rows.
- ``oq`` (an online-MXFP4 out_proj on the ASM FP4 GEMM): o_proj's weight is
  fp4x2 through ``shuffle_weight(layout=(16, 16))``, its scales through
  ``e8m0_shuffle``; the core rows go through MXFP4 (RCEIL), as
  ``per_1x32_f4_quant_hip`` quantizes the GEMM's input.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.kimi_k3.amd.mono.common.ops import (
    CM_DEV,
    bf16_pair,
    bf16_round,
    bf_hi,
    bf_lo,
    hw_rsq,
    ld_i32,
    load_ptr64,
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
from vllm.models.kimi_k3.amd.mono.common.ranks import peer_bases, sum_partials
from vllm.models.kimi_k3.amd.mono.common.sync import Mailbox, preg, shift, sreg
from vllm.models.kimi_k3.amd.mono.stages import gemv
from vllm.models.qwen3_5.amd.mono import layout as L
from vllm.models.qwen3_5.amd.mono import moe
from vllm.models.qwen3_5.amd.mono.layout import need
from vllm.models.qwen3_5.amd.mono.moe import (
    MAX_U,
    PS_MIN,
    f32_of,
    sigmoid,
    silu_mul,
    step_tag,
)
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
RT0 = G0 + G_TASKS  # 112: shared gate (SGATE) CTAs, a token each
PS0 = G0  # the fused shared expert's early ug CTAs (the idle gate | up ones)
TAG_SLOTS = moe.TAG_SLOTS

assert RT0 + L.MAX_TOKENS <= BLOCKS and OTASKS == 2 * BLOCKS and L.BLOCKS == BLOCKS
assert (E, TOPK, L.MAX_TOKENS) == (moe.E, moe.TOPK, moe.MAX_TOKENS)

_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class K2Build:
    tokens: int
    eps: float = 1e-6  # post_attention_layernorm
    fse: bool = False  # the shared expert fused as expert 512
    oq: bool = False  # o_proj MXFP4 (ASM FP4 GEMM layout)
    # o_proj's input is core x sigmoid(gate) (bf16; the attention output gate
    # as Qwen3NextAttention applies it)
    og: bool = False

    @property
    def picks(self) -> int:
        """Expert picks a token: the top-k (+ the fused shared expert)."""
        return TOPK + int(self.fse)

    @property
    def u1(self) -> bool:
        """One row: U is the token's picks in top-k order (no union pass)."""
        return self.tokens == 1

    @property
    def ps(self) -> bool:
        """The fused shared expert's ug tasks run before the route."""
        return self.fse and self.tokens >= PS_MIN


ABI = KernelAbi(
    (
        "core",
        "residual",
        "w_o",
        "w_os",
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
        "gate",
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
        "rpart": R_TASKS * s * ROWS,
        "sgu": s * SI,  # 2 SI bf16 a token, two a pair
        "sgate": s,
        "route": s * 2 * TOPK,
        "inter": s * key.picks * RI,
    }
    lay = pair_layout(pairs.items())
    end = max(o + n for o, n in lay.values())
    lay["xrow"] = ((end + 15) // 16 * 16, s * HIDDEN * 2)
    return lay


def scratch_bytes(key: K2Build) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


# ---------------------------------------------------------------- norm
@traced
def stage_norm(c, j):
    """Columns 128 j ..: the AR sum (rank order, fp32 -> bf16) + residual (fp32;
    bf16 -> ``res_out``); the bf16 AR sums -> the x buffer, the columns' sums
    of squares -> MSQ. The consumers build x from them (``wait_x``)."""
    s, tid, lane = c["S"], c["tid"], c["lane"]
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
    # the bf16 AR sums, drained before this token's (this wave's) MSQ
    if mine:
        aw = bf16_pair(lo, hi).bitcast(fx.Int32)
        at = (t * HIDDEN + col) // 2
        bo.buffer_store(aw, rsrc(c["xbuf"]), at, cache_modifier=CM_DEV)
    sq = wave_sum(z0 * z0 + z1 * z1)
    rocdl.s_waitcnt(vmcnt=0)
    if mine & (lane == 0):
        c["put"](c["msq"], j * s + t, sq)


@traced
def load_rstd(c):
    """Every norm task's sums of squares -> each token's rstd -> LDS ``rs``
    (fp32 sum in task order). A wave waits on the last token's first: the
    full poll then doesn't spin."""
    s, tid, lane, red = c["S"], c["tid"], c["lane"], c["red"]
    if tid < 64:
        c["poll"]([(c["msq"], tid * s + s - 1, 1)])
    gpu.barrier()
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


@traced
def x_rows_to_lds(c, klen, k0):
    """The x columns k0 .. k0 + klen = bf16((AR sum + residual) rstd (1 + w)) ->
    LDS, 8 columns a lane. The norm reads the unrounded sum, as
    ``fused_add_rms_norm`` does."""
    s, tid, a = c["S"], c["tid"], c["args"]
    xrow = klen + gemv.LDS_PAD
    words = s * klen // 8
    for i in range_constexpr((words + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, words - 1)
        t = e // (klen // 8)
        kk = e % (klen // 8) * 8
        g = (t * HIDDEN + k0 + kk) // 2
        av = fx.Vector(
            bo.buffer_load(
                rsrc(c["xbuf"]), g, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV
            )
        )
        rv = fx.Vector(bo.buffer_load(rsrc(a["residual"]), g, vec_width=4, dtype=T.i32))
        wv = fx.Vector(
            bo.buffer_load(rsrc(a["ln_w"]), (k0 + kk) // 2, vec_width=4, dtype=T.i32)
        )
        r = fx.ptr_load(c["rs"] + t)
        ys = []
        for d in range_constexpr(4):
            z0 = bf_lo(av[d]) + bf_lo(rv[d])
            z1 = bf_hi(av[d]) + bf_hi(rv[d])
            w0 = fx.Float32(1.0) + bf_lo(wv[d])
            w1 = fx.Float32(1.0) + bf_hi(wv[d])
            ys.append(bf16_pair(z0 * r * w0, z1 * r * w1).bitcast(fx.Int32))
        if tid + THREADS * i < words:
            fx.ptr_store(
                fx.Vector.from_elements(ys, fx.Int32), c["xl"] + (t * xrow + kk) // 2
            )
    gpu.barrier()


@traced
def wait_x(c, klen, k0):
    """Every norm task's sums in -> x columns k0 .. k0 + klen into LDS."""
    load_rstd(c)
    x_rows_to_lds(c, klen, k0)


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
    s, lane, wave, red = c["S"], c["lane"], c["wave"], c["red"]
    wait_x(c, R_K, q * R_K)
    acc = gemv.mfmas(lane, wave, c["xl"], R_K + gemv.LDS_PAD, R_K, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()

    def out(t, r, vs):
        for p in range_constexpr(TP):
            c["put_bf"](
                c["peer"]["rlog"][p], (c["rank"] * s + t) * EPR + i * ROWS + r, vs
            )

    moe.ksplit_rows(
        c,
        c["rpart"],
        lambda qq, t, r: ((i * R_PARTS + qq) * s + t) * ROWS + r,
        q,
        R_PARTS,
        out,
    )


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
    (ReplicatedLinear then torch.sigmoid, both bf16 out); fused (``fse``), the
    sigmoid stays fp32: the top-k kernel's on the combined gate's bf16 logit."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    wait_x(c, HIDDEN, 0)
    per = HIDDEN // THREADS  # 16: two 16 B loads of x and of w a thread
    acc = fx.Float32(0.0)
    for h in range_constexpr(per // 8):
        k = tid * per + 8 * h
        xw = fx.Vector(
            fx.ptr_load(
                c["xl"] + (t * (HIDDEN + gemv.LDS_PAD) + k) // 2,
                result_type=fx.Vector.make_type(4, fx.Int32),
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
        g = sigmoid(bf16_round(tot))
        c["put"](c["sgate"], t, g if c["fse"] else bf16_round(g))
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


def _rlog_at(rlog, s, t, e):
    """Token t's logit of expert e: rank e / EPR's slot of every rank's RLOG."""
    return (rlog, ((e // EPR * s + t) * EPR + e % EPR) // 2, 1)


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

    kt = key.picks

    @fx.struct
    class RouteLds:
        route: fx.Array[fx.Int32, s * 2 * kt, 16]
        flag: fx.Array[fx.Int32, E, 16]
        expert: fx.Array[fx.Int32, MAX_U, 16]
        kof: fx.Array[fx.Int32, MAX_U * 8, 16]
        scan: fx.Array[fx.Int32, WAVES, 16]

    @fx.struct
    class XLds:
        xl: fx.Array[fx.Int32, s * (HIDDEN + gemv.LDS_PAD) // 2, 16]

    @fx.struct
    class DownLds:
        il: fx.Array[fx.Int32, s * kt * (RI + gemv.LDS_PAD) // 2, 16]
        hl: fx.Array[fx.Int32, 16 if key.fse else s * (SI + gemv.LDS_PAD) // 2, 16]
        contrib: fx.Array[fx.Float32, s * kt * ROWS, 16]

    @fx.union
    class StageLds:
        x: XLds
        down: DownLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        rs: fx.Array[fx.Float32, 16, 16]

    keyed = key_tuple(key, digest())
    name = f"qwen38_mono_k2_s{s}_f{int(key.fse)}_o{int(key.oq)}_a{int(key.og)}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k2(
        core: Int64,
        residual: Int64,
        w_o: Int64,
        w_os: Int64,
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
        gate: Int64,
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
        bases = peer_bases(peers, TP)
        own = load_ptr64(peers, rank)
        peer = {}
        for region in ("ar", "rlog", "final"):
            off = play[region][0]
            peer[region] = [
                shift(preg(b, off, region), fx.Int64(par) * half) for b in bases
            ]
            peer[region + "_own"] = shift(preg(own, off, region), fx.Int64(par) * half)
        rlog = peer["rlog_own"]
        c = {
            "S": s,
            "TP": TP,
            "moe": L.MOE,
            "rlog_at": lambda t, lane, i: _rlog_at(rlog, s, t, lane + 64 * i),
            "rlog_last": (
                TP,
                lambda i: (rlog, ((i * s + s - 1) * EPR + EPR - 2) // 2, 1),
            ),
            "route_cta": RT0,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "rank": rank,
            "eps": key.eps,
            "fse": key.fse,
            "oq": key.oq,
            "KT": kt,
            "put": mb.put,
            "put_bf": mb.put_bf,
            "put_words": mb.put_words,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "rs": lds.rs.ptr,
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
            "u1": key.u1,
            "og": key.og,
            "args": {
                "core": core,
                "gate": gate,
                "residual": residual,
                "w_o": w_o,
                "w_os": w_os,
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
        for region in (
            "msq",
            "rpart",
            "sgu",
            "sgate",
            "route",
            "inter",
        ):
            c[region] = sreg(scratch, lay[region][0], region)
        r0, r_tasks = R0, R_TASKS
        is_r = (bid >= r0) & (bid < r0 + r_tasks)
        is_g = (bid >= G0) & (bid < G0 + G_TASKS)
        nj = OTASKS // BLOCKS
        # ---- o_proj (both tasks' weights in flight first; a router CTA's
        # router weights behind them, they land before x does), then the
        # post-attention add + norm, router, shared gate | up
        if is_r:
            oops = [moe.o_loads(c, bid + BLOCKS * j, OK) for j in range(nj)]
            ri = bid - r0
            rops = router_loads(c, ri // R_PARTS, ri % R_PARTS)
            moe.run_oproj(c, oops, OK)
            stage_router(c, ri // R_PARTS, ri % R_PARTS, rops)
        if ~is_r:
            moe.run_oproj(
                c, [moe.o_loads(c, bid + BLOCKS * j, OK) for j in range(nj)], OK
            )
        if bid < NRED:
            stage_norm(c, bid)
        if const_expr(not key.fse):  # noqa: SIM102 (traced: `and` does not combine traced values)
            if is_g:
                gi = bid - G0
                gops = gemv.loads(c["lane"], c["wave"], w_sgu, gi * ROWS, HIDDEN)
                stage_sgu(c, gi, gops)
        if (bid >= RT0) & (bid < RT0 + s):
            stage_sgate(c, bid - RT0)
        # ---- the routed experts, then down: two row groups a CTA
        nu = moe.run_ug(c, key.ps, PS0, lambda: wait_x(c, HIDDEN, 0))
        dops = moe.down_loads(c, bid, nu, fx.min(c["wave"], nu - 1))
        if const_expr(not key.fse):
            load_h(c)
        moe.load_inter(c)
        dops2 = moe.stage_down(c, bid, nu, dops, bid + BLOCKS)
        moe.stage_down(c, bid + BLOCKS, nu, dops2)
        if bid < NRED:
            stage_final(c, bid)

    @flyc.jit
    def launch(
        core: Int64,
        residual: Int64,
        w_o: Int64,
        w_os: Int64,
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
        gate: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k2(
            core,
            residual,
            w_o,
            w_os,
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
            gate,
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
    w_os=None,
    w_sgu=None,
    w_sd=None,
    gate=None,
):
    """Host launcher; every weight as vLLM holds it after loading, the experts'
    shuffled (``layout``). ``w_os``: an fp4 o_proj's scales (``key.oq``);
    ``w_sgu`` / ``w_sd``: the unfused shared expert (not ``key.fse``); ``gate``:
    the attention output gate (``key.og``), core's shape.
    ``peers``: the int64 table of every rank's peer buffer (at least
    ``peer_bytes(S)`` each)."""
    s = key.tokens
    bf = torch.bfloat16
    fp4 = (torch.uint8, torch.float4_e2m1fn_x2)
    ne = E + int(key.fse)
    need(
        core.shape == (s, OK) and residual.shape == (s, HIDDEN),
        "core / residual shape",
    )
    need(out.shape == (s, HIDDEN) and res_out.shape == (s, HIDDEN), "output shape")
    need(
        (gate is not None) == key.og and (gate is None or gate.shape == core.shape),
        "gate: core's shape, with og only",
    )
    need(ln_w.shape == (HIDDEN,), "norm shape")
    if key.oq:
        need(
            w_o.shape == (HIDDEN, OK // 2) and w_o.dtype in fp4,
            "o_proj: fp4x2 [8192, 1024]",
        )
        need(
            w_os is not None
            and w_os.numel() == HIDDEN * OK // 32
            and w_os.element_size() == 1,
            "o_proj scales: e8m0, one a 32-block",
        )
    else:
        need(w_o.shape == (HIDDEN, OK) and w_o.dtype == bf, "o_proj: bf16 [8192, 2048]")
    need(w_gate.shape == (E, HIDDEN) and w_sg.numel() == HIDDEN, "router shape")
    if not key.fse:
        need(
            w_sgu is not None
            and w_sd is not None
            and w_sgu.shape == (2 * SI, HIDDEN)
            and w_sd.shape == (HIDDEN, SI),
            "shared expert shape",
        )
    need(
        w13.shape == (ne, 2 * RI, HIDDEN // 2) and w13.dtype in fp4,
        f"w13: fp4x2 [{ne}, 512, 4096]",
    )
    need(
        w2.shape == (ne, HIDDEN, RI // 2) and w2.dtype in fp4,
        f"w2: fp4x2 [{ne}, 8192, 128]",
    )
    need(
        w13s.numel() == ne * 2 * RI * HIDDEN // 32 and w13s.element_size() == 1,
        "w13 scales: e8m0, one a 32-block",
    )
    need(
        w2s.numel() == ne * HIDDEN * RI // 32 and w2s.element_size() == 1,
        "w2 scales: e8m0, one a 32-block",
    )
    used = [w_o, w_os, w_gate, w_sg, w_sgu, w_sd, w13, w13s, w2, w2s]
    used += [gate]
    tensors = [core, residual, out, res_out, ln_w] + [w for w in used if w is not None]
    need(all(t.is_contiguous() for t in tensors), "a non-contiguous input")
    dense = [core, residual, out, res_out, ln_w, w_gate, w_sg]
    dense += [gate] if gate is not None else []
    dense += [w for w in (w_sgu, w_sd) if w is not None and not key.fse]
    need(all(t.dtype == bf for t in dense), "activations and dense weights: bf16")

    def ptr(t):
        return 0 if t is None else t.data_ptr()

    f = build(key)
    f(
        *ABI.pack(
            {
                "core": core.data_ptr(),
                "residual": residual.data_ptr(),
                "w_o": w_o.data_ptr(),
                "w_os": ptr(w_os) if key.oq else 0,
                "ln_w": ln_w.data_ptr(),
                "w_gate": w_gate.data_ptr(),
                "w_sg": w_sg.data_ptr(),
                "w_sgu": 0 if key.fse else ptr(w_sgu),
                "w_sd": 0 if key.fse else ptr(w_sd),
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
                "gate": ptr(gate),
            }
        ),
        stream=torch.cuda.current_stream(),
    )
