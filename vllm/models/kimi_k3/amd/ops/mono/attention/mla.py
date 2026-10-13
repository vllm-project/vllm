# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M1a ``mla_pre``: one Kimi-K3 MLA layer from the AttnRes seam to AITER's
decode input, for a spec-verify step (S rows) at TP8 + DCP8 (the DCP group the
TP group: rank r's 12 heads are heads 12 r .. of the gathered query) or TP8 +
DCP1 (``ranks`` 1: the query is this rank's 12 heads, no gather).

One launch a layer, ``BLOCKS`` CTAs x ``THREADS`` threads:

    attn CTAs 0 .. 55   AttnRes + input_layernorm as K1 (``kda.stage_*``) ->
                        bf16 x rows, an XRDY flag a slice
    gemv CTAs 56 ..     fused_qkv_a (132 x 16 rows) and g_proj (96 x 16), task
                        gi and gi + 200: q_c | kv_c | k_pe -> QA (bf16 pairs);
                        the gate bf16(sigmoid(bf16 g)) -> ``gate_out``
    norm (token t on CTA t)
                        q_a / kv_a RMSNorm: q_c -> plain bf16 rows, a QCR flag
                        a token; kv_c (normed) | k_pe -> fp8 cache entry when
                        the token's slot is this rank's (>= 0)
    q_b (16 rows, CTAs 64 ..)
                        q_c rows in LDS: q_nope -> QN (bf16 pairs); q_pe -> fp8
                        -> every rank's QG (DCP1: ``q_out``)
    W_UK (a wave a (token tile, head, 16 latent rows), CTAs 208 ..)
                        q_nope -> MXFP4 (aiter's even scale, a lane a 32-block),
                        one 16x16x128 FP4 MFMA against W_K -> bf16 -> fp8 ->
                        every rank's QG (DCP1: ``q_out``)
    unpack (every CTA, DCP8)
                        this rank's QG -> the dense gathered ``q_out``
                        [S, ranks 12, 576] fp8

Rounding follows the vLLM path where it is elementwise (bf16 hand-offs, the
fp8 quants, the MXFP4 quant of the W_UK input); split-K sums differ only in
their order.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.kimi_k3.amd.ops.mono.attention import kda
from vllm.models.kimi_k3.amd.ops.mono.attention.kda import (
    HIDDEN,
    LDS_PAD,
    MAX_S,
    NA,
    XT,
    bf16r,
    ld_bf4,
    sigmoid,
    stage_gates,
    stage_mix,
    stage_slice,
    step_tag,
    vals_all,
    vals_words,
)
from vllm.models.kimi_k3.amd.ops.mono.common.debug import (
    region_ids,
    stamp,
    stamp_begin,
    stamp_flush,
)
from vllm.models.kimi_k3.amd.ops.mono.common.mx import (
    FP4,
    clamp_fp8,
    mfma_scaled,
    pow2,
)
from vllm.models.kimi_k3.amd.ops.mono.common.ops import (
    CM_DEV,
    bf16_pair,
    bf_hi,
    bf_lo,
    butterfly,
    fp8_pack4,
    hw_rsq,
    ld_i32,
    load_ptr64,
    row_sum,
    rsrc,
    traced,
    uniform,
)
from vllm.models.kimi_k3.amd.ops.mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    KernelAbi,
    first_task,
    key_tuple,
    pair_layout,
    source_digest,
)
from vllm.models.kimi_k3.amd.ops.mono.common.ranks import peer_bases
from vllm.models.kimi_k3.amd.ops.mono.common.sync import (
    Mailbox,
    preg,
    publish,
    sreg,
)
from vllm.models.kimi_k3.amd.ops.mono.stages import gemv

NH = 12  # local heads at TP8
QL = 1536  # q_lora_rank
KVL = 512  # kv_lora_rank
NOPE = 128
ROPE = 64
QHD = NOPE + ROPE  # 192
QKV_A = QL + KVL + ROPE  # 2112: fused_qkv_a's rows
GROWS = NH * 128  # 1536: g_proj's rows (heads x v_head_dim)
QB = NH * QHD  # 2304: q_b's rows
ENTRY = KVL + ROPE  # 576: a query row / cache entry (fp8 bytes)
QW = ENTRY // 4  # 144 words
ROWS = 16
A_TASKS = QKV_A // ROWS  # 132
GEMV_TASKS = A_TASKS + GROWS // ROWS  # 228
GEMV_CTAS = BLOCKS - NA  # 200
QB_SUB = QHD // ROWS  # 12 tasks a head: 8 nope, 4 pe
QB_TASKS = NH * QB_SUB  # 144
QB0 = 64  # q_b's first CTA (the norm tasks on CTAs 0 .. S - 1)
UK_GROUPS = KVL // ROWS  # 32 latent row groups a head
UK0 = QB0 + QB_TASKS  # W_UK's first CTA (208)
QCROW = QL + LDS_PAD  # an LDS q_c row
# timeline points: start, slice, gate, mix, gemv0, gemv1, norm, q_b, W_UK,
# unpack
TL_POINTS = 10

assert GEMV_TASKS <= 2 * GEMV_CTAS and QB0 >= MAX_S and UK0 <= BLOCKS
assert NH * QHD == QB and QL % (4 * 384) == 0 and KVL == 4 * 128

_STREAM = fx.Stream(None)
REGIONS = ("part", "wgt", "msq", "xrdy", "qa", "qcr", "qn", "qg")
_SOURCES = source_digest(".")


@dataclass(frozen=True)
class MlaPreBuild:
    tokens: int  # S
    nblocks: int  # AttnRes block sources (the prefix is one more)
    delta: bool
    eps: float = 1e-5  # AttnRes norm
    out_eps: float = 1e-5  # input_layernorm
    q_eps: float = 1e-6  # q_a_layernorm
    kv_eps: float = 1e-6  # kv_a_layernorm
    ranks: int = 8  # the DCP (= TP) group; 1: no DCP, no peer memory
    peer_off: int = 0  # QG's byte offset in a rank's peer buffer
    timeline: bool = False


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
        "w_qkv",
        "w_g",
        "q_nw",
        "kv_nw",
        "w_qb",
        "w_k",
        "wk_sh",
        "wk_sn",
        "w_ks",
        "ks_sh",
        "ks_sn",
        "ks_sk",
        "kv_cache",
        "kv_sb",
        "kv_ss",
        "kv_bs",
        "slots",
        "gate_out",
        "q_out",
        "scratch",
        "peers",
        "rank",
        "layer",
        "epoch",
        "health",
        "alarm",
        "tl",
    )
)


def scratch_layout(key: MlaPreBuild) -> dict:
    s, ns = key.tokens, key.nblocks + 1
    pairs = {
        "part": NA * s * ns * 2,
        "wgt": s * (ns + 1),
        "msq": NA * s,
        "xrdy": NA,
        "qa": s * QKV_A // 2,
        "qcr": s,
        "qn": s * NH * NOPE // 2,
    }
    lay = pair_layout(pairs.items())
    end = max(o + n for o, n in lay.values())
    # the plain bf16 rows (not pairs), 16 B aligned: x, then the normed q_c
    lay["x"] = ((end + 15) // 16 * 16, s * HIDDEN * 2)
    lay["qc"] = (lay["x"][0] + s * HIDDEN * 2, s * QL * 2)
    return lay


def scratch_bytes(key: MlaPreBuild) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def peer_pairs(s: int, ranks: int) -> int:
    """QG: every source rank's [S, 12, 144] fp8 words, a pair each."""
    return ranks * s * NH * QW


def peer_bytes(s: int, ranks: int) -> int:
    return peer_pairs(s, ranks) * 8


def qg_pair(c, src, t, h, w):
    return ((src * c["S"] + t) * NH + h) * QW + w


def put_qg(c, t, h, w, word):
    """fp8 word ``w`` of (token t, local head h) -> every rank's QG; at DCP1
    straight to ``q_out`` (nothing in this launch reads it)."""
    if const_expr(c["R"] == 1):
        bo.buffer_store(word, rsrc(c["args"]["q_out"]), (t * NH + h) * QW + w)
    if const_expr(c["R"] > 1):
        i = qg_pair(c, c["rank"], t, h, w)
        for p in range_constexpr(c["R"]):
            c["put_words"](c["qg"][p], i, [word])


def fp8_word(v):
    """Four f32 -> four E4M3 (vLLM's saturating quant, scale 1)."""
    return fp8_pack4(*[clamp_fp8(x) for x in v])


def bf_words(ptr, i, n):
    """``n`` dwords of a bf16 row at word ``i`` (device scope)."""
    return fx.Vector(
        bo.buffer_load(rsrc(ptr), i, vec_width=n, dtype=T.i32, cache_modifier=CM_DEV)
    )


# ---------------------------------------------------------------- gemv CTAs
def proj_loads(c, task):
    """Task ``task``'s 16 weight rows (fused_qkv_a, then g_proj) for this wave;
    a task past the last loads the last."""
    a = c["args"]
    task = fx.min(task, GEMV_TASKS - 1)
    is_g = uniform((task >= A_TASKS).select(fx.Int32(1), fx.Int32(0)))
    w = a["w_qkv"] + fx.Int64(is_g) * (a["w_g"] - a["w_qkv"])
    row0 = (task - is_g * A_TASKS) * ROWS
    return gemv.loads(c["lane"], c["wave"], w, row0, HIDDEN)


@traced
def stage_proj(c, task, ops, t0, n):
    """Task ``task`` against the LDS x (tokens t0 .. t0 + n): q_c | kv_c | k_pe
    rows -> QA, g rows -> the gate."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    acc = gemv.mfmas(lane, wave, c["xl"], HIDDEN + LDS_PAD, HIDDEN, n, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    r2 = tid % 8
    tt = tid // 8
    live = (tid < 8 * n) & (task < GEMV_TASKS)
    v0 = bf16r(row_sum(red, 2 * r2, fx.min(tt, n - 1)))
    v1 = bf16r(row_sum(red, 2 * r2 + 1, fx.min(tt, n - 1)))
    t = t0 + tt
    if live & (task < A_TASKS):
        c["put_bf"](c["qa"], t * QKV_A + task * ROWS + 2 * r2, [v0, v1])
    if live & (task >= A_TASKS):
        col = (task - A_TASKS) * ROWS + 2 * r2
        g = bf16_pair(sigmoid(v0), sigmoid(v1))
        bo.buffer_store(g, rsrc(c["args"]["gate_out"]), (t * GROWS + col) // 2)
    gpu.barrier()


@traced
def gemv_tasks(c, gi):
    """Gemv CTA ``gi``: task gi (weights issued before x lands), then gi + 200."""
    s, on, tls, tid = c["S"], c["on"], c["tls"], c["tid"]
    ops0 = proj_loads(c, gi)
    kda.wait_x(c)
    for t0, n in gemv.token_tiles(s, XT):
        kda.load_x(c, t0, n)
        stage_proj(c, gi, ops0, t0, n)
    stamp(on, tls, tid, 4)
    if gi + GEMV_CTAS < GEMV_TASKS:
        ops1 = proj_loads(c, gi + GEMV_CTAS)
        for t0, n in gemv.token_tiles(s, XT):
            if const_expr(s > XT):
                kda.load_x(c, t0, n)
            stage_proj(c, gi + GEMV_CTAS, ops1, t0, n)
    stamp(on, tls, tid, 5)


# ---------------------------------------------------------------- norm
@traced
def block_sum2(c, v):
    """Waves 0 .. 5's sum and waves 6, 7's sum of ``v`` (q / kv threads)."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    v = butterfly(v, (32, 16, 8, 4, 2, 1))
    gpu.barrier()
    if lane == 0:
        fx.ptr_store(v, red + wave)
    gpu.barrier()
    q = fx.ptr_load(red + 0)
    for w in range_constexpr(1, 6):
        q = q + fx.ptr_load(red + w)
    kv = fx.ptr_load(red + 6) + fx.ptr_load(red + 7)
    _ = tid
    return q, kv


@traced
def stage_norm(c, t):
    """Token t: q_c (threads 0 .. 383, 4 columns each) and kv_c (384 ..) RMSNorm;
    q_c -> plain bf16 row + QCR; kv_c | k_pe (threads 0 .. 15) -> fp8 entry."""
    tid, a = c["tid"], c["args"]
    is_q = tid < QL // 4
    col = is_q.select(tid * 4, QL + (tid - QL // 4) * 4)
    pe = QL + KVL + fx.min(tid, ROPE // 4 - 1) * 4
    got = c["poll"]([(c["qa"], (t * QKV_A + col) // 2, 2), (c["qa"], (t * QKV_A + pe) // 2, 2)])
    w = got[0]
    v = [bf_lo(w[0]), bf_hi(w[0]), bf_lo(w[1]), bf_hi(w[1])]
    sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
    qs, kvs = block_sum2(c, sq)
    rq = hw_rsq(qs * fx.Float32(1.0 / QL) + fx.Float32(c["q_eps"]))
    rkv = hw_rsq(kvs * fx.Float32(1.0 / KVL) + fx.Float32(c["kv_eps"]))
    nw = ld_bf4(a["q_nw"], fx.min(tid, QL // 4 - 1) * 4)
    kw = ld_bf4(a["kv_nw"], fx.max(tid - QL // 4, 0) * 4)
    qn = [bf16r(v[j] * rq * nw[j]) for j in range(4)]
    kn = [bf16r(v[j] * rkv * kw[j]) for j in range(4)]
    if is_q:
        bo.buffer_store(
            kda.bf4_words(qn),
            rsrc(c["qc"]),
            (t * QL + tid * 4) // 2,
            cache_modifier=CM_DEV,
        )
    slot = uniform(ld_i32(a["slots"], t * 2))
    if slot >= 0:
        blk = slot // a["kv_bs"]
        off = slot % a["kv_bs"]
        base = (
            a["kv_cache"]
            + fx.Int64(blk) * fx.Int64(a["kv_sb"])
            + fx.Int64(off) * fx.Int64(a["kv_ss"])
        )
        if tid >= QL // 4:
            bo.buffer_store(fp8_word(kn), rsrc(base), tid - QL // 4)
        if tid < ROPE // 4:
            p = got[1]
            pv = [bf_lo(p[0]), bf_hi(p[0]), bf_lo(p[1]), bf_hi(p[1])]
            bo.buffer_store(fp8_word(pv), rsrc(base), KVL // 4 + tid)
    publish(c["put"], c["qcr"], t, fx.Int32(1), tid == 0)
    gpu.barrier()


# ---------------------------------------------------------------- q_b
@traced
def stage_qb(c, task):
    """q_b rows 16 task ..: head h = task / 12, its nope rows (sub < 8) -> QN or
    pe rows -> fp8 -> QG."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    h = task // QB_SUB
    sub = task % QB_SUB
    ops = gemv.loads(lane, wave, a["w_qb"], task * ROWS, QL)
    if tid < s:
        c["poll"]([(c["qcr"], tid, 1)])
    gpu.barrier()
    for t0, n in gemv.token_tiles(s):
        gemv.dev_rows_to_lds(tid, c["qc"], QL, n, c["xl"], QCROW, t0=t0)
        acc = gemv.mfmas(lane, wave, c["xl"], QCROW, QL, n, ops)
        fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
        gpu.barrier()
        r2 = tid % 8
        tt = fx.min(tid // 8, n - 1)
        if (tid < 8 * n) & (sub < NOPE // ROWS):
            v0 = row_sum(red, 2 * r2, tt)
            v1 = row_sum(red, 2 * r2 + 1, tt)
            at = ((t0 + tt) * NH + h) * NOPE + sub * ROWS + 2 * r2
            c["put_bf"](c["qn"], at, [v0, v1])
        r4 = tid % 4
        t4 = fx.min(tid // 4, n - 1)
        if (tid < 4 * n) & (sub >= NOPE // ROWS):
            v = [bf16r(row_sum(red, 4 * r4 + j, t4)) for j in range(4)]
            w = KVL // 4 + (sub - NOPE // ROWS) * 4 + r4
            put_qg(c, t0 + t4, h, w, fp8_word(v))
        gpu.barrier()


# ---------------------------------------------------------------- W_UK
def even_code(am):
    """aiter's ``_mxfp4_scale_from_amax`` code from |bf16| bits ``am``: the
    exponent of amax rounded at mantissa bit 21, minus 2, clamped at 0."""
    e = (((am << 16) + fx.Int32(0x200000)) >> 23) & 0xFF
    return fx.max(e - 2, fx.Int32(0))


def pk_fp4_bf16(old, word, scale, j):
    from vllm.models.kimi_k3.amd.ops.mono.stages.moe import pk_fp4_bf16 as pk

    return pk(old, word, scale, j)


@traced
def stage_uk(c, task):
    """Wave task (tile, head, 16 latent rows): W_K's rows against the 16 tokens'
    MXFP4 q_nope -> bf16 -> fp8 -> QG (a lane 4 rows of one token)."""
    s, lane = c["S"], c["lane"]
    a = c["args"]
    ng = task % UK_GROUPS
    h = (task // UK_GROUPS) % NH
    t0 = task // (UK_GROUPS * NH) * 16
    n = ng * ROWS + lane % 16
    kb = lane // 16
    wo = h * a["wk_sh"] + n * a["wk_sn"] + kb * 16
    wv = fx.Vector(bo.buffer_load(rsrc(a["w_k"]), wo // 4, vec_width=4, dtype=T.i32))
    so = h * a["ks_sh"] + n * a["ks_sn"] + kb * a["ks_sk"]
    sw = fx.Int32(bo.buffer_load(rsrc(a["w_ks"]), so // 4, vec_width=1, dtype=T.i32))
    sa = (sw >> ((so % 4) * 8)) & 0xFF
    t = t0 + lane % 16
    tc = fx.min(t, s - 1)
    p0 = (((tc * NH + h) * NOPE) + kb * 32) // 2
    got = c["poll"]([(c["qn"], p0 + 2 * j, 2) for j in range(8)])
    ws = [got[j // 2][j % 2] for j in range(16)]
    am = fx.Int32(0)
    for w in ws:
        am = fx.max(am, w & 0x7FFF)
        am = fx.max(am, (w >> 16) & 0x7FFF)
    # code 0 (amax < 2^-125) quantizes by 2^-126: the tiny lane stays ~0
    code = fx.max(even_code(am), fx.Int32(1))
    scale = pow2(code)
    xs = []
    for q in range_constexpr(4):
        pk = fx.Int32(0)
        for j in range_constexpr(4):
            pk = pk_fp4_bf16(pk, ws[4 * q + j], scale, j)
        xs.append(pk)
    xv = fx.Vector.from_elements(xs, fx.Int32)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    acc = mfma_scaled(wv, xv, acc, sa, code, FP4, FP4, 0, 0)
    v = [bf16r(acc[j]) for j in range(4)]
    if t < s:
        put_qg(c, t, h, ng * 4 + kb, fp8_word(v))


@traced
def uk_tasks(c):
    s, bid, wave = c["S"], c["bid"], c["wave"]
    ntask = ((s + 15) // 16) * NH * UK_GROUPS
    first = first_task(bid, UK0) * WAVES + wave
    for task in range(first, ntask, BLOCKS * WAVES):
        stage_uk(c, task)


# ---------------------------------------------------------------- unpack
@traced
def stage_unpack(c):
    """This rank's QG -> ``q_out`` [S, ranks x 12, 576]: a thread 2 words a
    round (QG pair 2 u, u = global thread + round x grid)."""
    s, R, tid, bid = c["S"], c["R"], c["tid"], c["bid"]
    nu = R * s * NH * QW // 2
    grid = BLOCKS * THREADS
    rounds = (nu + grid - 1) // grid
    g = bid * THREADS + tid
    us = [fx.min(g + grid * i, nu - 1) for i in range(rounds)]
    got = c["poll"]([(c["qg_own"], 2 * u, 2) for u in us])
    for i in range_constexpr(rounds):
        u = us[i]
        w2 = u % (QW // 2)
        h = (u // (QW // 2)) % NH
        t = (u // (NH * QW // 2)) % s
        src = u // (s * NH * QW // 2)
        at = ((t * R + src) * NH + h) * QW + 2 * w2
        if g + grid * i < nu:
            bo.buffer_store(
                fx.Vector.from_elements(got[i], fx.Int32),
                rsrc(c["args"]["q_out"]),
                at,
            )


# ---------------------------------------------------------------- kernel
_BUILDS: dict = {}


def build(key: MlaPreBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    s, R = key.tokens, key.ranks
    ns = key.nblocks + 1
    lay = scratch_layout(key)
    xrow = HIDDEN + LDS_PAD
    assert 1 <= s <= MAX_S and key.nblocks >= 1 and 1 <= R <= 8

    @fx.struct
    class AttnLds:
        vals: fx.Array[fx.Float32, vals_words(s, ns), 16]

    @fx.struct
    class GemvLds:
        xl: fx.Array[fx.Int32, max(min(s, XT) * xrow, 16 * QCROW) // 2, 16]

    @fx.union
    class RoleLds:
        attn: AttnLds
        gemv: GemvLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    keyed = key_tuple(key, _SOURCES)
    name = (
        f"k3_mono_mla_pre_s{s}_nb{key.nblocks}_d{int(key.delta)}_r{R}"
        f"_t{int(key.timeline)}"
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def mla_pre(
        prefix: Int64,
        delta: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        in_nw: Int64,
        w_qkv: Int64,
        w_g: Int64,
        q_nw: Int64,
        kv_nw: Int64,
        w_qb: Int64,
        w_k: Int64,
        wk_sh: Int32,
        wk_sn: Int32,
        w_ks: Int64,
        ks_sh: Int32,
        ks_sn: Int32,
        ks_sk: Int32,
        kv_cache: Int64,
        kv_sb: Int64,
        kv_ss: Int32,
        kv_bs: Int32,
        slots: Int64,
        gate_out: Int64,
        q_out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        layer: Int32,
        epoch: Int64,
        health: Int64,
        alarm: Int64,
        tl: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        role = alloc.allocate(RoleLds)
        al, gl = role.attn.peek(), role.gemv.peek()
        mb = Mailbox(step_tag(epoch, layer) - 1, health, alarm, region_ids(REGIONS))
        # DCP1 reads no peer memory (``peers`` may be 0)
        bases = peer_bases(peers, R) if R > 1 else []
        own = load_ptr64(peers, rank) if R > 1 else peers
        c = {
            "S": s,
            "R": R,
            "NS": ns,
            "NB": key.nblocks,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "rank": rank,
            "delta": key.delta,
            "eps": key.eps,
            "out_eps": key.out_eps,
            "q_eps": key.q_eps,
            "kv_eps": key.kv_eps,
            "vals_all": vals_all(s, ns),
            "put": mb.put,
            "put_bf": mb.put_bf,
            "put_words": mb.put_words,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "vals": al.vals.ptr,
            "xl": gl.xl.ptr,
            "xbuf": scratch + fx.Int64(lay["x"][0]),
            "qc": scratch + fx.Int64(lay["qc"][0]),
            "qg": [preg(b, key.peer_off, "qg") for b in bases],
            "qg_own": preg(own, key.peer_off, "qg"),
            "args": {
                "prefix": prefix,
                "delta": delta,
                "blocks": blocks,
                "blk_sm": blk_sm,
                "blk_sr": blk_sr,
                "ares_nw": ares_nw,
                "ares_qk": ares_qk,
                "in_nw": in_nw,
                "w_qkv": w_qkv,
                "w_g": w_g,
                "q_nw": q_nw,
                "kv_nw": kv_nw,
                "w_qb": w_qb,
                "w_k": w_k,
                "wk_sh": wk_sh,
                "wk_sn": wk_sn,
                "w_ks": w_ks,
                "ks_sh": ks_sh,
                "ks_sn": ks_sn,
                "ks_sk": ks_sk,
                "kv_cache": kv_cache,
                "kv_sb": kv_sb,
                "kv_ss": kv_ss,
                "kv_bs": kv_bs,
                "slots": slots,
                "gate_out": gate_out,
                "q_out": q_out,
            },
        }
        for region in REGIONS:
            if const_expr(region != "qg"):
                c[region] = sreg(scratch, lay[region][0], region)
        on, tls = key.timeline, lds.tls.ptr
        c["on"], c["tls"] = on, tls
        stamp_begin(on, tls, tid, TL_POINTS)
        if bid < NA:
            stage_slice(c, bid)
            stamp(on, tls, tid, 1)
            stage_gates(c, bid)
            stamp(on, tls, tid, 2)
            stage_mix(c, bid)
            stamp(on, tls, tid, 3)
        if bid >= NA:
            gemv_tasks(c, bid - NA)
        for t in range(first_task(bid, 0), s, BLOCKS):
            stage_norm(c, t)
        stamp(on, tls, tid, 6)
        for task in range(first_task(bid, QB0), QB_TASKS, BLOCKS):
            stage_qb(c, task)
        stamp(on, tls, tid, 7)
        uk_tasks(c)
        stamp(on, tls, tid, 8)
        if const_expr(R > 1):
            stage_unpack(c)
        stamp(on, tls, tid, 9)
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
        w_qkv: Int64,
        w_g: Int64,
        q_nw: Int64,
        kv_nw: Int64,
        w_qb: Int64,
        w_k: Int64,
        wk_sh: Int32,
        wk_sn: Int32,
        w_ks: Int64,
        ks_sh: Int32,
        ks_sn: Int32,
        ks_sk: Int32,
        kv_cache: Int64,
        kv_sb: Int64,
        kv_ss: Int32,
        kv_bs: Int32,
        slots: Int64,
        gate_out: Int64,
        q_out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        layer: Int32,
        epoch: Int64,
        health: Int64,
        alarm: Int64,
        tl: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        mla_pre(
            prefix,
            delta,
            blocks,
            blk_sm,
            blk_sr,
            ares_nw,
            ares_qk,
            in_nw,
            w_qkv,
            w_g,
            q_nw,
            kv_nw,
            w_qb,
            w_k,
            wk_sh,
            wk_sn,
            w_ks,
            ks_sh,
            ks_sn,
            ks_sk,
            kv_cache,
            kv_sb,
            kv_ss,
            kv_bs,
            slots,
            gate_out,
            q_out,
            scratch,
            peers,
            rank,
            layer,
            epoch,
            health,
            alarm,
            tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(mla_pre, launch)
    _BUILDS[key] = launch
    return launch


def ptr(t):
    return 0 if t is None else t.data_ptr()


def mla_pre(
    key: MlaPreBuild,
    *,
    prefix,
    delta,
    blocks,
    ares_nw,
    ares_qk,
    in_nw,
    w_qkv,
    w_g,
    q_nw,
    kv_nw,
    w_qb,
    w_k,
    w_k_scale,
    kv_cache,
    slots,
    gate_out,
    q_out,
    scratch,
    peers,
    rank,
    layer,
    epoch,
    health,
    alarm,
    tl=None,
):
    """Host launcher. ``w_k`` (12, 512, 64) u8 MXFP4 and ``w_k_scale`` (12, 512,
    4) e8m0, any strides but W_K's last; ``kv_cache`` [blocks, block_size, 576]
    one-byte, entries dense; ``slots`` [S] int64 (-1: another rank's token);
    ``gate_out`` [S, 1536] bf16; ``q_out`` [S, ranks x 12, 576] fp8 (dense);
    ``peers`` every rank's peer buffer base (i64 a rank; unread at ranks 1,
    may be None); ``layer`` a slot below
    ``kda.TAG_SLOTS``."""
    assert w_k.stride(-1) == 1 and kv_cache.stride(-1) == 1
    assert slots.dtype == torch.int64 and slots.size(0) == key.tokens
    assert q_out.is_contiguous() and gate_out.is_contiguous()
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
        w_qkv.data_ptr(),
        w_g.data_ptr(),
        q_nw.data_ptr(),
        kv_nw.data_ptr(),
        w_qb.data_ptr(),
        w_k.data_ptr(),
        w_k.stride(0),
        w_k.stride(1),
        w_k_scale.data_ptr(),
        w_k_scale.stride(0),
        w_k_scale.stride(1),
        w_k_scale.stride(2),
        kv_cache.data_ptr(),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.size(1),
        slots.data_ptr(),
        gate_out.data_ptr(),
        q_out.data_ptr(),
        scratch.data_ptr(),
        ptr(peers),
        rank,
        layer,
        epoch.data_ptr(),
        health.data_ptr(),
        alarm,
        ptr(tl),
        stream=torch.cuda.current_stream(),
    )
