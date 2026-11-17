# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`dsa_front`: a Kimi-K3.1 DSA layer's decode prologue at TP8 + DCP1, from the
block residuals to the absorbed MLA query and the indexer's exact top-k, in
one launch (`VLLM_KIMI_SD_DSA=auto`; K2 then attends, `dsa_attn`).

`BLOCKS` CTAs x `THREADS` threads; the stages hand rows to each other through
scratch mailboxes (`publish` / poll), so a stage starts as soon as its own
inputs land, not at a grid-wide barrier:

    attn-res (CTAs 0 .. NA - 1)
                        the block-residual slice, gates and mix (KDA's stages)
                        -> the layer input x, RMS-normed
    projections (CTAs NA ..; a task 16 rows)
                        x against [q_a | kv_a | k_pe] (2112), the output gate
                        (1536), indexer wk (128) and wv (32) -> scratch
    norm (a CTA a token)
                        q_a RMSNorm -> q_c; kv_a RMSNorm -> E4M3 + a power-of-
                        two scale a 128 latents, and k_pe, into the fp8_ds_mla
                        slot; the indexer key LayerNorm -> MXFP4 into its page
    q_b / indexer q (a task 16 rows of q_c)
                        q_b's nope rows -> scratch, its rope rows -> the query;
                        w_iq -> the indexer query (32 heads x 128)
    W_UK (a wave 16 latents of a head)
                        the nope query against W_UK -> the bf16 absorbed query
                        (512 latents, then the 64 rope) in `q_out`
    indexer q quant (a wave a (token, head pair))
                        MXFP4 payload and scales, the head weights x `wscale`
    score (a CTA a 64-token index page)
                        FP4 MFMAs of the query against the page, ReLU^2, the
                        weighted sum over heads -> fp32 `scores`
    select (the last S CTAs, a row each; the split pass on S x NSL CTAs)
                        rows of at most TOPK tokens take every position (-1
                        padded); longer rows a radix select over order-keyed
                        scores (an 11-bit histogram over NSL slices, then
                        11 + 10 bits over the candidates, ties to exactly
                        TOPK), `select_slow` past CAP candidates -> `topk_out`

Pure decode only, the layer's shapes and dtypes checked by the runtime
(`runtime/dsa.py`, `front_reasons`).
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.kernels_common import ord_signed_f32
from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_rowgroup import (
    _fma,
    _permlane_swap,
    _relu2,
    _vop,
)
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import kda
from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
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
from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla import (
    A_TASKS,
    ENTRY,
    GEMV_CTAS,
    GROWS,
    KVL,
    NH,
    NOPE,
    QB,
    QB0,
    QB_SUB,
    QB_TASKS,
    QCROW,
    QHD,
    QKV_A,
    QL,
    QW,
    ROPE,
    ROWS,
    UK_GROUPS,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.debug import (
    region_ids,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.mx import (
    FP4,
    clamp_fp8,
    code_ceil,
    mfma_scaled,
    pow2,
    rcp_pow2,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.ops import (
    CM_DEV,
    bf16_pair,
    bf_hi,
    bf_lo,
    block_excl_scan,
    butterfly,
    fp8_pack4,
    hw_rsq,
    ld_i32,
    mfma_bf16,
    row_sum,
    rsrc,
    traced,
    uniform,
    xshfl,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    KernelAbi,
    first_task,
    key_tuple,
    pair_layout,
    source_digest,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.sync import (
    Mailbox,
    publish,
    sreg,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.stages import gemv

RECORD = KVL + 16 + 2 * ROPE
IH = 32
ID = 128
IQ = IH * ID
IPAGE = 64
G_TASKS = GROWS // ROWS
K_TASKS = ID // ROWS
V_TASKS = IH // ROWS
G0 = A_TASKS
K0 = G0 + G_TASKS
V0 = K0 + K_TASKS
IQ_TASKS = IQ // ROWS
TOPK = 2048
SPAGE = 64
CAP = 8192
SLCAP = 16 * THREADS
NSL = 16
CBCAP = 16384
NB2K = 2048
NB1, NB2, NB3 = 8192, 1024, 512
SH1, SH2 = 19, 9
SEL_IT = 16 * THREADS
INT_MIN = -(1 << 31)

assert V0 + V_TASKS <= 2 * GEMV_CTAS and QB0 >= MAX_S
assert NH * QHD == QB and QL % (4 * 384) == 0 and KVL == 4 * 128

_STREAM = fx.Stream(None)
REGIONS = (
    "part",
    "wgt",
    "msq",
    "xrdy",
    "qa",
    "qcr",
    "qn",
    "ik",
    "iw",
    "iq",
    "iqr",
    "spg",
    "sfa",
    "sfb",
)
_SOURCES = source_digest(".")


@dataclass(frozen=True, slots=True)
class DsaFrontBuild:
    """A `dsa_front` build: `tokens` rows, `nblocks` block residuals."""

    tokens: int
    nblocks: int
    delta: bool
    eps: float = 1e-5
    out_eps: float = 1e-5
    q_eps: float = 1e-5
    kv_eps: float = 1e-5
    k_eps: float = 1e-5
    wscale: float = 0.015625
    max_pages: int = 16384


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
        "w_wk",
        "w_wv",
        "q_nw",
        "kv_nw",
        "w_qb",
        "w_iq",
        "w_uk",
        "ln_w",
        "ln_b",
        "kv_cache",
        "kv_sb",
        "kv_ss",
        "kv_bs",
        "idx_cache",
        "idx_sb",
        "slots",
        "islots",
        "gate_out",
        "q_out",
        "iq_pay",
        "iq_sc",
        "iq_w",
        "idx_bt",
        "row_ends",
        "scores",
        "score_ld",
        "topk_out",
        "scratch",
        "layer",
        "epoch",
        "health",
        "alarm",
    )
)


def scratch_layout(key: DsaFrontBuild) -> dict:
    """Scratch regions (offset, words): the mailboxes, then x, q_c and select."""
    s, ns = key.tokens, key.nblocks + 1
    pairs = {
        "part": NA * s * ns * 2,
        "wgt": s * (ns + 1),
        "msq": NA * s,
        "xrdy": NA,
        "qa": s * QKV_A // 2,
        "qcr": s,
        "qn": s * NH * NOPE // 2,
        "ik": s * ID // 2,
        "iw": s * IH // 2,
        "iq": s * IQ // 2,
        "iqr": 2 * s,
        "spg": BLOCKS,
        "sfa": s * NSL,
        "sfb": s * NSL,
    }
    lay = pair_layout(pairs.items())
    end = max(o + n for o, n in lay.values())
    lay["x"] = ((end + 15) // 16 * 16, s * HIDDEN * 2)
    lay["qc"] = (lay["x"][0] + s * HIDDEN * 2, s * QL * 2)
    lay["sh"] = (lay["qc"][0] + s * QL * 2, s * NSL * NB2K * 4)
    lay["cb"] = (lay["sh"][0] + s * NSL * NB2K * 4, s * CBCAP * 8)
    lay["ri"] = (lay["cb"][0] + s * CBCAP * 8, s * 16)
    return lay


def scratch_bytes(key: DsaFrontBuild) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def put_q(c, t, h, w, v):
    pair = fx.Vector.from_elements(
        [bf16_pair(v[0], v[1]), bf16_pair(v[2], v[3])], fx.Float32
    )
    bo.buffer_store(pair, rsrc(c["args"]["q_out"]), ((t * NH + h) * QW + w) * 2)


def fp8_word(v):
    return fp8_pack4(*[clamp_fp8(x) for x in v])


def words_f32(w):
    return [bf_lo(w[0]), bf_hi(w[0]), bf_lo(w[1]), bf_hi(w[1])]


def fmax(a, b):
    return fx.max(a, b)


def ior(a, b):
    return a | b


def fabs(v):
    return (v.bitcast(fx.Int32) & 0x7FFFFFFF).bitcast(fx.Float32)


def fp4_code(v, rcp):
    x = v * rcp
    a = fabs(x)
    one, zero = fx.Int32(1), fx.Int32(0)
    code = (a > 0.25).select(one, zero)
    code = code + (a >= 0.75).select(one, zero)
    code = code + (a > 1.25).select(one, zero)
    code = code + (a >= 1.75).select(one, zero)
    code = code + (a > 2.5).select(one, zero)
    code = code + (a >= 3.5).select(one, zero)
    code = code + (a > 5.0).select(one, zero)
    neg = (x < 0.0) & (code != 0)
    return code | neg.select(fx.Int32(8), zero)


def fp4_scale_code(amax):
    sc = fx.max(amax / fx.Float32(6.0), fx.Float32(1.0e-4))
    bits = sc.bitcast(fx.Int32)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).select(
        fx.Int32(1), fx.Int32(0)
    )
    return fx.min(e, fx.Int32(254))


def proj_loads(c, task):
    a = c["args"]
    task = fx.min(task, c["NG"] - 1)
    one, zero = fx.Int32(1), fx.Int32(0)
    is_g = uniform(((task >= G0) & (task < K0)).select(one, zero))
    is_k = uniform(((task >= K0) & (task < V0)).select(one, zero))
    is_v = uniform((task >= V0).select(one, zero))
    w = a["w_qkv"] + fx.Int64(is_g) * (a["w_g"] - a["w_qkv"])
    row0 = task - is_g * G0
    w = w + fx.Int64(is_k) * (a["w_wk"] - a["w_qkv"])
    w = w + fx.Int64(is_v) * (a["w_wv"] - a["w_qkv"])
    row0 = row0 - is_k * K0 - is_v * V0
    return gemv.loads(c["lane"], c["wave"], w, row0 * ROWS, HIDDEN, nt=False)


@traced
def stage_proj(c, task, ops, t0, n):
    """A projection task's 16 rows against x, the bf16 rows -> scratch."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    acc = gemv.mfmas(lane, wave, c["xl"], HIDDEN + LDS_PAD, HIDDEN, n, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    r2 = tid % 8
    tt = tid // 8
    live = (tid < 8 * n) & (task < c["NG"])
    v0 = bf16r(row_sum(red, 2 * r2, fx.min(tt, n - 1)))
    v1 = bf16r(row_sum(red, 2 * r2 + 1, fx.min(tt, n - 1)))
    t = t0 + tt
    if live & (task < G0):
        c["put_bf"](c["qa"], t * QKV_A + task * ROWS + 2 * r2, [v0, v1])
    if live & (task >= G0) & (task < K0):
        col = (task - G0) * ROWS + 2 * r2
        g = bf16_pair(sigmoid(v0), sigmoid(v1))
        bo.buffer_store(g, rsrc(c["args"]["gate_out"]), (t * GROWS + col) // 2)
    if live & (task >= K0) & (task < V0):
        c["put_bf"](c["ik"], t * ID + (task - K0) * ROWS + 2 * r2, [v0, v1])
    if live & (task >= V0):
        c["put_bf"](c["iw"], t * IH + (task - V0) * ROWS + 2 * r2, [v0, v1])
    gpu.barrier()


@traced
def gemv_tasks(c, gi):
    s = c["S"]
    ops0 = proj_loads(c, gi)
    kda.wait_x(c)
    for t0, n in gemv.token_tiles(s, XT):
        kda.load_x(c, t0, n)
        stage_proj(c, gi, ops0, t0, n)
    if gi + GEMV_CTAS < c["NG"]:
        ops1 = proj_loads(c, gi + GEMV_CTAS)
        for t0, n in gemv.token_tiles(s, XT):
            if const_expr(s > XT):
                kda.load_x(c, t0, n)
            stage_proj(c, gi + GEMV_CTAS, ops1, t0, n)


@traced
def block_sum2(c, v):
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
    """Token t: q_a / kv_a RMSNorms, q_c out, the latent and rope into its KV slot."""
    tid, lane, a = c["tid"], c["lane"], c["args"]
    is_q = tid < QL // 4
    col = is_q.select(tid * 4, QL + (tid - QL // 4) * 4)
    pe = QL + KVL + fx.min(tid, ROPE // 4 - 1) * 4
    got = c["poll"](
        [(c["qa"], (t * QKV_A + col) // 2, 2), (c["qa"], (t * QKV_A + pe) // 2, 2)]
    )
    v = words_f32(got[0])
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
    am = fx.max(fx.max(fabs(kn[0]), fabs(kn[1])), fx.max(fabs(kn[2]), fabs(kn[3])))
    am = butterfly(am, (16, 8, 4, 2, 1), fmax)
    code = code_ceil(fx.max(am, fx.Float32(1.0e-4)) / fx.Float32(448.0))
    rcp = rcp_pow2(code)
    if slot >= 0:
        blk = slot // a["kv_bs"]
        off = slot % a["kv_bs"]
        base = (
            a["kv_cache"]
            + fx.Int64(blk) * fx.Int64(a["kv_sb"])
            + fx.Int64(off) * fx.Int64(a["kv_ss"])
        )
        kt = tid - QL // 4
        if tid >= QL // 4:
            bo.buffer_store(fp8_word([x * rcp for x in kn]), rsrc(base), kt)
            if lane % 32 == 0:
                bo.buffer_store(pow2(code), rsrc(base), KVL // 4 + kt // 32)
        if tid < ROPE // 4:
            p = fx.Vector.from_elements(got[1], fx.Int32)
            bo.buffer_store(p, rsrc(base), (KVL + 16) // 4 + 2 * tid)
    stage_key(c, t, uniform(ld_i32(a["islots"], t * 2)))
    publish(c["put"], c["qcr"], t, fx.Int32(1), tid == 0)
    gpu.barrier()


@traced
def stage_key(c, t, slot):
    """Token t's indexer key: LayerNorm -> MXFP4 payload and E8M0 scale in its page."""
    tid, lane, a = c["tid"], c["lane"], c["args"]
    kl = fx.min(tid, ID // 4 - 1)
    got = c["poll"]([(c["ik"], (t * ID + kl * 4) // 2, 2)])
    v = words_f32(got[0])
    mean = butterfly(v[0] + v[1] + v[2] + v[3], (16, 8, 4, 2, 1)) * fx.Float32(1.0 / ID)
    d = [v[j] - mean for j in range(4)]
    var = butterfly(
        d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3], (16, 8, 4, 2, 1)
    ) * fx.Float32(1.0 / ID)
    rstd = hw_rsq(var + fx.Float32(c["k_eps"]))
    lw = ld_bf4(a["ln_w"], kl * 4)
    lb = ld_bf4(a["ln_b"], kl * 4)
    y = [bf16r(d[j] * rstd * lw[j] + lb[j]) for j in range(4)]
    am = fx.max(fx.max(fabs(y[0]), fabs(y[1])), fx.max(fabs(y[2]), fabs(y[3])))
    am = butterfly(am, (4, 2, 1), fmax)
    code = fp4_scale_code(am)
    rcp = rcp_pow2(code)
    cs = [fp4_code(y[j], rcp) for j in range(4)]
    half = cs[0] | (cs[1] << 4) | (cs[2] << 8) | (cs[3] << 12)
    word = half << ((lane % 2) * 16)
    word = word | xshfl(word, 1)
    if slot >= 0:
        page = slot // IPAGE
        off = slot % IPAGE
        g = lane // 8
        base = (
            a["idx_cache"]
            + fx.Int64(page) * fx.Int64(a["idx_sb"])
            + fx.Int64(g * 1024 + off * 16)
        )
        if (tid < ID // 4) & (lane % 2 == 0):
            bo.buffer_store(word, rsrc(base), (lane % 8) // 2, cache_modifier=CM_DEV)
        sbase = (
            a["idx_cache"]
            + fx.Int64(page) * fx.Int64(a["idx_sb"])
            + fx.Int64(4096 + g * 64 + (off % 16) * 4 + off // 16)
        )
        if (tid < ID // 4) & (lane % 8 == 0):
            bo.buffer_store(
                fx.Int8(code),
                rsrc(sbase),
                0,
                cache_modifier=CM_DEV,
                offset_is_bytes=True,
            )


@traced
def stage_qb(c, task):
    """A q_b (or w_iq) task's 16 rows against q_c: nope rows, rope rows or indexer q."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    h = task // QB_SUB
    sub = task % QB_SUB
    is_i = uniform((task >= c["NQB"]).select(fx.Int32(1), fx.Int32(0)))
    w = a["w_qb"]
    row0 = task * ROWS
    w = w + fx.Int64(is_i) * (a["w_iq"] - a["w_qb"])
    row0 = row0 - is_i * (c["NQB"] * ROWS)
    ops = gemv.loads(lane, wave, w, row0, QL, nt=False)
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
        v0 = row_sum(red, 2 * r2, tt)
        v1 = row_sum(red, 2 * r2 + 1, tt)
        if (tid < 8 * n) & (is_i == 0) & (sub < NOPE // ROWS):
            at = ((t0 + tt) * NH + h) * NOPE + sub * ROWS + 2 * r2
            c["put_bf"](c["qn"], at, [v0, v1])
        if (tid < 8 * n) & (is_i == 1):
            at = (t0 + tt) * IQ + row0 + 2 * r2
            c["put_bf"](c["iq"], at, [v0, v1])
        r4 = tid % 4
        t4 = fx.min(tid // 4, n - 1)
        if (tid < 4 * n) & (is_i == 0) & (sub >= NOPE // ROWS):
            v = [bf16r(row_sum(red, 4 * r4 + j, t4)) for j in range(4)]
            wq = KVL // 4 + (sub - NOPE // ROWS) * 4 + r4
            put_q(c, t0 + t4, h, wq, v)
        gpu.barrier()


@traced
def stage_uk(c, task):
    """16 tokens x 16 latents of a head: the nope query against W_UK -> `q_out`."""
    s, lane = c["S"], c["lane"]
    a = c["args"]
    ng = task % UK_GROUPS
    h = (task // UK_GROUPS) % NH
    t0 = task // (UK_GROUPS * NH) * 16
    r = lane % 16
    g = lane // 16
    r_w = rsrc(a["w_uk"])
    wbase = ((h * KVL + ng * ROWS + r) * NOPE + g * 8) // 2
    ws = [
        fx.Vector(bo.buffer_load(r_w, wbase + 16 * q, vec_width=4, dtype=T.i32))
        for q in range(4)
    ]
    t = t0 + r
    tc = fx.min(t, s - 1)
    p0 = ((tc * NH + h) * NOPE + g * 8) // 2
    got = c["poll"](
        [(c["qn"], p0 + 16 * q + 2 * j, 2) for q in range(4) for j in range(2)]
    )
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for q in range_constexpr(4):
        xw = got[2 * q] + got[2 * q + 1]
        xb = fx.Vector.from_elements(xw, fx.Int32)
        acc = mfma_bf16(ws[q].bitcast(fx.BFloat16), xb.bitcast(fx.BFloat16), acc)
    v = [bf16r(acc[j]) for j in range(4)]
    if t < s:
        put_q(c, t, h, ng * 4 + g, v)


@traced
def uk_tasks(c):
    s, bid, wave = c["S"], c["bid"], c["wave"]
    ntask = ((s + 15) // 16) * NH * UK_GROUPS
    first = first_task(bid, c["UK0"]) * WAVES + wave
    for task in range(first, ntask, BLOCKS * WAVES):
        stage_uk(c, task)


@traced
def stage_iq(c, task):
    """Token t, heads hp and hp + 16: the indexer query's MXFP4 quant and weights."""
    lane, a = c["lane"], c["args"]
    hp = task % 16
    t = task // 16
    g = lane // 16
    got = c["poll"](
        [
            (c["iq"], (t * IQ + hp * ID) // 2 + lane, 1),
            (c["iq"], (t * IQ + (hp + 16) * ID) // 2 + lane, 1),
            (c["iw"], (t * IH + hp) // 2, 1),
            (c["iw"], (t * IH + hp + 16) // 2, 1),
        ]
    )
    sc = fx.Int32(0)
    for i in range_constexpr(2):
        h = hp + 16 * i
        w = got[i][0]
        v0, v1 = bf_lo(w), bf_hi(w)
        am = butterfly(fx.max(fabs(v0), fabs(v1)), (8, 4, 2, 1), fmax)
        code = fp4_scale_code(am)
        rcp = rcp_pow2(code)
        byte = fp4_code(v0, rcp) | (fp4_code(v1, rcp) << 4)
        word = byte << ((lane % 4) * 8)
        word = butterfly(word, (2, 1), ior)
        if lane % 4 == 0:
            bo.buffer_store(
                word,
                rsrc(a["iq_pay"]),
                (t * IH + h) * 16 + lane // 4,
                cache_modifier=CM_DEV,
            )
        sc = sc | (code << (8 * i))
    if lane % 16 == 0:
        bo.buffer_store(
            sc, rsrc(a["iq_sc"]), t * 64 + g * 16 + hp, cache_modifier=CM_DEV
        )
    if lane == 0:
        for i in range_constexpr(2):
            w = got[2 + i][0]
            v = (hp % 2 == 0).select(bf_lo(w), bf_hi(w))
            bo.buffer_store(
                v * fx.Float32(c["wscale"]),
                rsrc(a["iq_w"]),
                t * IH + hp + 16 * i,
                cache_modifier=CM_DEV,
            )


@traced
def row_end(c, t):
    return uniform(ld_i32(c["args"]["row_ends"], t))


def ld_dev4(ptr, i):
    return fx.Vector(
        bo.buffer_load(rsrc(ptr), i, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV)
    )


def ld_dev1(ptr, i):
    return fx.Int32(
        bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV)
    )


@traced
def score_pages(c):
    s = c["S"]
    np_ = fx.Int32(0)
    for t in range_constexpr(s):
        e = row_end(c, t)
        np_ = fx.max(np_, (e > TOPK).select((e + SPAGE - 1) // SPAGE, fx.Int32(0)))
    return np_


@traced
def stage_score(c, p):
    """Index page p against every long row's query: fp32 scores per token."""
    s, lane, wave, a = c["S"], c["lane"], c["wave"], c["args"]
    ln = lane % 16
    g = lane // 16
    phys = uniform(ld_i32(a["idx_bt"], p))
    base = a["idx_cache"] + fx.Int64(phys) * fx.Int64(a["idx_sb"])
    kv = [ld_dev4(base, (g * 1024 + ln * 16 + nt * 256) // 4) for nt in range(4)]
    kvs = ld_dev1(base, (4096 + g * 64 + ln * 4) // 4)
    for i in range_constexpr((s + WAVES - 1) // WAVES):
        row = wave + WAVES * i
        rc = fx.min(row, s - 1)
        e = row_end(c, rc)
        if (row < s) & (e > TOPK) & (p * SPAGE < e):
            c["poll"]([(c["iqr"], 2 * rc, 1), (c["iqr"], 2 * rc + 1, 1)])
            q = [
                ld_dev4(a["iq_pay"], ((rc * IH + 16 * mi + ln) * 64 + 16 * g) // 4)
                for mi in range(2)
            ]
            qs = ld_dev1(a["iq_sc"], rc * 64 + g * 16 + ln)
            wv = [
                ld_dev4(a["iq_w"], rc * IH + 16 * mi + 4 * g).bitcast(fx.Float32)
                for mi in range(2)
            ]
            wl = [[wv[mi][e4] * fx.Float32(0.5) for e4 in range(4)] for mi in range(2)]
            zero = fx.Vector.filled(4, 0.0, fx.Float32)
            pt = []
            for nt in range_constexpr(4):
                accs = [
                    mfma_scaled(
                        q[mi],
                        kv[nt],
                        zero,
                        qs >> (8 * mi),
                        (kvs >> (8 * nt)) & 0xFF,
                        FP4,
                        FP4,
                        0,
                        0,
                    )
                    for mi in range(2)
                ]
                total = None
                for mi in range_constexpr(2):
                    for e4 in range_constexpr(4):
                        tv = _relu2(accs[mi][e4])
                        w = wl[mi][e4]
                        total = (
                            _vop("v_mul_f32 $0, $1, $2", tv, w)
                            if total is None
                            else _fma(tv, w, total)
                        )
                pt.append(total)
            x, y = _permlane_swap(rocdl.permlane32_swap, pt[0], pt[2])
            lo = x + y
            x, y = _permlane_swap(rocdl.permlane32_swap, pt[1], pt[3])
            hi = x + y
            x, y = _permlane_swap(rocdl.permlane16_swap, lo, hi)
            tot = x + y
            key = p * SPAGE + g * 16 + ln
            if key < e:
                bo.buffer_store(
                    fx.Float32(tot),
                    rsrc(a["scores"] + fx.Int64(rc) * fx.Int64(a["score_ld"]) * 4),
                    key,
                    cache_modifier=CM_DEV,
                )


@traced
def score_tasks(c):
    s, bid, tid = c["S"], c["bid"], c["tid"]
    if tid < s:
        c["poll"]([(c["qcr"], tid, 1)])
    gpu.barrier()
    np_ = score_pages(c)
    for p in range(bid, np_, BLOCKS):
        stage_score(c, p)
    publish(c["put"], c["spg"], bid, fx.Int32(1), tid == 0)


def okey(v):
    return ord_signed_f32(fx.Float32(v))


def ukey(o):
    return o ^ fx.Int32(INT_MIN)


@traced
def zero_hist(c, nb, h=None):
    h = c["hist"] if h is None else h
    for i in range_constexpr(nb // THREADS):
        fx.ptr_store(fx.Int32(0), h + (c["tid"] + THREADS * i))
    gpu.barrier()


@traced
def find_bin(c, nb, need, h=None):
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    h = c["hist"] if h is None else h
    per = nb // THREADS
    hs = [fx.ptr_load(h + (nb - 1 - (tid * per + i))) for i in range(per)]
    tot = hs[0]
    for i in range_constexpr(1, per):
        tot = tot + hs[i]
    above, _ = block_excl_scan(tot, lane, wave, c["scan"])
    run = above
    for i in range_constexpr(per):
        b = nb - 1 - (tid * per + i)
        if (run < need) & (run + hs[i] >= need):
            fx.ptr_store(fx.Int32(b), c["misc"] + 0)
            fx.ptr_store(run, c["misc"] + 1)
        run = run + hs[i]
    gpu.barrier()
    b = fx.ptr_load(c["misc"] + 0)
    ab = fx.ptr_load(c["misc"] + 1)
    gpu.barrier()
    return b, ab


@traced
def each_global(c, t, e, body):
    a, tid = c["args"], c["tid"]
    row = a["scores"] + fx.Int64(t) * fx.Int64(a["score_ld"]) * 4
    nit = (e + SEL_IT - 1) // SEL_IT
    for it in range(0, nit, 1):
        b0 = it * SEL_IT + tid * 16
        vs = [ld_dev4(row, b0 + 4 * q) for q in range(4)]
        for j in range_constexpr(16):
            idx = b0 + j
            o = okey(vs[j // 4][j % 4].bitcast(fx.Float32))
            body(idx, o, idx < e)


@traced
def each_cand(c, n, body):
    tid = c["tid"]
    for j in range_constexpr(CAP // THREADS):
        i = tid + THREADS * j
        live = i < n
        ic = fx.min(i, CAP - 1)
        o = fx.ptr_load(c["cu"] + ic)
        idx = fx.ptr_load(c["ci"] + ic)
        body(idx, o, live)


def hadd(c, b):
    fx.llvm.atomic_add(c["hist"] + b, fx.Int32(1), syncscope="workgroup")


@traced
def select_slow(c, t, e):
    """Row t's top-k by a radix select over all its scores (split fallback)."""
    tid, lane, wave, a = c["tid"], c["lane"], c["wave"], c["args"]
    out = a["topk_out"]
    zero_hist(c, NB1)

    def b1_body(idx, o, live):
        if live:
            hadd(c, (ukey(o) >> SH1) & (NB1 - 1))

    each_global(c, t, e, b1_body)
    gpu.barrier()
    b1, ab1 = find_bin(c, NB1, fx.Int32(TOPK))
    if tid == 0:
        fx.ptr_store(fx.Int32(0), c["misc"] + 2)
    gpu.barrier()

    def cand_body(idx, o, live):
        if live & (((ukey(o) >> SH1) & (NB1 - 1)) == b1):
            pos = fx.Int32(
                fx.llvm.atomic_add(c["misc"] + 2, fx.Int32(1), syncscope="workgroup")
            )
            if pos < CAP:
                fx.ptr_store(o, c["cu"] + pos)
                fx.ptr_store(idx, c["ci"] + pos)

    each_global(c, t, e, cand_body)
    gpu.barrier()
    nc = fx.ptr_load(c["misc"] + 2)
    need2 = fx.Int32(TOPK) - ab1
    zero_hist(c, NB2)

    def b2_body(idx, o, live):
        u = ukey(o)
        if live & (((u >> SH1) & (NB1 - 1)) == b1):
            hadd(c, (u >> SH2) & (NB2 - 1))

    if nc <= CAP:
        each_cand(c, nc, b2_body)
    if nc > CAP:
        each_global(c, t, e, b2_body)
    gpu.barrier()
    b2, ab2 = find_bin(c, NB2, need2)
    need3 = need2 - ab2
    zero_hist(c, NB3)

    def b3_body(idx, o, live):
        u = ukey(o)
        if live & (
            ((u >> SH2) & ((1 << (32 - SH2)) - 1)) == ((b1 << (SH1 - SH2)) | b2)
        ):
            hadd(c, u & (NB3 - 1))

    if nc <= CAP:
        each_cand(c, nc, b3_body)
    if nc > CAP:
        each_global(c, t, e, b3_body)
    gpu.barrier()
    b3, ab3 = find_bin(c, NB3, need3)
    thr = ukey((b1 << SH1) | (b2 << SH2) | b3)
    ng = ab1 + ab2 + ab3
    ne = fx.Int32(TOPK) - ng
    if tid == 0:
        fx.ptr_store(fx.Int32(0), c["misc"] + 3)
        fx.ptr_store(fx.Int32(0), c["misc"] + 4)
    gpu.barrier()
    row = a["scores"] + fx.Int64(t) * fx.Int64(a["score_ld"]) * 4
    nit = (e + SEL_IT - 1) // SEL_IT
    for it in range(0, nit, 1):
        rg = fx.ptr_load(c["misc"] + 3)
        re_ = fx.ptr_load(c["misc"] + 4)
        b0 = it * SEL_IT + tid * 16
        vs = [ld_dev4(row, b0 + 4 * q) for q in range(4)]
        os_ = [okey(vs[j // 4][j % 4].bitcast(fx.Float32)) for j in range(16)]
        gts = [
            ((b0 + j < e) & (os_[j] > thr)).select(fx.Int32(1), fx.Int32(0))
            for j in range(16)
        ]
        eqs = [
            ((b0 + j < e) & (os_[j] == thr)).select(fx.Int32(1), fx.Int32(0))
            for j in range(16)
        ]
        ngl = gts[0]
        nel = eqs[0]
        for j in range_constexpr(1, 16):
            ngl = ngl + gts[j]
            nel = nel + eqs[j]
        ex, tot = block_excl_scan(ngl + (nel << 16), lane, wave, c["scan"])
        og = rg + (ex & 0xFFFF)
        oe = re_ + (ex >> 16)
        for j in range_constexpr(16):
            if gts[j] == 1:
                bo.buffer_store(b0 + j, rsrc(out), t * TOPK + og)
            if (eqs[j] == 1) & (oe < ne):
                bo.buffer_store(b0 + j, rsrc(out), t * TOPK + ng + oe)
            og = og + gts[j]
            oe = oe + eqs[j]
        gpu.barrier()
        if tid == 0:
            fx.ptr_store(rg + (tot & 0xFFFF), c["misc"] + 3)
            fx.ptr_store(re_ + (tot >> 16), c["misc"] + 4)
        gpu.barrier()


@traced
def hadd2(c, b):
    fx.llvm.atomic_add(c["h2"] + b, fx.Int32(1), syncscope="workgroup")


@traced
def hadd2_if(c, hit, b):
    if hit:
        hadd2(c, b)


@traced
def kth_digit(c, src, n, need, pre, sh, bits, pmask):
    tid = c["tid"]
    nb = 1 << bits
    zero_hist(c, NB2K, c["h2"])
    for j in range(tid, n, THREADS):
        u = ukey(fx.ptr_load(src + j))
        if const_expr(pmask == 0):
            hadd2(c, (u >> sh) & (nb - 1))
        if const_expr(pmask != 0):
            hadd2_if(c, ((u >> (sh + bits)) & pmask) == pre, (u >> sh) & (nb - 1))
    gpu.barrier()
    return find_bin(c, nb, need, c["h2"])


def ld_dev2(ptr, i):
    return fx.Vector(
        bo.buffer_load(rsrc(ptr), i, vec_width=2, dtype=T.i32, cache_modifier=CM_DEV)
    )


@traced
def split_slice(c, r, sl):
    """Slice sl of row r's scores: its 11-bit histogram and candidates."""
    tid, lane, wave, a = c["tid"], c["lane"], c["wave"], c["args"]
    NSL, SPLIT_MAX = c["NSL"], c["NSL"] * SLCAP
    e = row_end(c, r)
    if (e > TOPK) & (e <= SPLIT_MAX):
        if tid < BLOCKS:
            c["poll"]([(c["spg"], tid, 1)])
        gpu.barrier()
        ls = (e + NSL * 64 - 1) // (NSL * 64) * 64
        k0 = sl * ls
        k1 = fx.min(k0 + ls, e)
        row = a["scores"] + fx.Int64(r) * fx.Int64(a["score_ld"]) * 4
        base = k0 + tid * 16
        vs = [ld_dev4(row, base + 4 * q) for q in range(4)]
        os_ = [okey(vs[j // 4][j % 4].bitcast(fx.Float32)) for j in range(16)]
        lv = [base + j < k1 for j in range(16)]
        bins = [(ukey(os_[j]) >> 21) & (NB2K - 1) for j in range(16)]
        h = c["h2"]
        zero_hist(c, NB2K, h)
        for j in range_constexpr(16):
            if lv[j]:
                hadd2(c, bins[j])
        gpu.barrier()
        hrow = c["sh"] + fx.Int64((r * NSL + sl) * NB2K) * 4
        hv = [fx.ptr_load(h + (tid * 4 + i)) for i in range(4)]
        bo.buffer_store(
            fx.Vector.from_elements(hv, fx.Int32),
            rsrc(hrow),
            tid * 4,
            cache_modifier=CM_DEV,
        )
        publish(c["put"], c["sfa"], r * NSL + sl, fx.Int32(1), tid == 0)
        if tid < NSL:
            c["poll"]([(c["sfa"], r * NSL + tid, 1)])
        gpu.barrier()
        tt = c["gb"]
        pp = c["gb"] + NB2K
        acc = [fx.Int32(0)] * 4
        pre = [fx.Int32(0)] * 4
        for g4 in range_constexpr(NSL // 4):
            hs = [
                ld_dev4(c["sh"] + fx.Int64((r * NSL + 4 * g4 + q) * NB2K) * 4, tid * 4)
                for q in range(4)
            ]
            for q in range_constexpr(4):
                for i in range_constexpr(4):
                    acc[i] = acc[i] + hs[q][i]
                    pre[i] = pre[i] + (4 * g4 + q < sl).select(hs[q][i], fx.Int32(0))
        for i in range_constexpr(4):
            fx.ptr_store(acc[i], tt + (tid * 4 + i))
            fx.ptr_store(pre[i], pp + (tid * 4 + i))
        gpu.barrier()
        b1, ab1 = find_bin(c, NB2K, fx.Int32(TOPK), tt)
        pv = fx.Int32(0)
        for i in range_constexpr(4):
            pv = pv + (tid * 4 + i > b1).select(pre[i], fx.Int32(0))
        _, apre = block_excl_scan(pv, lane, wave, c["scan"])
        cpre = fx.ptr_load(pp + b1)
        c1 = fx.ptr_load(tt + b1)
        gts = [
            (lv[j] & (bins[j] > b1)).select(fx.Int32(1), fx.Int32(0)) for j in range(16)
        ]
        cfs = [
            (lv[j] & (bins[j] == b1)).select(fx.Int32(1), fx.Int32(0))
            for j in range(16)
        ]
        cg = gts[0]
        cc = cfs[0]
        for j in range_constexpr(1, 16):
            cg = cg + gts[j]
            cc = cc + cfs[j]
        ex, _ = block_excl_scan(cg + (cc << 16), lane, wave, c["scan"])
        og = apre + (ex & 0xFFFF)
        oc = cpre + (ex >> 16)
        out = a["topk_out"]
        cbr = c["cb"] + fx.Int64(r * CBCAP) * 8
        for j in range_constexpr(16):
            if gts[j] == 1:
                bo.buffer_store(base + j, rsrc(out), r * TOPK + og)
            if (cfs[j] == 1) & (oc < CBCAP):
                bo.buffer_store(
                    fx.Vector.from_elements([os_[j], base + j], fx.Int32),
                    rsrc(cbr),
                    oc * 2,
                    cache_modifier=CM_DEV,
                )
            og = og + gts[j]
            oc = oc + cfs[j]
        if (sl == 0) & (tid == 0):
            bo.buffer_store(
                fx.Vector.from_elements([b1, ab1, c1, fx.Int32(0)], fx.Int32),
                rsrc(c["ri"]),
                r * 4,
                cache_modifier=CM_DEV,
            )
        publish(c["put"], c["sfb"], r * NSL + sl, fx.Int32(1), tid == 0)


@traced
def split_tasks(c):
    s, bid, NSL = c["S"], c["bid"], c["NSL"]
    if bid < s * NSL:
        split_slice(c, bid // NSL, bid % NSL)


@traced
def split_select(c, t, e):
    """Row t from the slices' histograms: the exact top-k in two more digits."""
    tid, lane, wave, a = c["tid"], c["lane"], c["wave"], c["args"]
    cu, ci, misc = c["cu"], c["ci"], c["misc"]
    NSL = c["NSL"]
    out = a["topk_out"]
    if tid < NSL:
        c["poll"]([(c["sfb"], t * NSL + tid, 1)])
    gpu.barrier()
    riv = ld_dev4(c["ri"], t * 4)
    b1 = uniform(riv[0])
    ab1 = uniform(riv[1])
    c1 = uniform(riv[2])
    if c1 <= CAP:
        per = (c1 + THREADS - 1) // THREADS
        cbr = c["cb"] + fx.Int64(t * CBCAP) * 8
        prs = [ld_dev2(cbr, fx.min(tid * per + i, CBCAP - 1) * 2) for i in range(16)]
        for i in range_constexpr(16):
            jj = tid * per + i
            if (i < per) & (jj < c1):
                fx.ptr_store(prs[i][0], cu + jj)
                fx.ptr_store(prs[i][1], ci + jj)
        gpu.barrier()
        need = fx.Int32(TOPK) - ab1
        b2, ab2 = kth_digit(c, cu, c1, need, b1, 10, 11, 0x7FF)
        b3, ab3 = kth_digit(c, cu, c1, need - ab2, (b1 << 11) | b2, 0, 10, 0x3FFFFF)
        thr = ukey((b1 << 21) | (b2 << 10) | b3)
        g2 = ab2 + ab3
        ne = need - g2
        xe_ = []
        gts = []
        eqs = []
        for i in range_constexpr(16):
            jj = tid * per + i
            lvv = (i < per) & (jj < c1)
            jc = fx.min(jj, CAP - 1)
            oc = fx.ptr_load(cu + jc)
            xe_.append(fx.ptr_load(ci + jc))
            gts.append((lvv & (oc > thr)).select(fx.Int32(1), fx.Int32(0)))
            eqs.append((lvv & (oc == thr)).select(fx.Int32(1), fx.Int32(0)))
        cg = gts[0]
        ce = eqs[0]
        for i in range_constexpr(1, 16):
            cg = cg + gts[i]
            ce = ce + eqs[i]
        ex2, _ = block_excl_scan(cg + (ce << 16), lane, wave, c["scan"])
        og = ab1 + (ex2 & 0xFFFF)
        oe = ex2 >> 16
        for i in range_constexpr(16):
            if gts[i] == 1:
                bo.buffer_store(xe_[i], rsrc(out), t * TOPK + og)
            if (eqs[i] == 1) & (oe < ne):
                bo.buffer_store(xe_[i], rsrc(out), t * TOPK + ab1 + g2 + oe)
            og = og + gts[i]
            oe = oe + eqs[i]
        if tid == 0:
            fx.ptr_store(fx.Int32(1), misc + 5)
    gpu.barrier()


@traced
def select_row(c, t):
    """Row t's top-k: every position, the split select, or `select_slow`."""
    tid, a = c["tid"], c["args"]
    e = row_end(c, t)
    out = a["topk_out"]
    if e <= TOPK:
        for j in range_constexpr(TOPK // THREADS):
            i = tid + THREADS * j
            bo.buffer_store((i < e).select(i, fx.Int32(-1)), rsrc(out), t * TOPK + i)
    if e > TOPK:
        if tid < BLOCKS:
            c["poll"]([(c["spg"], tid, 1)])
        if tid == 0:
            fx.ptr_store(fx.Int32(0), c["misc"] + 5)
        gpu.barrier()
        if e <= c["NSL"] * SLCAP:
            split_select(c, t, e)
        done = fx.ptr_load(c["misc"] + 5)
        gpu.barrier()
        if done == 0:
            select_slow(c, t, e)


@traced
def select_tasks(c):
    s, bid = c["S"], c["bid"]
    t = bid - (BLOCKS - s)
    if t >= 0:
        select_row(c, t)


@traced
def iq_tasks(c):
    s, bid, wave = c["S"], c["bid"], c["wave"]
    ntask = s * 16
    first = bid * WAVES + wave
    for task in range(first, ntask, BLOCKS * WAVES):
        stage_iq(c, task)


_BUILDS: dict = {}


def build(key: DsaFrontBuild):
    """The compiled launch for `key` (cached)."""
    if key in _BUILDS:
        return _BUILDS[key]
    s = key.tokens
    ns = key.nblocks + 1
    lay = scratch_layout(key)
    xrow = HIDDEN + LDS_PAD
    assert 1 <= s <= MAX_S and s * NSL <= BLOCKS - s and key.nblocks >= 1
    assert key.max_pages % (SEL_IT // SPAGE) == 0
    ng, nq = V0 + V_TASKS, QB_TASKS + IQ_TASKS
    uk0 = QB0 + nq

    @fx.struct
    class AttnLds:
        vals: fx.Array[fx.Float32, vals_words(s, ns), 16]

    @fx.struct
    class GemvLds:
        xl: fx.Array[fx.Int32, max(min(s, XT) * xrow, 16 * QCROW) // 2, 16]

    @fx.struct
    class SelLds:
        h2: fx.Array[fx.Int32, NB2K, 16]
        gb: fx.Array[fx.Int32, NB1, 16]
        cu: fx.Array[fx.Int32, CAP, 16]
        ci: fx.Array[fx.Int32, CAP, 16]
        scan: fx.Array[fx.Int32, 16, 16]
        misc: fx.Array[fx.Int32, 16, 16]

    @fx.union
    class RoleLds:
        attn: AttnLds
        gemv: GemvLds
        sel: SelLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]

    keyed = key_tuple(key, _SOURCES)
    name = f"k3_mono_dsa_front_s{s}_nb{key.nblocks}_d{int(key.delta)}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def dsa_front(
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
        w_wk: Int64,
        w_wv: Int64,
        q_nw: Int64,
        kv_nw: Int64,
        w_qb: Int64,
        w_iq: Int64,
        w_uk: Int64,
        ln_w: Int64,
        ln_b: Int64,
        kv_cache: Int64,
        kv_sb: Int64,
        kv_ss: Int32,
        kv_bs: Int32,
        idx_cache: Int64,
        idx_sb: Int64,
        slots: Int64,
        islots: Int64,
        gate_out: Int64,
        q_out: Int64,
        iq_pay: Int64,
        iq_sc: Int64,
        iq_w: Int64,
        idx_bt: Int64,
        row_ends: Int64,
        scores: Int64,
        score_ld: Int32,
        topk_out: Int64,
        scratch: Int64,
        layer: Int32,
        epoch: Int64,
        health: Int64,
        alarm: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        role = alloc.allocate(RoleLds)
        al, gl, sl = role.attn.peek(), role.gemv.peek(), role.sel.peek()
        mb = Mailbox(step_tag(epoch, layer) - 1, health, alarm, region_ids(REGIONS))
        c = {
            "S": s,
            "NS": ns,
            "NB": key.nblocks,
            "NG": ng,
            "UK0": uk0,
            "NQB": QB_TASKS,
            "wscale": key.wscale,
            "hist": sl.gb.ptr,
            "gb": sl.gb.ptr,
            "h2": sl.h2.ptr,
            "NSL": NSL,
            "sh": scratch + fx.Int64(lay["sh"][0]),
            "cb": scratch + fx.Int64(lay["cb"][0]),
            "ri": scratch + fx.Int64(lay["ri"][0]),
            "cu": sl.cu.ptr,
            "ci": sl.ci.ptr,
            "scan": sl.scan.ptr,
            "misc": sl.misc.ptr,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "delta": key.delta,
            "eps": key.eps,
            "out_eps": key.out_eps,
            "q_eps": key.q_eps,
            "kv_eps": key.kv_eps,
            "k_eps": key.k_eps,
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
                "w_wk": w_wk,
                "w_wv": w_wv,
                "q_nw": q_nw,
                "kv_nw": kv_nw,
                "w_qb": w_qb,
                "w_iq": w_iq,
                "w_uk": w_uk,
                "ln_w": ln_w,
                "ln_b": ln_b,
                "kv_cache": kv_cache,
                "kv_sb": kv_sb,
                "kv_ss": kv_ss,
                "kv_bs": kv_bs,
                "idx_cache": idx_cache,
                "idx_sb": idx_sb,
                "slots": slots,
                "islots": islots,
                "gate_out": gate_out,
                "q_out": q_out,
                "iq_pay": iq_pay,
                "iq_sc": iq_sc,
                "iq_w": iq_w,
                "idx_bt": idx_bt,
                "row_ends": row_ends,
                "scores": scores,
                "score_ld": score_ld,
                "topk_out": topk_out,
            },
        }
        for region in REGIONS:
            c[region] = sreg(scratch, lay[region][0], region)
        c["on"], c["tls"] = False, 0
        if bid < NA:
            stage_slice(c, bid)
            stage_gates(c, bid)
            stage_mix(c, bid)
        if bid >= NA:
            gemv_tasks(c, bid - NA)
        for t in range(first_task(bid, 0), s, BLOCKS):
            stage_norm(c, t)
        for task in range(first_task(bid, QB0), nq, BLOCKS):
            stage_qb(c, task)
        uk_tasks(c)
        iq_tasks(c)
        if bid < (s * 16 + WAVES - 1) // WAVES:
            publish(c["put"], c["iqr"], bid, fx.Int32(1), tid == 0)
        score_tasks(c)
        split_tasks(c)
        select_tasks(c)

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
        w_wk: Int64,
        w_wv: Int64,
        q_nw: Int64,
        kv_nw: Int64,
        w_qb: Int64,
        w_iq: Int64,
        w_uk: Int64,
        ln_w: Int64,
        ln_b: Int64,
        kv_cache: Int64,
        kv_sb: Int64,
        kv_ss: Int32,
        kv_bs: Int32,
        idx_cache: Int64,
        idx_sb: Int64,
        slots: Int64,
        islots: Int64,
        gate_out: Int64,
        q_out: Int64,
        iq_pay: Int64,
        iq_sc: Int64,
        iq_w: Int64,
        idx_bt: Int64,
        row_ends: Int64,
        scores: Int64,
        score_ld: Int32,
        topk_out: Int64,
        scratch: Int64,
        layer: Int32,
        epoch: Int64,
        health: Int64,
        alarm: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        dsa_front(
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
            w_wk,
            w_wv,
            q_nw,
            kv_nw,
            w_qb,
            w_iq,
            w_uk,
            ln_w,
            ln_b,
            kv_cache,
            kv_sb,
            kv_ss,
            kv_bs,
            idx_cache,
            idx_sb,
            slots,
            islots,
            gate_out,
            q_out,
            iq_pay,
            iq_sc,
            iq_w,
            idx_bt,
            row_ends,
            scores,
            score_ld,
            topk_out,
            scratch,
            layer,
            epoch,
            health,
            alarm,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(dsa_front, launch)
    _BUILDS[key] = launch
    return launch


def ptr(t):
    return 0 if t is None else t.data_ptr()


def dsa_front(
    key: DsaFrontBuild,
    *,
    prefix,
    delta,
    blocks,
    ares_nw,
    ares_qk,
    in_nw,
    w_qkv,
    w_g,
    w_wk,
    w_wv,
    q_nw,
    kv_nw,
    w_qb,
    w_iq,
    w_uk,
    ln_w,
    ln_b,
    kv_cache,
    idx_cache,
    slots,
    islots,
    gate_out,
    q_out,
    iq_pay,
    iq_sc,
    iq_w,
    scratch,
    layer,
    epoch,
    health,
    alarm,
    idx_bt=None,
    row_ends=None,
    scores=None,
    topk_out=None,
):
    """Launch `dsa_front` for `key` on the current stream."""
    assert kv_cache.stride(-1) == 1 and kv_cache.size(-1) == RECORD
    assert slots.dtype == torch.int64 and slots.size(0) == key.tokens
    assert islots is None or (islots.dtype == torch.int64 and islots.stride(0) == 1)
    assert slots.stride(0) == 1
    assert q_out.is_contiguous() and gate_out.is_contiguous()
    assert q_out.shape[1:] == (NH, ENTRY) and q_out.element_size() == 2
    assert w_uk.shape == (NH, KVL, NOPE) and w_uk.is_contiguous()
    assert w_qb.shape == (QB, QL) and w_qb.is_contiguous()
    assert idx_cache.stride(1) == 68
    assert row_ends.dtype == torch.int32 and idx_bt.dtype == torch.int32
    assert scores.dtype == torch.float32 and scores.stride(1) == 1
    assert topk_out.dtype == torch.int32 and topk_out.stride() == (TOPK, 1)
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
        ptr(w_wk),
        ptr(w_wv),
        q_nw.data_ptr(),
        kv_nw.data_ptr(),
        w_qb.data_ptr(),
        ptr(w_iq),
        ptr(w_uk),
        ptr(ln_w),
        ptr(ln_b),
        kv_cache.data_ptr(),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.size(1),
        ptr(idx_cache),
        0 if idx_cache is None else idx_cache.stride(0),
        slots.data_ptr(),
        ptr(islots),
        gate_out.data_ptr(),
        q_out.data_ptr(),
        ptr(iq_pay),
        ptr(iq_sc),
        ptr(iq_w),
        ptr(idx_bt),
        ptr(row_ends),
        ptr(scores),
        0 if scores is None else scores.stride(0),
        ptr(topk_out),
        scratch.data_ptr(),
        layer,
        epoch.data_ptr(),
        health.data_ptr(),
        alarm,
        stream=torch.cuda.current_stream(),
    )
