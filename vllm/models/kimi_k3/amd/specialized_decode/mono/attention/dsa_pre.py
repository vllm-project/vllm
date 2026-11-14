# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""D1a ``dsa_pre``: one Kimi-K3.1 DSA layer from the AttnRes seam to the sparse
attention's and the indexer's inputs, for a spec-verify step (S rows) at TP8 +
DCP1. M1a's (``mla``) structure, with the DSA layer's formats:

One launch a layer, ``BLOCKS`` CTAs x ``THREADS`` threads:

    attn CTAs 0 .. 55   AttnRes + input_layernorm (``kda.stage_*``) -> bf16 x
                        rows, an XRDY flag a slice
    gemv CTAs 56 ..     fused_qkv_a (132 x 16 rows), the indexer's wk (8) and
                        wv (2), g_proj (96), task gi and gi + 200:
                        q_c | kv_c | k_pe -> QA, wk -> IKR (bf16 pairs); wv ->
                        ``iv_out``; the gate bf16(sigmoid(bf16 g)) ->
                        ``gate_out``
    norm (token t on CTA t)
                        q_a / kv_a RMSNorm: q_c -> plain bf16 rows, a QCR flag;
                        kv_c (normed) | k_pe -> the fp8_ds_mla record (e4m3
                        latent, pow2 per-128 scales, bf16 rope) when the slot
                        is >= 0; wk's LayerNorm (bias) -> ``ik_out``
    q_b (CTAs 64 ..)    q_nope -> QN (bf16 pairs); q_pe -> ``q_out`` [.., 512:]
    wq (a task a CTA)   the indexer query (q_c normed) -> ``iq_out``
    indexer prep (``idx``)
                        the MXFP4 key (norm's token) -> the indexer cache, its
                        top-k row reset to -1; the MXFP4 query (wq's tasks, a
                        32 group's two halves swapping amax through IQA) and
                        wv's FP32 head weights -> ``kimi_fp4_indexer``'s
                        scoring inputs
    W_UK (CTAs 208 ..)  (head, 128 latent columns) a CTA, a wave 16 of the
                        128 nope rows: bf16 W_UK_T against QN -> ``q_out``
                        [.., :512] bf16

The outputs are what the vLLM ops of ``runtime.dsa.prologue`` hand on: the
attention's bf16 query (``concat_mla_q``'s rows), the gate's sigmoid, the cache
record, and the indexer's projections (``KimiK3DSAIndexer.forward`` before
``kimi_fp4_indexer``). Elementwise roundings follow those ops (bf16 hand-offs;
the record is ``concat_and_cache_ds_mla_kernel``'s bit for bit on the same
normed row, the MXFP4 key and query ``fp4_indexer._quantize``'s on the same
bf16 rows); GEMV sums differ only in their order.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import kda
from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
    HIDDEN,
    LDS_PAD,
    MAX_S,
    NA,
    XT,
    bf4_words,
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
    ENTRY,
    GROWS,
    KVL,
    NH,
    NOPE,
    QCROW,
    QHD,
    QKV_A,
    QL,
    ROPE,
    ROWS,
    block_sum2,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.debug import (
    region_ids,
    stamp,
    stamp_begin,
    stamp_flush,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.mx import (
    clamp_fp8,
    code_ceil,
    pow2,
    rcp_pow2,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.ops import (
    CM_DEV,
    bf16_pair,
    bf_hi,
    bf_lo,
    butterfly,
    div_rn,
    fp8_pack4,
    hw_rsq,
    ld_i32,
    row_sum,
    rsrc,
    traced,
    uniform,
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

IKD = 128  # the indexer's head dim (wk rows)
IVN = 32  # the indexer's heads (wv rows)
IQN = IVN * IKD  # 4096: wq rows
TOPK = 2048  # the indexer's top-k row (int32)
IK_PAGE = 64  # the indexer cache's slots a page
IK_SCALES = 4096  # a page's scale bytes past its MXFP4 payload
FP4_DIV = 6.0  # E2M1's largest magnitude
REC = KVL + 16 + 2 * ROPE  # 656: an fp8_ds_mla record (bytes)
REC_SCALE_W = KVL // 4  # the record's scale words (one f32 a 128 tile)
REC_ROPE_W = (KVL + 16) // 4  # its rope words
FP8_DIV = 448.0  # concat_and_cache_ds_mla_kernel's kFp8ScaleDivisor (gfx950)
SCALE_MIN = 1e-4
A_TASKS = QKV_A // ROWS  # 132
IK0 = A_TASKS  # wk's 8 tasks
IV0 = IK0 + IKD // ROWS  # wv's 2
G0 = IV0 + IVN // ROWS  # g_proj's 96
GEMV_TASKS = G0 + GROWS // ROWS  # 238
GEMV_CTAS = BLOCKS - NA  # 200
QB_SUB = QHD // ROWS  # 12 tasks a head: 8 nope, 4 pe
QB_TASKS = NH * QB_SUB  # 144
QB0 = 64  # q_b's first CTA (the norm tasks on CTAs 0 .. S - 1)
IQ_TASKS = IQN // ROWS  # 256: one a CTA
UK_Q = 4  # latent quarters (128 columns) a head
UK_TASKS = NH * UK_Q  # 48
UK0 = QB0 + QB_TASKS  # W_UK's first CTA (208)
IQ0 = UK0  # wq: task i on CTA 208 + i (mod 256), first on the W_UK CTAs
UK_KW = NOPE // WAVES  # 16 nope rows a wave
# timeline points: start, slice, gate, mix, gemv0, gemv1, norm, q_b, wq, W_UK
TL_POINTS = 10

assert GEMV_TASKS <= 2 * GEMV_CTAS and QB0 >= MAX_S and UK0 + UK_TASKS == BLOCKS
assert IQ_TASKS == BLOCKS and NOPE % WAVES == 0 and QL % (4 * 384) == 0

_STREAM = fx.Stream(None)
REGIONS = ("part", "wgt", "msq", "xrdy", "qa", "qcr", "qn", "ikr", "iqa")
_SOURCES = source_digest(".")


@dataclass(frozen=True)
class DsaPreBuild:
    tokens: int  # S
    nblocks: int  # AttnRes block sources (the prefix is one more)
    delta: bool
    eps: float = 1e-5  # AttnRes norm
    out_eps: float = 1e-5  # input_layernorm
    q_eps: float = 1e-5  # q_a_layernorm
    kv_eps: float = 1e-5  # kv_a_layernorm
    k_eps: float = 1e-5  # the indexer key's LayerNorm
    timeline: bool = False
    # debug: the normed kv_c rows -> scratch ``kvn`` (the record's input)
    dump: bool = False
    # the indexer's prep (``kimi_fp4_indexer``'s key store, top-k reset, query
    # quantize and head weights times ``iw_scale``) in this launch
    idx: bool = False
    iw_scale: float = 1.0


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
        "w_ik",
        "w_iv",
        "q_nw",
        "kv_nw",
        "ik_nw",
        "ik_nb",
        "w_qb",
        "w_iq",
        "w_uk",
        "uk_sh",
        "uk_sk",
        "kv_cache",
        "kv_sb",
        "kv_ss",
        "kv_bs",
        "slots",
        "gate_out",
        "q_out",
        "iq_out",
        "ik_out",
        "iv_out",
        "ik_cache",
        "ik_cs",
        "islots",
        "topk",
        "iq_pay",
        "iq_sc",
        "iw_out",
        "scratch",
        "layer",
        "epoch",
        "health",
        "alarm",
        "tl",
    )
)


def scratch_layout(key: DsaPreBuild) -> dict:
    s, ns = key.tokens, key.nblocks + 1
    pairs = {
        "part": NA * s * ns * 2,
        "wgt": s * (ns + 1),
        "msq": NA * s,
        "xrdy": NA,
        "qa": s * QKV_A // 2,
        "qcr": s,
        "qn": s * NH * NOPE // 2,
        "ikr": s * IKD // 2,
        "iqa": s * IQ_TASKS,
    }
    lay = pair_layout(pairs.items())
    end = max(o + n for o, n in lay.values())
    # the plain bf16 rows (not pairs), 16 B aligned: x, the normed q_c, the
    # normed kv_c (``dump``)
    lay["x"] = ((end + 15) // 16 * 16, s * HIDDEN * 2)
    lay["qc"] = (lay["x"][0] + s * HIDDEN * 2, s * QL * 2)
    lay["kvn"] = (lay["qc"][0] + s * QL * 2, s * KVL * 2)
    return lay


def scratch_bytes(key: DsaPreBuild) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def max_build() -> DsaPreBuild:
    """The largest layout a served build takes (``MonoRunner``'s scratch)."""
    return DsaPreBuild(tokens=16, nblocks=8, delta=True, dump=True)


# ---------------------------------------------------------------- gemv CTAs
def proj_loads(c, task):
    """Task ``task``'s 16 weight rows (fused_qkv_a, wk, wv, then g_proj) for
    this wave; a task past the last loads the last."""
    a = c["args"]
    task = fx.min(task, GEMV_TASKS - 1)
    is_k = uniform((task >= IK0).select(fx.Int32(1), fx.Int32(0)))
    is_v = uniform((task >= IV0).select(fx.Int32(1), fx.Int32(0)))
    is_g = uniform((task >= G0).select(fx.Int32(1), fx.Int32(0)))
    w = (
        a["w_qkv"]
        + fx.Int64(is_k) * (a["w_ik"] - a["w_qkv"])
        + fx.Int64(is_v) * (a["w_iv"] - a["w_ik"])
        + fx.Int64(is_g) * (a["w_g"] - a["w_iv"])
    )
    row0 = (task - is_k * IK0 - is_v * (IV0 - IK0) - is_g * (G0 - IV0)) * ROWS
    return gemv.loads(c["lane"], c["wave"], w, row0, HIDDEN)


@traced
def stage_proj(c, task, ops, t0, n):
    """Task ``task`` against the LDS x (tokens t0 .. t0 + n): q_c | kv_c | k_pe
    rows -> QA, wk rows -> IKR, wv rows -> ``iv_out``, g rows -> the gate."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    acc = gemv.mfmas(lane, wave, c["xl"], HIDDEN + LDS_PAD, HIDDEN, n, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    r2 = tid % 8
    tt = tid // 8
    live = (tid < 8 * n) & (task < GEMV_TASKS)
    v0 = bf16r(row_sum(red, 2 * r2, fx.min(tt, n - 1)))
    v1 = bf16r(row_sum(red, 2 * r2 + 1, fx.min(tt, n - 1)))
    t = t0 + tt
    if live & (task < IK0):
        c["put_bf"](c["qa"], t * QKV_A + task * ROWS + 2 * r2, [v0, v1])
    if live & (task >= IK0) & (task < IV0):
        c["put_bf"](c["ikr"], t * IKD + (task - IK0) * ROWS + 2 * r2, [v0, v1])
    if live & (task >= IV0) & (task < G0):
        col = (task - IV0) * ROWS + 2 * r2
        bo.buffer_store(bf16_pair(v0, v1), rsrc(a["iv_out"]), (t * IVN + col) // 2)
    if live & (task >= G0):
        col = (task - G0) * ROWS + 2 * r2
        g = bf16_pair(sigmoid(v0), sigmoid(v1))
        bo.buffer_store(g, rsrc(a["gate_out"]), (t * GROWS + col) // 2)
    if const_expr(c["idx"]):  # noqa: SIM102 (compile-time, then traced)
        if live & (task >= IV0) & (task < G0):
            # the indexer's head weights: weights.float() * weight_scale
            col = (task - IV0) * ROWS + 2 * r2
            ws = fx.Float32(c["iw_scale"])
            bo.buffer_store(
                fx.Vector.from_elements([v0 * ws, v1 * ws], fx.Float32),
                rsrc(a["iw_out"]),
                t * IVN + col,
            )
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
def _fmax(a, b):
    return fx.max(a, b)


def _absf(v):
    """fabsf: the sign bit cleared."""
    return (v.bitcast(fx.Int32) & fx.Int32(0x7FFFFFFF)).bitcast(fx.Float32)


def mxfp4_scale(am):
    """A 32 group's e8m0 code and 1 / scale from its amax (``fp4_indexer.
    _quantize``: the power of two >= max(amax / 6, 1e-4))."""
    sc = fx.max(
        div_rn(am, fx.Float32(FP4_DIV), fx.Float32(1.0 / FP4_DIV)),
        fx.Float32(SCALE_MIN),
    )
    code = code_ceil(sc)
    return code, rcp_pow2(code)


# E2M1 levels 0, .5, 1, 1.5, 2, 3, 4, 6: the midpoints, ties to the even code
_E2M1_CUTS = ((0.25, False), (0.75, True), (1.25, False), (1.75, True),
              (2.5, False), (3.5, True), (5.0, False))  # fmt: skip


def mxfp4_code(x):
    """The E2M1 nibble of ``x`` (already over its group's scale)."""
    a = _absf(x)
    code = fx.Int32(0)
    for cut, ge in _E2M1_CUTS:
        hit = (a >= fx.Float32(cut)) if ge else (a > fx.Float32(cut))
        code = code + hit.select(fx.Int32(1), fx.Int32(0))
    neg = (x < fx.Float32(0.0)) & (code != 0)
    return code | neg.select(fx.Int32(8), fx.Int32(0))


@traced
def stage_ikey(c, t, y):
    """Token t's indexer key (``y``: elements 4 (tid % 32) .., LayerNormed bf16)
    -> MXFP4 in the indexer cache at its slot (``_store_key``'s page layout:
    4 x 1024 B payload, then AITER's transposed 16 x 4 scale tile), and its
    top-k row reset to -1 (``store_keys_and_reset``)."""
    tid, a = c["tid"], c["args"]
    am = fx.max(fx.max(_absf(y[0]), _absf(y[1])), fx.max(_absf(y[2]), _absf(y[3])))
    am = butterfly(am, (4, 2, 1), op=_fmax)  # a 32 group: 8 lanes
    code, inv = mxfp4_scale(am)
    q = [mxfp4_code(y[j] * inv) for j in range(4)]
    h = q[0] | (q[1] << 4) | (q[2] << 8) | (q[3] << 12)
    islot = uniform(ld_i32(a["islots"], t * 2))
    if (tid < 32) & (islot >= 0):
        g = tid // 8
        off = islot % IK_PAGE
        base = a["ik_cache"] + fx.Int64(islot // IK_PAGE) * a["ik_cs"]
        bo.buffer_store(
            fx.Int32(h).to(fx.Uint16), rsrc(base), (g * 1024 + off * 16) // 2 + tid % 8
        )
        if tid % 8 == 0:
            at = IK_SCALES + g * 64 + (off % 16) * 4 + off // 16
            bo.buffer_store(fx.Int32(code).to(fx.Uint8), rsrc(base), at)
    m1 = fx.Int32(-1)
    bo.buffer_store(
        fx.Vector.from_elements([m1, m1, m1, m1], fx.Int32),
        rsrc(a["topk"]),
        t * TOPK + tid * 4,
    )


@traced
def stage_norm(c, t):
    """Token t: q_c (threads 0 .. 383, 4 columns each) and kv_c (384 ..) RMSNorm;
    q_c -> plain bf16 row + QCR; kv_c | k_pe -> the fp8_ds_mla record; wk's
    row (threads 0 .. 31) LayerNorm -> ``ik_out``."""
    tid, a = c["tid"], c["args"]
    is_q = tid < QL // 4
    col = is_q.select(tid * 4, QL + (tid - QL // 4) * 4)
    pe = QL + KVL + fx.min(tid, ROPE // 4 - 1) * 4
    kc = (tid % 32) * 4
    got = c["poll"](
        [
            (c["qa"], (t * QKV_A + col) // 2, 2),
            (c["qa"], (t * QKV_A + pe) // 2, 2),
            (c["ikr"], (t * IKD + kc) // 2, 2),
        ]
    )
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
            bf4_words(qn),
            rsrc(c["qc"]),
            (t * QL + tid * 4) // 2,
            cache_modifier=CM_DEV,
        )
    if const_expr(c["dump"]):  # noqa: SIM102 (compile-time, then traced)
        if tid >= QL // 4:
            bo.buffer_store(
                bf4_words(kn), rsrc(c["kvn"]), (t * KVL + (tid - QL // 4) * 4) // 2
            )
    # concat_and_cache_ds_mla_kernel: a 128 tile's amax over its 32 threads
    # (lane halves of waves 6, 7), scale = pow2 >= max(amax / 448, 1e-4), the
    # e4m3 of value / scale (the division by a power of two: a multiply)
    ak = [_absf(x) for x in kn]
    am = fx.max(fx.max(ak[0], ak[1]), fx.max(ak[2], ak[3]))
    am = butterfly(am, (16, 8, 4, 2, 1), op=_fmax)
    sc = fx.max(
        div_rn(am, fx.Float32(FP8_DIV), fx.Float32(1.0 / FP8_DIV)),
        fx.Float32(SCALE_MIN),
    )
    code = code_ceil(sc)
    inv = rcp_pow2(code)
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
            j = tid - QL // 4
            word = fp8_pack4(*[clamp_fp8(kn[e] * inv) for e in range(4)])
            bo.buffer_store(word, rsrc(base), j)
            if j % 32 == 0:
                bo.buffer_store(pow2(code), rsrc(base), REC_SCALE_W + j // 32)
        if tid < ROPE // 4:
            p = got[1]
            bo.buffer_store(
                fx.Vector.from_elements([p[0], p[1]], fx.Int32),
                rsrc(base),
                REC_ROPE_W + 2 * tid,
            )
    # the indexer key: F.layer_norm (biased variance, fp32) -> bf16
    kv = got[2]
    x = [bf_lo(kv[0]), bf_hi(kv[0]), bf_lo(kv[1]), bf_hi(kv[1])]
    s1 = butterfly(x[0] + x[1] + x[2] + x[3], (16, 8, 4, 2, 1))
    mean = s1 * fx.Float32(1.0 / IKD)
    d = [x[j] - mean for j in range(4)]
    s2 = butterfly(d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3],
                   (16, 8, 4, 2, 1))  # fmt: skip
    rs = hw_rsq(s2 * fx.Float32(1.0 / IKD) + fx.Float32(c["k_eps"]))
    lw = ld_bf4(a["ik_nw"], kc)
    lb = ld_bf4(a["ik_nb"], kc)
    y = [bf16r(d[j] * rs * lw[j] + lb[j]) for j in range(4)]
    if tid < 32:
        bo.buffer_store(bf4_words(y), rsrc(a["ik_out"]), (t * IKD + kc) // 2)
    if const_expr(c["idx"]):
        stage_ikey(c, t, y)
    publish(c["put"], c["qcr"], t, fx.Int32(1), tid == 0)
    gpu.barrier()


# ---------------------------------------------------------------- q_b, wq
@traced
def wait_qc(c):
    """Every token's normed q_c row stored (its QCR flag)."""
    s, tid = c["S"], c["tid"]
    if tid < s:
        c["poll"]([(c["qcr"], tid, 1)])
    gpu.barrier()


@traced
def stage_qb(c, task):
    """q_b rows 16 task ..: head h = task / 12, its nope rows (sub < 8) -> QN or
    pe rows -> ``q_out``'s rope columns (bf16)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    h = task // QB_SUB
    sub = task % QB_SUB
    ops = gemv.loads(lane, wave, a["w_qb"], task * ROWS, QL)
    wait_qc(c)
    for t0, n in gemv.token_tiles(s):
        gemv.dev_rows_to_lds(tid, c["qc"], QL, n, c["xl"], QCROW, t0=t0)
        acc = gemv.mfmas(lane, wave, c["xl"], QCROW, QL, n, ops)
        fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
        gpu.barrier()
        r2 = tid % 8
        tt = fx.min(tid // 8, n - 1)
        v0 = row_sum(red, 2 * r2, tt)
        v1 = row_sum(red, 2 * r2 + 1, tt)
        if (tid < 8 * n) & (sub < NOPE // ROWS):
            at = ((t0 + tt) * NH + h) * NOPE + sub * ROWS + 2 * r2
            c["put_bf"](c["qn"], at, [v0, v1])
        if (tid < 8 * n) & (sub >= NOPE // ROWS):
            col = KVL + (sub - NOPE // ROWS) * ROWS + 2 * r2
            bo.buffer_store(
                bf16_pair(v0, v1),
                rsrc(a["q_out"]),
                (((t0 + tt) * NH + h) * ENTRY + col) // 2,
            )
        gpu.barrier()


@traced
def stage_iq(c, task):
    """The indexer's wq rows 16 task .. against the normed q_c -> ``iq_out``
    (bf16)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    ops = gemv.loads(lane, wave, a["w_iq"], task * ROWS, QL)
    wait_qc(c)
    for t0, n in gemv.token_tiles(s):
        gemv.dev_rows_to_lds(tid, c["qc"], QL, n, c["xl"], QCROW, t0=t0)
        acc = gemv.mfmas(lane, wave, c["xl"], QCROW, QL, n, ops)
        fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
        gpu.barrier()
        r2 = tid % 8
        tt = fx.min(tid // 8, n - 1)
        v0 = row_sum(red, 2 * r2, tt)
        v1 = row_sum(red, 2 * r2 + 1, tt)
        if tid < 8 * n:
            at = (t0 + tt) * IQN + task * ROWS + 2 * r2
            bo.buffer_store(bf16_pair(v0, v1), rsrc(a["iq_out"]), at // 2)
        if const_expr(c["idx"]):
            iq_fp4(c, task, t0 + tt, bf16r(v0), bf16r(v1), tid < 8 * n)
        gpu.barrier()


@traced
def iq_fp4(c, task, t, v0, v1, live):
    """Token t's indexer query rows 16 task .. (two a lane, bf16) -> MXFP4
    (``_quantize_query``'s layout: [S, 32 heads, 64 B], scales [S, 4 groups,
    16, 4]). A 32 group is two tasks: each publishes its half's amax (IQA) and
    takes its partner's. The half-1 task writes the group's pad scale byte
    (heads 32 .. 63 of the 64-head tile: 0)."""
    tid, a = c["tid"], c["args"]
    am = butterfly(fx.max(_absf(v0), _absf(v1)), (4, 2, 1), op=_fmax)
    half = task % 2
    if live & (tid % 8 == 0):
        c["put"](c["iqa"], t * IQ_TASKS + task, am)
    other = c["poll"]([(c["iqa"], t * IQ_TASKS + task + 1 - 2 * half, 1)])
    code, inv = mxfp4_scale(fx.max(am, other[0][0].bitcast(fx.Float32)))
    b = mxfp4_code(v0 * inv) | (mxfp4_code(v1 * inv) << 4)
    r2 = tid % 8
    h = task // 8
    g = (task % 8) // 2
    if live:
        at = t * (IQN // 2) + task * 8 + r2
        bo.buffer_store(fx.Int32(b).to(fx.Uint8), rsrc(a["iq_pay"]), at)
    if live & (r2 == 0):
        sc = (half == 0).select(code, fx.Int32(0))
        at = t * 256 + g * 64 + (h % 16) * 4 + 2 * half + h // 16
        bo.buffer_store(fx.Int32(sc).to(fx.Uint8), rsrc(a["iq_sc"]), at)


# ---------------------------------------------------------------- W_UK (bf16)
@traced
def stage_uk(c, task):
    """Head h = task / 4, latent columns 128 (task % 4) ..: q_nope [S, 128]
    (QN) -> LDS f32; wave w sums nope rows 16 w .. of W_UK_T (bf16, latent
    contiguous) a lane two columns; the waves' partials summed in order ->
    ``q_out`` bf16."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    a = c["args"]
    h = task // UK_Q
    n0 = (task % UK_Q) * 128
    ql, pl = c["ukq"], c["ukp"]
    # this lane's W_UK_T words first (they need no hand-off)
    wbase = h * a["uk_sh"] + n0 + 2 * lane
    ws = []
    for i in range_constexpr(UK_KW):
        k = wave * UK_KW + i
        ws.append(
            fx.Int32(
                bo.buffer_load(
                    rsrc(a["w_uk"]),
                    (wbase + k * a["uk_sk"]) // 2,
                    vec_width=1,
                    dtype=T.i32,
                )
            )
        )
    n_items = s * NOPE // 2
    rounds = (n_items + THREADS - 1) // THREADS
    idx = [fx.min(tid + THREADS * r, n_items - 1) for r in range(rounds)]
    got = c["poll"](
        [
            (c["qn"], ((e // (NOPE // 2)) * NH + h) * (NOPE // 2) + e % (NOPE // 2), 1)
            for e in idx
        ]
    )
    for r in range_constexpr(rounds):
        e = idx[r]
        if tid + THREADS * r < n_items:
            fx.ptr_store(bf_lo(got[r][0]), ql + 2 * e)
            fx.ptr_store(bf_hi(got[r][0]), ql + 2 * e + 1)
    gpu.barrier()
    acc = [[fx.Float32(0.0), fx.Float32(0.0)] for _ in range(s)]
    for i in range_constexpr(UK_KW):
        k = wave * UK_KW + i
        w0, w1 = bf_lo(ws[i]), bf_hi(ws[i])
        for t in range_constexpr(s):
            qv = fx.ptr_load(ql + (t * NOPE + k))
            acc[t][0] = acc[t][0] + qv * w0
            acc[t][1] = acc[t][1] + qv * w1
    for t in range_constexpr(s):
        fx.ptr_store(
            fx.Vector.from_elements([acc[t][0], acc[t][1]], fx.Float32),
            pl + ((wave * s + t) * 128 + 2 * lane),
        )
    gpu.barrier()
    n_out = s * 64
    for r in range_constexpr((n_out + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * r, n_out - 1)
        t = e // 64
        p = e % 64
        o0 = fx.Float32(0.0)
        o1 = fx.Float32(0.0)
        for wv in range_constexpr(WAVES):
            at = (wv * s + t) * 128 + 2 * p
            o0 = o0 + fx.ptr_load(pl + at)
            o1 = o1 + fx.ptr_load(pl + at + 1)
        if tid + THREADS * r < n_out:
            bo.buffer_store(
                bf16_pair(o0, o1),
                rsrc(a["q_out"]),
                ((t * NH + h) * ENTRY + n0 + 2 * p) // 2,
            )
    gpu.barrier()


# ---------------------------------------------------------------- kernel
_BUILDS: dict = {}


def build(key: DsaPreBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    s = key.tokens
    ns = key.nblocks + 1
    lay = scratch_layout(key)
    xrow = HIDDEN + LDS_PAD
    assert 1 <= s <= 16 and key.nblocks >= 1

    @fx.struct
    class AttnLds:
        vals: fx.Array[fx.Float32, vals_words(s, ns), 16]

    @fx.struct
    class GemvLds:
        xl: fx.Array[fx.Int32, max(min(s, XT) * xrow, 16 * QCROW) // 2, 16]

    @fx.struct
    class UkLds:
        q: fx.Array[fx.Float32, s * NOPE, 16]
        p: fx.Array[fx.Float32, WAVES * s * 128, 16]

    @fx.union
    class RoleLds:
        attn: AttnLds
        gemv: GemvLds
        uk: UkLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    keyed = key_tuple(key, _SOURCES)
    name = (
        f"k3_mono_dsa_pre_s{s}_nb{key.nblocks}_d{int(key.delta)}"
        f"_t{int(key.timeline)}{'_dump' if key.dump else ''}"
        f"{'_idx' if key.idx else ''}"
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def dsa_pre(
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
        w_ik: Int64,
        w_iv: Int64,
        q_nw: Int64,
        kv_nw: Int64,
        ik_nw: Int64,
        ik_nb: Int64,
        w_qb: Int64,
        w_iq: Int64,
        w_uk: Int64,
        uk_sh: Int32,
        uk_sk: Int32,
        kv_cache: Int64,
        kv_sb: Int64,
        kv_ss: Int32,
        kv_bs: Int32,
        slots: Int64,
        gate_out: Int64,
        q_out: Int64,
        iq_out: Int64,
        ik_out: Int64,
        iv_out: Int64,
        ik_cache: Int64,
        ik_cs: Int64,
        islots: Int64,
        topk: Int64,
        iq_pay: Int64,
        iq_sc: Int64,
        iw_out: Int64,
        scratch: Int64,
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
        al, gl, ul = role.attn.peek(), role.gemv.peek(), role.uk.peek()
        mb = Mailbox(step_tag(epoch, layer) - 1, health, alarm, region_ids(REGIONS))
        c = {
            "S": s,
            "NS": ns,
            "NB": key.nblocks,
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
            "dump": key.dump,
            "idx": key.idx,
            "iw_scale": key.iw_scale,
            "vals_all": vals_all(s, ns),
            "put": mb.put,
            "put_bf": mb.put_bf,
            "put_words": mb.put_words,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "vals": al.vals.ptr,
            "xl": gl.xl.ptr,
            "ukq": ul.q.ptr,
            "ukp": ul.p.ptr,
            "xbuf": scratch + fx.Int64(lay["x"][0]),
            "qc": scratch + fx.Int64(lay["qc"][0]),
            "kvn": scratch + fx.Int64(lay["kvn"][0]),
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
                "w_ik": w_ik,
                "w_iv": w_iv,
                "q_nw": q_nw,
                "kv_nw": kv_nw,
                "ik_nw": ik_nw,
                "ik_nb": ik_nb,
                "w_qb": w_qb,
                "w_iq": w_iq,
                "w_uk": w_uk,
                "uk_sh": uk_sh,
                "uk_sk": uk_sk,
                "kv_cache": kv_cache,
                "kv_sb": kv_sb,
                "kv_ss": kv_ss,
                "kv_bs": kv_bs,
                "slots": slots,
                "gate_out": gate_out,
                "q_out": q_out,
                "iq_out": iq_out,
                "ik_out": ik_out,
                "iv_out": iv_out,
                "ik_cache": ik_cache,
                "ik_cs": ik_cs,
                "islots": islots,
                "topk": topk,
                "iq_pay": iq_pay,
                "iq_sc": iq_sc,
                "iw_out": iw_out,
            },
        }
        for region in REGIONS:
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
        for task in range(first_task(bid, IQ0), IQ_TASKS, BLOCKS):
            stage_iq(c, task)
        stamp(on, tls, tid, 8)
        for task in range(first_task(bid, UK0), UK_TASKS, BLOCKS):
            stage_uk(c, task)
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
        w_ik: Int64,
        w_iv: Int64,
        q_nw: Int64,
        kv_nw: Int64,
        ik_nw: Int64,
        ik_nb: Int64,
        w_qb: Int64,
        w_iq: Int64,
        w_uk: Int64,
        uk_sh: Int32,
        uk_sk: Int32,
        kv_cache: Int64,
        kv_sb: Int64,
        kv_ss: Int32,
        kv_bs: Int32,
        slots: Int64,
        gate_out: Int64,
        q_out: Int64,
        iq_out: Int64,
        ik_out: Int64,
        iv_out: Int64,
        ik_cache: Int64,
        ik_cs: Int64,
        islots: Int64,
        topk: Int64,
        iq_pay: Int64,
        iq_sc: Int64,
        iw_out: Int64,
        scratch: Int64,
        layer: Int32,
        epoch: Int64,
        health: Int64,
        alarm: Int64,
        tl: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        dsa_pre(
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
            w_ik,
            w_iv,
            q_nw,
            kv_nw,
            ik_nw,
            ik_nb,
            w_qb,
            w_iq,
            w_uk,
            uk_sh,
            uk_sk,
            kv_cache,
            kv_sb,
            kv_ss,
            kv_bs,
            slots,
            gate_out,
            q_out,
            iq_out,
            ik_out,
            iv_out,
            ik_cache,
            ik_cs,
            islots,
            topk,
            iq_pay,
            iq_sc,
            iw_out,
            scratch,
            layer,
            epoch,
            health,
            alarm,
            tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(dsa_pre, launch)
    _BUILDS[key] = launch
    return launch


def ptr(t):
    return 0 if t is None else t.data_ptr()


def idx_reasons(ik_cache, islots, topk, s=None) -> list[str]:
    """Why D1a cannot run the indexer's prep into these (empty: it can)."""
    bad = []
    if (
        ik_cache is None
        or ik_cache.dtype != torch.uint8
        or ik_cache.dim() != 3
        or tuple(ik_cache.shape[1:]) != (IK_PAGE, 68)
        or ik_cache.stride()[1:] != (68, 1)
        or ik_cache.stride(0) < IK_PAGE * 68
        or ik_cache.data_ptr() % 4
    ):
        bad.append(f"an indexer cache not MXFP4 [pages, 64, 68]: {_desc(ik_cache)}")
    if (
        islots is None
        or islots.dtype != torch.int64
        or islots.stride(-1) != 1
        or (s is not None and islots.numel() < s)
    ):
        bad.append(f"indexer slots not dense int64, one a token: {_desc(islots)}")
    if (
        topk is None
        or topk.dtype != torch.int32
        or topk.dim() != 2
        or topk.stride() != (TOPK, 1)
        or (s is not None and topk.size(0) < s)
    ):
        bad.append(f"top-k rows not dense int32 [.., 2048]: {_desc(topk)}")
    return bad


def _desc(t) -> str:
    return "(none)" if t is None else f"{tuple(t.shape)} {t.dtype} {t.stride()}"


def check_inputs(**t) -> list[str]:
    """Why these tensors cannot feed ``dsa_pre`` (empty: they can)."""
    bad = []
    for name in ("w_qkv", "w_g", "w_ik", "w_iv", "w_qb", "w_iq"):
        w = t[name]
        if w.dtype != torch.bfloat16 or not w.is_contiguous():
            bad.append(f"{name} {w.dtype} not dense bf16")
    shapes = dict(
        w_qkv=(QKV_A, HIDDEN), w_g=(GROWS, HIDDEN), w_ik=(IKD, HIDDEN),
        w_iv=(IVN, HIDDEN), w_qb=(NH * QHD, QL), w_iq=(IQN, QL),
    )  # fmt: skip
    for name, shape in shapes.items():
        if tuple(t[name].shape) != shape:
            bad.append(f"{name} {tuple(t[name].shape)} != {shape}")
    uk = t["w_uk"]
    if (
        uk.dtype != torch.bfloat16
        or tuple(uk.shape) != (NH, NOPE, KVL)
        or uk.stride(-1) != 1
        or uk.stride(0) % 2
        or uk.stride(1) % 2
    ):
        bad.append(f"W_UK_T {tuple(uk.shape)} {uk.dtype} {uk.stride()}")
    for name in ("ik_nw", "ik_nb"):
        if t[name].dtype != torch.bfloat16 or t[name].numel() != IKD:
            bad.append(f"{name} {t[name].dtype} {tuple(t[name].shape)}")
    kv = t["kv_cache"]
    if kv.element_size() != 1 or kv.size(-1) != REC or kv.stride(-1) != 1:
        bad.append(f"a KV cache {tuple(kv.shape)} {kv.dtype} not fp8_ds_mla rows")
    elif kv.stride(1) % 4 or kv.stride(0) % 4 or kv.data_ptr() % 4:
        bad.append(f"KV cache strides {kv.stride()} not word aligned")
    return bad


def dsa_pre(
    key: DsaPreBuild,
    *,
    prefix,
    delta,
    blocks,
    ares_nw,
    ares_qk,
    in_nw,
    w_qkv,
    w_g,
    w_ik,
    w_iv,
    q_nw,
    kv_nw,
    ik_nw,
    ik_nb,
    w_qb,
    w_iq,
    w_uk,
    kv_cache,
    slots,
    gate_out,
    q_out,
    iq_out,
    ik_out,
    iv_out,
    scratch,
    layer,
    epoch,
    health,
    alarm,
    tl=None,
    ik_cache=None,
    islots=None,
    topk=None,
    iq_pay=None,
    iq_sc=None,
    iw_out=None,
):
    """Host launcher. ``w_uk`` vLLM's ``W_UK_T`` [12, 128, 512] bf16 (latent
    contiguous); ``kv_cache`` [blocks, block_size, 656] one-byte records;
    ``slots`` [S] int64 (-1: padding); ``q_out`` [S, 12, 576] bf16; ``gate_out``
    [S, 1536]; ``iq_out`` [S, 4096], ``ik_out`` [S, 128], ``iv_out`` [S, 32]
    bf16, all dense; ``layer`` a slot below ``kda.TAG_SLOTS``.

    ``key.idx``: ``ik_cache`` the indexer's uint8 [pages, 64, 68] sections
    (``fp4_indexer.cache_views``), ``islots`` its [S] int64 slots, ``topk`` [S,
    2048] int32 rows, ``iq_pay`` [S, 32, 64] / ``iq_sc`` [S, 1, 4, 16, 4]
    uint8 and ``iw_out`` [S, 32] f32 (the query's MXFP4 and head weights)."""
    bad = check_inputs(
        w_qkv=w_qkv, w_g=w_g, w_ik=w_ik, w_iv=w_iv, w_qb=w_qb, w_iq=w_iq,
        w_uk=w_uk, ik_nw=ik_nw, ik_nb=ik_nb, kv_cache=kv_cache,
    )  # fmt: skip
    assert not bad, bad
    s = key.tokens
    assert slots.dtype == torch.int64 and slots.size(0) == s
    for o, shape in (
        (q_out, (s, NH, ENTRY)),
        (gate_out, (s, GROWS)),
        (iq_out, (s, IQN)),
        (ik_out, (s, IKD)),
        (iv_out, (s, IVN)),
    ):
        assert o.dtype == torch.bfloat16 and o.is_contiguous(), o.dtype
        assert tuple(o.shape) == shape, (tuple(o.shape), shape)
    assert scratch.numel() >= scratch_bytes(key)
    if key.idx:
        assert idx_reasons(ik_cache, islots, topk, s) == []
        for o, shape, dt in (
            (iq_pay, (s, IVN, IKD // 2), torch.uint8),
            (iq_sc, (s, 1, 4, 16, 4), torch.uint8),
            (iw_out, (s, IVN), torch.float32),
        ):
            assert o.dtype == dt and o.is_contiguous(), o.dtype
            assert tuple(o.shape) == shape, (tuple(o.shape), shape)
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
        w_ik.data_ptr(),
        w_iv.data_ptr(),
        q_nw.data_ptr(),
        kv_nw.data_ptr(),
        ik_nw.data_ptr(),
        ik_nb.data_ptr(),
        w_qb.data_ptr(),
        w_iq.data_ptr(),
        w_uk.data_ptr(),
        w_uk.stride(0),
        w_uk.stride(1),
        kv_cache.data_ptr(),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.size(1),
        slots.data_ptr(),
        gate_out.data_ptr(),
        q_out.data_ptr(),
        iq_out.data_ptr(),
        ik_out.data_ptr(),
        iv_out.data_ptr(),
        ptr(ik_cache),
        0 if ik_cache is None else ik_cache.stride(0),
        ptr(islots),
        ptr(topk),
        ptr(iq_pay),
        ptr(iq_sc),
        ptr(iw_out),
        scratch.data_ptr(),
        layer,
        epoch.data_ptr(),
        health.data_ptr(),
        alarm,
        ptr(tl),
        stream=torch.cuda.current_stream(),
    )
