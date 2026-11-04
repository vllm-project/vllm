# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 DSpark MLA frontend. S7/TP8, one resident CTA per CU."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
    bf4_words,
    bf16r,
    ld_bf4,
    ld_f32,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla import block_sum2
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.mx import clamp_fp8
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.ops import (
    CM_DEV,
    bf_hi,
    bf_lo,
    butterfly,
    fp8_pack4,
    hw_rsq,
    ld_i32,
    mfma_bf16,
    row_sum,
    rsrc,
    traced,
    uniform,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import pair_layout
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.sync import (
    publish,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.stages import gemv

from .attention import fp8_unpack4

S = 7
H = 7168
QL = 1536
KVL = 512
NH = 8
NOPE = 128
PE = 64
QA = 2112
QB = 1536
ENTRY = 576
THREADS = 512
BLOCKS = 256
REGIONS = ("qa", "qcr", "qn")
LAY = pair_layout([("qa", S * QA // 2), ("qcr", S), ("qn", S * NH * NOPE // 2)])
END = max(o + n for o, n in LAY.values())
QCOFF = (END + 15) // 16 * 16
NBYTES = QCOFF + S * QL * 2
STREAM = fx.Stream(None)
BUILDS: dict = {}
SOURCE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Key:
    eps: float = 1e-5
    padded_heads: int = 8


DEFAULT_KEY = Key()


def fp8(v, scale):
    return fp8_pack4(*[clamp_fp8(x * scale) for x in v])


# xb (draft run 7): gemv.dev_rows_to_lds guards each unrolled round's LDS store
# with a runtime ``if`` (a scheduling region's end: the next round's load is
# not hoisted above it), so at K = 7168 a thread's 13 device-scope loads went
# one round trip at a time. Here every load is issued first; the stores follow
# unguarded: a thread past the end of the last round loads the last 16 B (its
# index clamped, as gemv's) and stores that same value into that same slot.
XB = True


@traced
def dev_rows_to_lds_b(tid, src, k, s, xl, xrow):
    """`gemv.dev_rows_to_lds(tid, src, k, s, xl, xrow)`: the same words into
    the same LDS slots, every load in flight before the first store."""
    words = s * k // 8
    r = rsrc(src)
    got = []
    for i in range_constexpr((words + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, words - 1)
        t = e // (k // 8)
        kk = e % (k // 8) * 8
        v = fx.Vector(
            bo.buffer_load(
                r, (t * k + kk) // 2, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV
            )
        )
        got.append((t, kk, v))
    for t, kk, v in got:
        fx.ptr_store(v, xl + (t * xrow + kk) // 2)
    gpu.barrier()


def x_rows_to_lds(tid, src, k, s, xl, xrow):
    if XB:
        dev_rows_to_lds_b(tid, src, k, s, xl, xrow)
    else:
        gemv.dev_rows_to_lds(tid, src, k, s, xl, xrow)


def rotate(c, t, p, v):
    a = c["a"]
    pos = ld_i32(a["positions"], t * 2)
    cs = a["cos_sin"] + fx.Int64(pos) * 256
    out = []
    for i in range(2):
        co = ld_f32(cs, p // 2 + i)
        si = ld_f32(cs, p // 2 + i + 32)
        out.extend(
            [v[2 * i] * co - v[2 * i + 1] * si, v[2 * i] * si + v[2 * i + 1] * co]
        )
    return out


def put_query(c, t, h, w, v):
    a = c["a"]
    packed = fp8(v, ld_f32(a["qsi"], 0))
    exact = fp8_unpack4(packed)
    words = fx.Vector.from_elements(
        [bf4_words(exact)[0], bf4_words(exact)[1]], fx.Int32
    )
    bo.buffer_store(words, rsrc(a["out"]), ((t * NH + h) * ENTRY) // 2 + 2 * w)


@traced
def proj(c, task):
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    ops = gemv.loads(lane, wave, c["a"]["wqkv"], task * 16, H)
    if tid < c.get("nrdy", S):  # the input norm's ready flags (layer.NN)
        c["poll"]([(c["xrdy"], tid, 1)])
    gpu.barrier()
    x_rows_to_lds(tid, c["a"]["x"], H, S, c["xl"], H + 8)
    acc = gemv.mfmas(lane, wave, c["xl"], H + 8, H, S, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    row = tid % 8
    t = fx.min(tid // 8, S - 1)
    v = [row_sum(red, 2 * row + j, t) for j in range(2)]
    if tid < 8 * S:
        c["put_bf"](c["qa"], t * QA + task * 16 + 2 * row, v)
    gpu.barrier()


@traced
def norm(c, t):
    tid, a = c["tid"], c["a"]
    isq = tid < QL // 4
    col = isq.select(tid * 4, QL + (tid - QL // 4) * 4)
    pecol = QL + KVL + fx.min(tid, 15) * 4
    got = c["poll"](
        [(c["qa"], (t * QA + col) // 2, 2), (c["qa"], (t * QA + pecol) // 2, 2)]
    )
    w = got[0]
    v = [bf_lo(w[0]), bf_hi(w[0]), bf_lo(w[1]), bf_hi(w[1])]
    sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
    qs, kvs = block_sum2(c, sq)
    rq = hw_rsq(qs * fx.Float32(1 / QL) + fx.Float32(c["eps"]))
    rk = hw_rsq(kvs * fx.Float32(1 / KVL) + fx.Float32(c["eps"]))
    nw = ld_bf4(a["qnw"], fx.min(tid, QL // 4 - 1) * 4)
    kw = ld_bf4(a["kvnw"], fx.max(tid - QL // 4, 0) * 4)
    qn = [bf16r(v[j] * rq * nw[j]) for j in range(4)]
    kn = [bf16r(v[j] * rk * kw[j]) for j in range(4)]
    if isq:
        bo.buffer_store(
            bf4_words(qn), rsrc(c["qc"]), (t * QL + tid * 4) // 2, cache_modifier=CM_DEV
        )
    # Drain the payload before publishing the flag, using the established helper.
    publish(c["put"], c["qcr"], t, fx.Int32(1), tid == 0)
    slot = uniform(ld_i32(a["slots"], t * 2))
    if slot >= 0:
        base = a["cache"] + fx.Int64(slot) * ENTRY
        scale = ld_f32(a["ksi"], 0)
        if tid >= QL // 4:
            bo.buffer_store(fp8(kn, scale), rsrc(base), tid - QL // 4)
        if tid < 16:
            p = got[1]
            pv = [bf_lo(p[0]), bf_hi(p[0]), bf_lo(p[1]), bf_hi(p[1])]
            rv = rotate(c, t, tid * 4, pv)
            bo.buffer_store(fp8(rv, scale), rsrc(base), KVL // 4 + tid)
    gpu.barrier()


@traced
def qb(c, task):
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    a = c["a"]
    ops = gemv.loads(lane, wave, a["wqb"], task * 16, QL)
    # q_c's RMSNorm here, from qa's tagged pairs: ``norm``'s math and sum order
    # (thread tid < 384 columns 4 tid .., the butterfly, waves 0..5 in order),
    # so no QCR hand-off and no qc re-read. First wait on one token (a cheap
    # spin), then poll the 7 rows.
    col = fx.min(tid, QL // 4 - 1) * 4
    c["poll"]([(c["qa"], ((S - 1) * QA + col) // 2, 2)])
    got = c["poll"]([(c["qa"], (t * QA + col) // 2, 2) for t in range(S)], batch=S)
    nw = ld_bf4(a["qnw"], col)
    vs = []
    for t in range_constexpr(S):
        w = got[t]
        v = [bf_lo(w[0]), bf_hi(w[0]), bf_lo(w[1]), bf_hi(w[1])]
        sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
        sq = butterfly(sq, (32, 16, 8, 4, 2, 1))
        if lane == 0:
            fx.ptr_store(sq, red + 8 * t + wave)
        vs.append(v)
    gpu.barrier()
    for t in range_constexpr(S):
        qs = fx.ptr_load(red + 8 * t)
        for w in range_constexpr(1, 6):
            qs = qs + fx.ptr_load(red + 8 * t + w)
        rq = hw_rsq(qs * fx.Float32(1 / QL) + fx.Float32(c["eps"]))
        qn = [bf16r(vs[t][j] * rq * nw[j]) for j in range(4)]
        if tid < QL // 4:
            fx.ptr_store(bf4_words(qn), c["xl"] + (t * (QL + 8) + col) // 2)
    gpu.barrier()
    acc = gemv.mfmas(lane, wave, c["xl"], QL + 8, QL, S, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    head = task // 12
    sub = task % 12
    r2 = tid % 8
    t2 = fx.min(tid // 8, S - 1)
    if (tid < 8 * S) & (sub < 8):
        v = [row_sum(red, 2 * r2 + j, t2) for j in range(2)]
        c["put_bf"](c["qn"], (t2 * NH + head) * NOPE + sub * 16 + 2 * r2, v)
    r4 = tid % 4
    t4 = fx.min(tid // 4, S - 1)
    if (tid < 4 * S) & (sub >= 8):
        v = [bf16r(row_sum(red, 4 * r4 + j, t4)) for j in range(4)]
        p = (sub - 8) * 16 + 4 * r4
        put_query(c, t4, head, KVL // 4 + p // 4, rotate(c, t4, p, v))
    gpu.barrier()


@traced
def uk(c, task):
    lane = c["lane"]
    a = c["a"]
    ng = task % 32
    head = task // 32
    row = ng * 16 + lane % 16
    g = lane // 16
    t = lane % 16
    tc = fx.min(t, S - 1)
    # BF16 [head,latent,nope] weights. Four K32 operands, prefetched before QN.
    ws = []
    for q in range_constexpr(4):
        off = (head * KVL * NOPE + row * NOPE + q * 32 + g * 8) // 2
        ws.append(
            fx.Vector(bo.buffer_load(rsrc(a["wuk"]), off, vec_width=4, dtype=T.i32))
        )
    base = ((tc * NH + head) * NOPE + g * 8) // 2
    got = c["poll"](
        [(c["qn"], base + q * 16 + j * 2, 2) for q in range(4) for j in range(2)]
    )
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for q in range_constexpr(4):
        xv = fx.Vector.from_elements(
            [got[2 * q + j // 2][j % 2] for j in range(4)], fx.Int32
        )
        acc = mfma_bf16(ws[q].bitcast(fx.BFloat16), xv.bitcast(fx.BFloat16), acc)
    if t < S:
        put_query(c, t, head, ng * 4 + g, [bf16r(acc[j]) for j in range(4)])
