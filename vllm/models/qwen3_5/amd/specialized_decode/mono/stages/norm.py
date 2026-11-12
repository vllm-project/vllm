# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The residual add + RMSNorm (Gemma, 1 + w) around a block, without an x
tile in LDS: at 8192 columns a 16-row bf16 x is 262 KB, more than a CU's LDS.

The norm tasks (CTAs 0 .. NRED - 1, 128 columns each) add the block input
and the residual -> ``res_out`` and put their columns' sums of squares (MSQ).
Every consumer then builds what it reads of x = bf16(z rstd (1 + w)) from
global memory: a bf16 K slice (``x_rows_to_lds``: the router, in_proj_ba),
the rows' MXFP4 codes (``x_fp4_to_lds``: in_proj_qkvz and the routed
experts, as their GEMMs quantize x), or a dot product (``stage_sgate``). At
the wider widths the norm tasks quantize their columns once instead
(``stage_xq``) and the MXFP4 consumers copy the image (``xq_to_lds``).
A source ``src`` is (a, cache modifier, b or None): z = a + b, fp32 of the
bf16 words, as ``fused_add_rms_norm`` reads the unrounded sum."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from vllm.models.qwen3_5.amd.specialized_decode.mono import layout as L
from vllm.models.qwen3_5.amd.specialized_decode.mono.common.ops import (
    CM_DEV,
    bf16_pair,
    bf16_round,
    bf_hi,
    bf_lo,
    hw_rsq,
    ld_i32,
    rsrc,
    traced,
    wave_sum,
    xred,
)
from vllm.models.qwen3_5.amd.specialized_decode.mono.common.plan import (
    THREADS,
    WAVES,
)
from vllm.models.qwen3_5.amd.specialized_decode.mono.common.ranks import (
    sum_partials,
)
from vllm.models.qwen3_5.amd.specialized_decode.mono.common.sync import publish
from vllm.models.qwen3_5.amd.specialized_decode.mono.stages import gemv
from vllm.models.qwen3_5.amd.specialized_decode.mono.stages.moe import (
    f32_of,
    mx_scale,
    mx_scale_round,
    pk_fp4_bf16,
    pow2,
    sigmoid,
    xq_row,
    xq_scales,
)

H = L.HIDDEN
COLS = 128  # columns a norm / final task
NRED = H // COLS  # 64
LOAD_GROUP = 4  # x_fp4_to_lds / local_rstd rounds whose loads go out together
XQ_GROUP = 8  # xq_to_lds rounds whose loads go out together


def token_passes(s):
    """Passes of a CTA over s tokens at a wave a token (64 threads, 128
    columns)."""
    return -(-s // WAVES)


def _pair(z0, z1):
    return bf16_pair(z0, z1).bitcast(fx.Int32)


@traced
def stage_front(c, j, src, msq=True):
    """Columns 128 j ..: a + b (fp32; bf16 -> ``res_out``) and, with ``msq``, the
    columns' sums of squares -> MSQ."""
    s, tid, lane = c["S"], c["tid"], c["lane"]
    a, cm, b = src
    for p in range_constexpr(token_passes(s)):
        tt = tid // 64 + WAVES * p
        mine = tt < s
        t = fx.min(tt, s - 1)
        at = (t * H + j * COLS + 2 * (tid % 64)) // 2
        aw = fx.Int32(
            bo.buffer_load(rsrc(a), at, vec_width=1, dtype=T.i32, cache_modifier=cm)
        )
        z0, z1 = bf_lo(aw), bf_hi(aw)
        if const_expr(b is not None):
            bw = ld_i32(b, at)
            z0, z1 = z0 + bf_lo(bw), z1 + bf_hi(bw)
        if mine:
            bo.buffer_store(_pair(z0, z1), rsrc(c["args"]["res_out"]), at)
        if const_expr(msq):
            sq = wave_sum(z0 * z0 + z1 * z1)
            if mine & (lane == 0):
                c["put"](c["msq"], j * s + t, sq)


@traced
def stage_norm(c, j):
    """Columns 128 j ..: the AR sum (rank order, fp32 -> bf16) + residual (fp32;
    bf16 -> ``res_out``); the bf16 AR sums -> ``xbuf`` and the columns' sums of
    squares -> MSQ, the store drained first."""
    s, tid, lane = c["S"], c["tid"], c["lane"]
    a = c["args"]
    for p in range_constexpr(token_passes(s)):
        tt = tid // 64 + WAVES * p
        mine = tt < s
        t = fx.min(tt, s - 1)
        col = j * COLS + 2 * (tid % 64)
        lo, hi = sum_partials(
            c["poll"],
            c["peer"]["ar_own"],
            lambda src, t=t, col=col: ((src * s + t) * H + col) // 2,
            c["TP"],
        )
        at = (t * H + col) // 2
        rw = ld_i32(a["residual"], at)
        z0 = bf16_round(lo) + bf_lo(rw)
        z1 = bf16_round(hi) + bf_hi(rw)
        if mine:
            bo.buffer_store(_pair(z0, z1), rsrc(a["res_out"]), at)
            bo.buffer_store(_pair(lo, hi), rsrc(c["xbuf"]), at, cache_modifier=CM_DEV)
        sq = wave_sum(z0 * z0 + z1 * z1)
        rocdl.s_waitcnt(vmcnt=0)
        if mine & (lane == 0):
            c["put"](c["msq"], j * s + t, sq)


@traced
def load_rstd(c):
    """Every norm task's sums of squares -> each token's rstd -> LDS ``rs``
    (fp32 sum in task order). A wave waits on the last token's first, so the
    full polls don't spin."""
    s, tid, lane, red = c["S"], c["tid"], c["lane"], c["red"]
    if tid < NRED:
        c["poll"]([(c["msq"], tid * s + s - 1, 1)])
    gpu.barrier()
    for p in range_constexpr(token_passes(s)):
        t = tid // 64 + WAVES * p
        if t < s:
            v = c["poll"]([(c["msq"], lane * s + t, 1)])[0][0]
            fx.ptr_store(f32_of(v), red + (t * 64 + lane))
    gpu.barrier()
    if tid < s:
        tot = fx.ptr_load(red + tid * 64)
        for q in range_constexpr(1, NRED):
            tot = tot + fx.ptr_load(red + (tid * 64 + q))
        r = hw_rsq(tot * fx.Float32(1.0 / H) + fx.Float32(c["eps"]))
        fx.ptr_store(r, c["rs"] + tid)
    gpu.barrier()


def _z_words(src, t, kk):
    """Token t's a, b words of columns kk .. kk + 7 (4 packed bf16 each)."""
    a, cm, b = src
    g = (t * H + kk) // 2
    av = fx.Vector(
        bo.buffer_load(rsrc(a), g, vec_width=4, dtype=T.i32, cache_modifier=cm)
    )
    bv = None
    if const_expr(b is not None):
        bv = fx.Vector(bo.buffer_load(rsrc(b), g, vec_width=4, dtype=T.i32))
    return av, bv


@traced
def local_rstd(c, src):
    """Every token's rstd from all its columns on this CTA, no hand-off (fp32
    sums in a fixed order, the same on every CTA) -> LDS ``rs``: at a few rows
    cheaper than waiting on the norm tasks' MSQ."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    per = H // 8 // THREADS  # rounds a token
    n = s * per
    for g0 in range_constexpr(0, n, LOAD_GROUP):
        idx = range(g0, min(g0 + LOAD_GROUP, n))
        got = [_z_words(src, i // per, (i % per * THREADS + tid) * 8) for i in idx]
        for i, (av, bv) in zip(idx, got):
            sq = fx.Float32(0.0)
            for d in range_constexpr(4):
                z0, z1 = bf_lo(av[d]), bf_hi(av[d])
                if const_expr(bv is not None):
                    z0, z1 = z0 + bf_lo(bv[d]), z1 + bf_hi(bv[d])
                sq = sq + z0 * z0 + z1 * z1
            sq = wave_sum(sq)
            if lane == 0:
                fx.ptr_store(sq, red + (i * WAVES + wave))
    gpu.barrier()
    if tid < s:
        tot = fx.ptr_load(red + tid * per * WAVES)
        for q in range_constexpr(1, per * WAVES):
            tot = tot + fx.ptr_load(red + (tid * per * WAVES + q))
        r = hw_rsq(tot * fx.Float32(1.0 / H) + fx.Float32(c["eps"]))
        fx.ptr_store(r, c["rs"] + tid)
    gpu.barrier()


def _x_words(c, src, t, kk):
    """Token t's x columns kk .. kk + 7 (4 packed bf16 words), from ``src``."""
    av, bv = _z_words(src, t, kk)
    wv = fx.Vector(
        bo.buffer_load(rsrc(c["args"]["ln_w"]), kk // 2, vec_width=4, dtype=T.i32)
    )
    return av, bv, wv


def _x_norm(c, t, loaded):
    av, bv, wv = loaded
    r = fx.ptr_load(c["rs"] + t)
    ys = []
    for d in range_constexpr(4):
        z0, z1 = bf_lo(av[d]), bf_hi(av[d])
        if const_expr(bv is not None):
            z0, z1 = z0 + bf_lo(bv[d]), z1 + bf_hi(bv[d])
        w0 = fx.Float32(1.0) + bf_lo(wv[d])
        w1 = fx.Float32(1.0) + bf_hi(wv[d])
        ys.append(_pair(z0 * r * w0, z1 * r * w1))
    return ys


@traced
def x_rows_to_lds(c, src, klen, k0):
    """Every token's x columns k0 .. k0 + klen (bf16) -> LDS ``xl`` rows of
    ``klen`` + pad, 8 columns a lane."""
    s, tid = c["S"], c["tid"]
    xrow = klen + gemv.LDS_PAD
    words = s * klen // 8
    for i in range_constexpr((words + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, words - 1)
        t = e // (klen // 8)
        kk = e % (klen // 8) * 8
        ys = _x_norm(c, t, _x_words(c, src, t, k0 + kk))
        if tid + THREADS * i < words:
            fx.ptr_store(
                fx.Vector.from_elements(ys, fx.Int32), c["xl"] + (t * xrow + kk) // 2
            )
    gpu.barrier()


@traced
def x_fp4_to_lds(c, src, rceil=True):
    """Every token's x (bf16) through MXFP4 -> LDS ``xl``, as ``quant_x_fp4``
    lays it out (token t's block b at byte t ``xq_row`` + 16 b, its e8m0 code
    at dword ``xq_scales`` + t (H / 32) + b), 8 columns a lane, a block on 4
    adjacent lanes. ``rceil``: the RCEIL scale of the stock decode MoE's and
    the ASM FP4 GEMM's activation quant, else the Triton FP4 GEMM's
    (``mx_scale_round``). ``LOAD_GROUP`` rounds' loads go out together."""
    s, tid = c["S"], c["tid"]
    kq = H // 8
    nq = s * kq
    rounds = (nq + THREADS - 1) // THREADS
    for g0 in range_constexpr(0, rounds, LOAD_GROUP):
        idx = range(g0, min(g0 + LOAD_GROUP, rounds))
        es = [fx.min(tid + THREADS * i, nq - 1) for i in idx]
        got = [_x_words(c, src, e // kq, e % kq * 8) for e in es]
        for i, e, loaded in zip(idx, es, got):
            ys = _x_norm(c, e // kq, loaded)
            amax = fx.Float32(0.0)
            for d in range_constexpr(4):
                amax = fx.max(amax, fx.Float32(fmath.absf(bf_lo(ys[d]))))
                amax = fx.max(amax, fx.Float32(fmath.absf(bf_hi(ys[d]))))
            for off in range_constexpr(2):
                amax = xred(amax, 1 << off, fx.max)
            code = mx_scale(amax) if rceil else mx_scale_round(amax)
            w = fx.Int32(0)
            for d in range_constexpr(4):
                w = pk_fp4_bf16(w, ys[d], pow2(code), d)
            if tid + THREADS * i < nq:
                fx.ptr_store(w, c["xl"] + ((e // kq) * xq_row(H) // 4 + e % kq))
                if e % 4 == 0:
                    fx.ptr_store(code, c["xl"] + (xq_scales(s, H) + e // 4))
    gpu.barrier()


def xq_dwords(s):
    """Dwords of the x rows' MXFP4 image (``x_fp4_to_lds``' LDS layout)."""
    return xq_scales(s, H) + s * H // 32


@traced
def stage_xq(c, j, src, rceil=True):
    """Columns 128 j .. of every token's x through MXFP4 (``x_fp4_to_lds``) ->
    the shared image ``xq``, then XQF pair j (``rs`` loaded)."""
    s, tid = c["S"], c["tid"]
    per = COLS // 8
    nq = s * per
    e = fx.min(tid, nq - 1)
    t = e // per
    kk = j * per + e % per
    ys = _x_norm(c, t, _x_words(c, src, t, kk * 8))
    amax = fx.Float32(0.0)
    for d in range_constexpr(4):
        amax = fx.max(amax, fx.Float32(fmath.absf(bf_lo(ys[d]))))
        amax = fx.max(amax, fx.Float32(fmath.absf(bf_hi(ys[d]))))
    for off in range_constexpr(2):
        amax = xred(amax, 1 << off, fx.max)
    code = mx_scale(amax) if rceil else mx_scale_round(amax)
    w = fx.Int32(0)
    for d in range_constexpr(4):
        w = pk_fp4_bf16(w, ys[d], pow2(code), d)
    if tid < nq:
        q = rsrc(c["xq"])
        bo.buffer_store(w, q, t * xq_row(H) // 4 + kk, cache_modifier=CM_DEV)
        if kk % 4 == 0:
            bo.buffer_store(
                code,
                q,
                xq_scales(s, H) + t * (H // 32) + kk // 4,
                cache_modifier=CM_DEV,
            )
    publish(c["put"], c["xqf"], j, 1, tid == 0)


@traced
def xq_to_lds(c):
    """Every norm task's XQF, then the shared MXFP4 image ``xq`` -> LDS ``xl``."""
    s, tid = c["S"], c["tid"]
    if tid < NRED:
        c["poll"]([(c["xqf"], tid, 1)])
    gpu.barrier()
    n4 = xq_dwords(s) // 4
    rounds = (n4 + THREADS - 1) // THREADS
    for g0 in range_constexpr(0, rounds, XQ_GROUP):
        idx = range(g0, min(g0 + XQ_GROUP, rounds))
        es = [fx.min(tid + THREADS * i, n4 - 1) for i in idx]
        got = [
            fx.Vector(
                bo.buffer_load(
                    rsrc(c["xq"]),
                    4 * e,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )
            for e in es
        ]
        for i, e, v in zip(idx, es, got):
            if tid + THREADS * i < n4:
                fx.ptr_store(v, c["xl"] + 4 * e)
    gpu.barrier()


@traced
def stage_sgate(c, t, src):
    """Token t's shared-expert weight: sigmoid(bf16(x . w_sg)) in fp32 (the
    top-k kernel's fused sigmoid on the combined gate's bf16 logit) -> SGATE,
    x built from ``src`` (``rs`` loaded)."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    per = H // THREADS  # 16: two 8-column words a thread
    acc = fx.Float32(0.0)
    for h in range_constexpr(per // 8):
        kk = tid * per + 8 * h
        ys = _x_norm(c, t, _x_words(c, src, t, kk))
        ww = fx.Vector(
            bo.buffer_load(rsrc(c["args"]["w_sg"]), kk // 2, vec_width=4, dtype=T.i32)
        )
        for d in range_constexpr(4):
            acc = acc + bf_lo(ys[d]) * bf_lo(ww[d]) + bf_hi(ys[d]) * bf_hi(ww[d])
    acc = wave_sum(acc)
    if lane == 0:
        fx.ptr_store(acc, red + wave)
    gpu.barrier()
    if tid == 0:
        tot = fx.ptr_load(red + 0)
        for w in range_constexpr(1, WAVES):
            tot = tot + fx.ptr_load(red + w)
        c["put"](c["sgate"], t, sigmoid(bf16_round(tot)))
    gpu.barrier()


@traced
def stage_final(c, j):
    """Columns 128 j ..: FINAL in rank order -> bf16 ``out``."""
    s, tid = c["S"], c["tid"]
    for p in range_constexpr(token_passes(s)):
        t = tid // 64 + WAVES * p
        if t < s:
            col = j * COLS + 2 * (tid % 64)
            lo, hi = sum_partials(
                c["poll"],
                c["peer"]["final_own"],
                lambda src, t=t, col=col: ((src * s + t) * H + col) // 2,
                c["TP"],
            )
            bo.buffer_store(_pair(lo, hi), rsrc(c["args"]["out"]), (t * H + col) // 2)
    gpu.barrier()
