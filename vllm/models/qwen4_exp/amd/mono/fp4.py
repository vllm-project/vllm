# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 pieces at the Flash-Next shapes, whose K steps need not split evenly
over the waves: an online-MXFP4 linear's GEMV (in_proj_qkvz, out_proj) and
the token rows' MXFP4 codes for the routed experts. The arithmetic is the
Qwen3.8 mono's (``qwen3_5.amd.mono.layer``); only the K walk differs: wave w
takes K steps w, w + 8, ..., each with its own scale word."""

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr

from vllm.models.kimi_k3.amd.mono.common.ops import rsrc, traced
from vllm.models.kimi_k3.amd.mono.common.plan import THREADS, WAVES
from vllm.models.kimi_k3.amd.mono.stages import gemv
from vllm.models.qwen3_5.amd.mono.layer import (
    _quad_scale,
    fp4_mfmas,
    fp4_tile,
    pk_fp4_bf16,
    pow2,
    scale_word,
)
from vllm.models.qwen4_exp.amd.mono import layout as L


def steps(k):
    """A wave's K steps for K = ``k``: (count a wave, total)."""
    n = k // L.FP4_STEP
    return (n + WAVES - 1) // WAVES, n


def fp4_gemv_loads(lane, wave, w, ws, rg, k):
    """Rows 16 rg .. of an online-MXFP4 linear (K ``k``; the ASM FP4 GEMM's
    shuffled weight and scales) for this wave: a tile and a scale word a K
    step (clamped past the last: ``fp4_gemv`` zeroes those)."""
    per, n = steps(k)
    r_w, r_s = rsrc(w), rsrc(ws)
    row = rg * L.ROWS + lane % L.ROWS
    sts = [fx.min(wave + WAVES * i, n - 1) for i in range(per)]
    tiles = [fp4_tile(r_w, L.fp4_tile_dword(0, rg, k, st, lane)) for st in sts]
    words = [
        scale_word(r_s, L.scale_index(row, 4 * st + lane // 16, L.scale_cols(k)))
        for st in sts
    ]
    return tiles + words


def fp4_gemv(lane, wave, xl, xrow, k, s, rg, ops):
    """The ``fp4_gemv_loads`` operands against the MXFP4-exact bf16 rows at LDS
    ``xl`` (row stride ``xrow``) -> this wave's fp32 partial MFMA tile."""
    per, n = steps(k)
    t = fx.min(lane % 16, s - 1)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(per):
        raw = wave + WAVES * i
        st = fx.min(raw, n - 1)
        code = (ops[per + i] >> L.scale_code_shift(st, rg)) & 0xFF
        scale = pow2(code)
        if const_expr(per * WAVES != n):
            scale = (raw < n).select(scale, fx.Float32(0.0))
        xoff = t * xrow + st * L.FP4_STEP + 32 * (lane // 16)
        acc = fp4_mfmas(ops[i], scale, xl, xoff, acc)
    return acc


def xq_row(k):
    """Bytes a token's fp4 x row of K ``k`` takes in LDS (padded)."""
    return k // 2 + 16


def xq_scales(s, k):
    """Dword of LDS ``xl`` where the fp4 x rows' e8m0 codes start (a dword a
    1 x 32 block, token-major)."""
    return s * xq_row(k) // 4


@traced
def quant_x_fp4(c, k):
    """The x rows in LDS (bf16, K ``k``) -> MXFP4 codes over them (RCEIL, 1 x
    32 blocks, as the stock decode MoE quantizes its input): token t's block b
    at byte t ``xq_row`` + 16 b (element 2 j in byte j's low nibble; a lane's 8
    elements one word), its code at dword ``xq_scales`` + t (k / 32) + b.
    Every thread reads before any writes."""
    s, tid, xl = c["S"], c["tid"], c["xl"]
    xrow = k + gemv.LDS_PAD
    kq = k // 8
    nq = s * kq
    rounds = (nq + THREADS - 1) // THREADS
    codes = []
    for g0 in range_constexpr(0, rounds, 4):
        words = []
        for i in range_constexpr(g0, min(g0 + 4, rounds)):
            e = fx.min(tid + THREADS * i, nq - 1)
            wv, code = _quad_scale(xl + ((e // kq) * xrow + 8 * (e % kq)) // 2)
            w = fx.Int32(0)
            for d in range_constexpr(4):
                w = pk_fp4_bf16(w, wv[d], pow2(code), d)
            words.append(w)
            if const_expr(i % 4 == 0):
                codes.append(code)
            else:
                codes[-1] = codes[-1] | (code << (8 * (i % 4)))
        gpu.barrier()
        for i in range_constexpr(g0, min(g0 + 4, rounds)):
            e = fx.min(tid + THREADS * i, nq - 1)
            if tid + THREADS * i < nq:
                fx.ptr_store(words[i - g0], xl + ((e // kq) * xq_row(k) // 4 + e % kq))
    gpu.barrier()
    for i in range_constexpr(rounds):
        e = fx.min(tid + THREADS * i, nq - 1)
        if (tid + THREADS * i < nq) & (e % 4 == 0):
            code = (codes[i // 4] >> (8 * (i % 4))) & 0xFF
            fx.ptr_store(code, xl + (xq_scales(s, k) + e // 4))
    gpu.barrier()
