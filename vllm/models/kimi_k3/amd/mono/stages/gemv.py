# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""bf16 GEMV pieces the K3 mono kernels share: 16 weight rows a task against up
to 16 token rows in LDS, one ``mfma_f32_16x16x32_bf16`` per 32 K.

K is walked in 128-wide chunks, wave w taking chunks w, w + 8, ...; a lane reads
64 contiguous bytes of its row a chunk (4 dwordx4 at q 32 + g 8 elements: lane
group g = lane / 16), and the token rows are read at the same K. The K order
inside a chunk is a permutation both operands share, so the dot is unchanged.

The token rows are in LDS whole on gfx950. gfx942's 64 KB holds a window of
``win_rounds`` rounds (a round: the WAVES chunks the waves take together) at a
time (``mfmas_windowed``): the next window's rows are loaded into registers
under the current one's MFMAs, then stored over it.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from vllm.models.kimi_k3.amd.mono.common.ops import (
    CM_DEV,
    CM_NT,
    mfma_bf16,
    rsrc,
    traced,
)
from vllm.models.kimi_k3.amd.mono.common.plan import GFX942, THREADS, WAVES

ROWS = 16
KCH = 128
LDS_PAD = 8  # bf16 a padded token row
ROUND_K = WAVES * KCH  # 1024
XL_WINDOW = 33 * 1024  # gfx942: the LDS a GEMV's token-row window takes, at most


def chunks(k):
    """A wave's K chunks for K = ``k``: (count a wave, total)."""
    n = k // KCH
    return (n + WAVES - 1) // WAVES, n


def loads(lane, wave, w, row0, k, nt=True, ldk=None, k0=0):
    """This wave's operands of weight rows ``row0`` .. + 16 (row-major, ``ldk``
    bf16 a row at ``w``, K columns ``k0`` .. ``k0 + k``): a list a chunk
    (clamped into K: ``mfmas`` zeroes the extra)."""
    per, n = chunks(k)
    ldk = k if ldk is None else ldk
    r_w = rsrc(w)
    row = row0 + lane % ROWS
    g = lane // ROWS
    out = []
    for i in range(per):
        ch = fx.min(wave + WAVES * i, n - 1)
        base = (row * ldk + k0 + ch * KCH + g * 8) // 2
        out.append(
            [
                fx.Vector(
                    bo.buffer_load(
                        r_w,
                        base + 16 * q,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_NT if nt else 0,
                    )
                )
                for q in range(4)
            ]
        )
    return out


def mfmas(lane, wave, xl, xrow, k, s, ops):
    """``loads``' operands against token rows 0 .. s - 1 of the LDS rows ``xl``
    (``xrow`` bf16 a row) -> this wave's 16x16 accumulator."""
    per, n = chunks(k)
    g = lane // ROWS
    t = fx.min(lane % ROWS, s - 1)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(per):
        ch = wave + WAVES * i
        live = ch < n
        kk = fx.min(ch, n - 1) * KCH + g * 8
        for q in range_constexpr(4):
            xb = fx.Vector(
                fx.ptr_load(
                    xl + (t * xrow + kk + 32 * q) // 2,
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
            )
            if const_live(per, n):
                xb = fx.Vector.from_elements(
                    [live.select(xb[d], fx.Int32(0)) for d in range(4)], fx.Int32
                )
            acc = mfma_bf16(
                ops[i][q].bitcast(fx.BFloat16), xb.bitcast(fx.BFloat16), acc
            )
    return acc


def const_live(per, n):
    """Whether some wave's last chunk is past K (its MFMA must add zero)."""
    return per * WAVES != n


@traced
def rows_to_lds(tid, src, k, s, xl, xrow, cm=0, ldk=None, k0=0):
    """Columns k0 .. k0 + k of token rows 0 .. s - 1 of the bf16 [s, ldk] at
    ``src`` -> LDS (``xrow`` a row), 16 B a lane."""
    ldk = k if ldk is None else ldk
    words = s * k // 8
    for i in range_constexpr((words + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, words - 1)
        t = e // (k // 8)
        kk = e % (k // 8) * 8
        v = fx.Vector(
            bo.buffer_load(
                rsrc(src),
                (t * ldk + k0 + kk) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=cm,
            )
        )
        if tid + THREADS * i < words:
            fx.ptr_store(v, xl + (t * xrow + kk) // 2)
    gpu.barrier()


def dev_rows_to_lds(tid, src, k, s, xl, xrow):
    """``rows_to_lds`` at device scope: rows another CTA of this launch wrote."""
    rows_to_lds(tid, src, k, s, xl, xrow, CM_DEV)


# ---------------------------------------------------------------- K windows
def win_rounds(s, k):
    """Rounds of a token-row window for K = ``k`` at S = ``s``: all of K on
    gfx950, on gfx942 as many as fit XL_WINDOW."""
    per, _ = chunks(k)
    if not GFX942:
        return per
    fit = (XL_WINDOW // (2 * s) - LDS_PAD) // ROUND_K
    return max(1, min(per, fit))


def win_k(s, k):
    """K columns of a window (the last may be shorter)."""
    return min(win_rounds(s, k) * ROUND_K, k)


def win_row(s, k):
    """bf16 a window's LDS row (padded)."""
    return win_k(s, k) + LDS_PAD


def windows(s, k):
    per, _ = chunks(k)
    r = win_rounds(s, k)
    return (per + r - 1) // r


def window_loads(tid, src, ldk, k0, s, kw, cm):
    """This thread's 16 B pieces of columns k0 .. k0 + kw of token rows
    0 .. s - 1 of the bf16 [s, ldk] at ``src`` (``window_store`` stores them)."""
    words = s * kw // 8
    out = []
    for i in range((words + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, words - 1)
        t = e // (kw // 8)
        kk = e % (kw // 8) * 8
        out.append(
            fx.Vector(
                bo.buffer_load(
                    rsrc(src),
                    (t * ldk + k0 + kk) // 2,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=cm,
                )
            )
        )
    return out


@traced
def window_store(tid, got, s, kw, xl, xrow):
    """``window_loads``' pieces -> LDS rows of ``xrow`` bf16."""
    words = s * kw // 8
    for i in range_constexpr(len(got)):
        e = fx.min(tid + THREADS * i, words - 1)
        t = e // (kw // 8)
        kk = e % (kw // 8) * 8
        if tid + THREADS * i < words:
            fx.ptr_store(got[i], xl + (t * xrow + kk) // 2)


def chunk_mfmas(lane, xl, xrow, s, ops_i, kk, acc, live=None):
    """One 128-K chunk: ``ops_i`` (a ``loads`` entry) against the LDS token rows
    at local column ``kk`` (+ the lane group's 8) -> ``acc``; ``live`` False:
    the chunk is past K (adds zero)."""
    g = lane // ROWS
    t = fx.min(lane % ROWS, s - 1)
    for q in range_constexpr(4):
        xb = fx.Vector(
            fx.ptr_load(
                xl + (t * xrow + kk + g * 8 + 32 * q) // 2,
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
        )
        if live is not None:
            xb = fx.Vector.from_elements(
                [live.select(xb[d], fx.Int32(0)) for d in range(4)], fx.Int32
            )
        acc = mfma_bf16(ops_i[q].bitcast(fx.BFloat16), xb.bitcast(fx.BFloat16), acc)
    return acc


def mfmas_windowed(tid, lane, wave, src, k, s, xl, ops, cm=0, ldk=None, k0=0):
    """``rows_to_lds`` + ``mfmas`` (columns k0 .. k0 + k of the bf16 [s, ldk]
    token rows at ``src``), the rows a window at a time where LDS holds less
    than all of K. ``ops``: every chunk's operands, issued before."""
    ldk = k if ldk is None else ldk
    nw = windows(s, k)
    xrow = win_row(s, k)
    if nw == 1:
        rows_to_lds(tid, src, k, s, xl, xrow, cm, ldk=ldk, k0=k0)
        return mfmas(lane, wave, xl, xrow, k, s, ops)
    per, n = chunks(k)
    r = win_rounds(s, k)
    wk = r * ROUND_K
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    pre = window_loads(tid, src, ldk, k0, s, min(wk, k), cm)
    for p in range_constexpr(nw):
        kw = min(wk, k - p * wk)
        if p > 0:
            gpu.barrier()  # the last window's reads are done
        window_store(tid, pre, s, kw, xl, xrow)
        if p + 1 < nw:
            pre = window_loads(
                tid, src, ldk, k0 + (p + 1) * wk, s, min(wk, k - (p + 1) * wk), cm
            )
        gpu.barrier()
        for i in range_constexpr(p * r, min((p + 1) * r, per)):
            ch = wave + WAVES * i
            live = (ch < n) if (i + 1) * WAVES > n else None
            kk = fx.min(ch * KCH - p * wk, kw - KCH)
            acc = chunk_mfmas(lane, xl, xrow, s, ops[i], kk, acc, live)
    return acc
