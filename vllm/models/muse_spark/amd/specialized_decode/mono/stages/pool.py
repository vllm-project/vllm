# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2 phase 2c and 3a (README, "K2 program order"): the pooler and the
pre-expert norm + MXFP4 a1.

    pooler  (CTA P0 + task, task < 128: g = task // 4, q = task % 4) pooled
            columns n0 = 512 rank + 16 g .. + 16 against ffn_in's K quarter q
            (``xrow``, the 16 seam slices covering it): a 16-row bf16 GEMV
            (``gemv``); parts q > 0 hand their fp32 partials to part 0
            (PPART), which adds them in part order, rounds to bf16 once and
            all-gathers the rows (every rank's POOL).
    load_xq (every CTA) all 4096 pooled columns of every row -> the
            pre-expert RMSNorm (``fused_rmsnorm_expand``: x rsqrt(mean x^2 +
            eps), times (w + 1), bf16) -> MXFP4 a1 by aiter's HIP RoundUp rule
            (``fused_dynamic_mx_quant_moe_sort``: a 32-group's amax floored at
            1e-10, the ceil code of amax * f32(1/6), RNE e2m1) -> LDS ``xq``
            (fp4 words) and ``xsc`` (codes), the ug stage's B operand.

Every CTA computes identical a1 bytes from identical inputs, so nothing is
handed off. A pad row's normed values are zeros by ``select`` (its a1 is fp4
zeros with the floor's code 92).

The pooler is K3's early latent task (pr60683 ``mono/stages/moe.py``
``early_loads`` / ``early_compute`` / ``early_out``) at Muse's shapes;
``load_xq`` is K3's ``load_latq`` with the norm and the 1e-10 floor added.
"""

import functools

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from vllm.models.muse_spark.amd.specialized_decode.mono.common.mx import (
    e8m0_roundup,
    pk_fp4_bf16,
    pow2,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import (
    CM_DEV,
    bf2_f32,
    bf16_pair,
    hw_rsq,
    ld_pad,
    row_sum,
    rsrc,
    traced,
    wave_sum,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.plan import (
    cdiv,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.spec import (
    AMAX_FLOOR,
    COLS,
    GAIN,
    HIDDEN,
    MOE_HIDDEN,
    OWN_COLS,
    P_K,
    P_PARTS,
    PRE_EXP_EPS,
    ROWS,
    THREADS,
    TP,
    XQ_WORDS,
    XROW,
    XSC_WORDS,
    pool_pair,
    ppart_pair,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.stages import gemv as gemv

XL_ROW = P_K + gemv.LDS_PAD  # bf16 a token row of the pooler's LDS x slice
SLICES_A_PART = P_K // COLS  # 16 seam slices cover a K quarter
GROUP = 32  # an MX group: 32 values, one e8m0 code, 4 fp4 words
GROUP_WORDS = GROUP // 2  # 16 bf16 pair words a group
assert XSC_WORDS * GROUP == MOE_HIDDEN and XQ_WORDS * 8 == MOE_HIDDEN


@functools.cache
def lds_struct(s: int):
    """The pooler's LDS: the K quarter of rows 0 .. s-1 of ffn_in (bf16, a
    row every ``XL_ROW``)."""

    @fx.struct
    class PoolLds:
        xl: fx.Array[fx.Int32, s * XL_ROW // 2, 16]

    return PoolLds


@functools.cache
def xq_lds_struct(s: int):
    """``load_xq``'s LDS, live until the last ug task (K2's layer build
    composes it with ``experts.ug_lds_struct`` as ``xq_ug``): ``xq`` row t's
    fp4 words (word ``t 512 + 4 kb + w``: K ``32 kb + 8 w ..``), ``xsc`` its
    e8m0 codes (a word each, ``t 128 + kb``), ``xss`` the 32-groups' sums of
    squares."""

    @fx.struct
    class XqLds:
        xq: fx.Array[fx.Int32, s * XQ_WORDS, 16]
        xsc: fx.Array[fx.Int32, s * XSC_WORDS, 16]
        xss: fx.Array[fx.Float32, s * XSC_WORDS, 16]

    return XqLds


# ---------------------------------------------------------------- pooler


def _task(c, task):
    """(g, q, n0): the pooler task's row group, K quarter and first pooled
    column (row of ``w_pool``)."""
    g = task // P_PARTS
    q = task % P_PARTS
    return g, q, c["rank"] * OWN_COLS + g * ROWS


def pool_loads(c, task):
    """Pooler task ``task``'s ``w_pool`` operands for this wave: rows n0 .. +16,
    K ``2048 q ..`` (2 chunks a wave, nontemporal). Issued before the XRDY
    waits."""
    _, q, n0 = _task(c, task)
    return gemv.loads(
        c["lane"], c["wave"], c["args"]["w_pool"], n0, P_K, ldk=HIDDEN, k0=q * P_K
    )


@traced
def stage_pool(c, task, ops):
    """Pooler task ``task``: its K quarter of rows 0 .. S-1 of xrow -> LDS
    (after the 16 seam slices covering it) -> ``gemv.mfmas`` -> fp32 rows.
    q > 0: PPART. q == 0: + the 3 parts in part order -> bf16 -> every rank's
    POOL (element ``t 4096 + col``)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    g, q, n0 = _task(c, task)
    xl = c["lds"]["pool"]["xl"]
    if tid < SLICES_A_PART:
        c["poll"]([(c["scr"]["xrdy"], q * SLICES_A_PART + tid, 1)])
    gpu.barrier()
    gemv.rows_to_lds(
        tid, c["scr"][XROW], P_K, s, xl, XL_ROW, CM_DEV, ldk=HIDDEN, k0=q * P_K
    )
    acc = gemv.mfmas(lane, wave, xl, XL_ROW, P_K, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if (q > 0) & (tid < s * ROWS):
        r = tid % ROWS
        t = tid // ROWS
        c["put"](c["scr"]["ppart"], ppart_pair(g, q, t, r), row_sum(red, r, t))
    if (q == 0) & (tid < s * (ROWS // 2)):
        rp = tid % (ROWS // 2)
        t = tid // (ROWS // 2)
        # rows 2 rp and 2 rp + 1 are adjacent pairs of each part
        got = c["poll"](
            [
                (c["scr"]["ppart"], ppart_pair(g, qq, t, 2 * rp), 2)
                for qq in range(1, P_PARTS)
            ]
        )
        v0 = row_sum(red, 2 * rp, t)
        v1 = row_sum(red, 2 * rp + 1, t)
        for qq in range_constexpr(P_PARTS - 1):
            v0 = v0 + got[qq][0].bitcast(fx.Float32)
            v1 = v1 + got[qq][1].bitcast(fx.Float32)
        at = t * MOE_HIDDEN + n0 + 2 * rp
        for p in range_constexpr(TP):
            c["put_bf"](c["peer"]["pool"][p], at, [v0, v1])
    gpu.barrier()


# ---------------------------------------------------------------- a1


def _group_quant(ws, wq, r, pad_t):
    """A 32-group: pooled bf16 pair words ``ws`` (16), pre-expert weight pair
    words ``wq`` (16), the row's rsqrt ``r``, its pad flag -> (the e8m0 code,
    the 4 fp4 words). xe = bf16((p r) (w + 1)), as
    ``_rmsnorm_expand_kernel``; then the HIP RoundUp quant of those bf16."""
    xe = []
    am = fx.Int32(0)
    for m in range(GROUP_WORDS):
        p0, p1 = bf2_f32(ws[m])
        w0, w1 = bf2_f32(wq[m])
        y0 = (p0 * r) * (w0 + fx.Float32(GAIN))
        y1 = (p1 * r) * (w1 + fx.Float32(GAIN))
        word = bf16_pair(y0, y1).bitcast(fx.Int32)
        word = fx.Int32(pad_t.select(fx.Int32(0), word))
        xe.append(word)
        # |bf16| ordered as its magnitude bits (finite values)
        am = fx.max(am, word & 0x7FFF)
        am = fx.max(am, (word >> 16) & 0x7FFF)
    amax = fx.max((am << 16).bitcast(fx.Float32), fx.Float32(AMAX_FLOOR))
    code = e8m0_roundup(amax)
    scale = pow2(code)
    pks = []
    for w in range(GROUP // 8):
        pk = fx.Int32(0)
        for j in range(4):
            pk = pk_fp4_bf16(pk, xe[4 * w + j], scale, j)
        pks.append(pk)
    return code, pks


@traced
def load_xq(c):
    """POOL (all 4096 columns of rows 0 .. S-1) -> pre-expert norm -> MXFP4:
    LDS ``xq`` / ``xsc`` (and ``tot``: a row's sum of squares). Thread a
    32-group b = tid + 512 i: row t = b // 128, columns ``32 kb ..``
    (kb = b % 128)."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    a = c["args"]
    lds = c["lds"]["xq"]
    xq, xsc, xss = lds["xq"], lds["xsc"], lds["xss"]
    tot = c["rt"]["tot"]
    ngrp = s * XSC_WORDS
    r_w = rsrc(a["w_preexp"])
    # every round's weights and pad byte before the first POOL wait (the
    # "loads before waits" rule: at S > 4 a thread has two rounds)
    pre = []
    for i in range_constexpr(cdiv(ngrp, THREADS)):
        b = fx.min(tid + THREADS * i, ngrp - 1)
        t = b // XSC_WORDS
        kb = b % XSC_WORDS
        wq = []
        for h in range_constexpr(GROUP_WORDS // 4):
            v = fx.Vector(
                bo.buffer_load(r_w, kb * GROUP_WORDS + 4 * h, vec_width=4, dtype=T.i32)
            )
            wq += [v[e] for e in range(4)]
        pre.append((i, b, t, kb, wq, ld_pad(a["pad"], t)))
    groups = []
    for i, b, t, kb, wq, pad_t in pre:
        got = c["poll"](
            [
                (c["peer"]["pool_own"], pool_pair(t, GROUP * kb) + 2 * jj, 2)
                for jj in range(GROUP_WORDS // 2)
            ]
        )
        ws = [got[jj][m] for jj in range(GROUP_WORDS // 2) for m in range(2)]
        gss = fx.Float32(0.0)
        for m in range_constexpr(GROUP_WORDS):
            p0, p1 = bf2_f32(ws[m])
            gss = fx.Float32(fmath.fma(p0, p0, gss))
            gss = fx.Float32(fmath.fma(p1, p1, gss))
        if tid + THREADS * i < ngrp:
            fx.ptr_store(gss, xss + b)
        groups.append((i, t, kb, wq, ws, pad_t))
    gpu.barrier()
    # a row's sum of squares: its 128 group sums, a wave a row
    if wave < s:
        v = fx.ptr_load(xss + (wave * XSC_WORDS + lane)) + fx.ptr_load(
            xss + (wave * XSC_WORDS + 64 + lane)
        )
        v = wave_sum(v)
        if lane == 0:
            fx.ptr_store(v, tot + wave)
    gpu.barrier()
    for i, t, kb, wq, ws, pad_t in groups:
        # sum / D + eps: one fma, as the stock kernel compiles (v_fmamk_f32;
        # bit-equal to mul + add here: 1 / 4096 is a power of two)
        r = hw_rsq(
            fmath.fma(
                fx.ptr_load(tot + t),
                fx.Float32(1.0 / MOE_HIDDEN),
                fx.Float32(PRE_EXP_EPS),
            )
        )
        code, pks = _group_quant(ws, wq, r, pad_t)
        if tid + THREADS * i < ngrp:
            for w in range_constexpr(GROUP // 8):
                fx.ptr_store(pks[w], xq + (t * XQ_WORDS + kb * (GROUP // 8) + w))
            fx.ptr_store(code, xsc + (t * XSC_WORDS + kb))
    gpu.barrier()
