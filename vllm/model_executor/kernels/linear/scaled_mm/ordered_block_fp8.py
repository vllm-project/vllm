# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ordered K128 FP32 partials for the qualified SM120 low-M route."""

from vllm.triton_utils import tl, triton


@triton.jit
def _raw_partial(
    a,
    b,
    raw,
    M: tl.constexpr,
    N: tl.constexpr,
    GROUPS: tl.constexpr,
    GPS: tl.constexpr,
    ASTR: tl.constexpr,
    BSTR: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pm, pn, ps = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows, cols = pm * BM + tl.arange(0, BM), pn * BN + tl.arange(0, BN)
    rm, cm = rows < M, cols < N
    for local_g in range(0, GPS):
        g = ps * GPS + local_g
        kk = g * 128 + tl.arange(0, 128)
        av = tl.load(
            a + rows[:, None] * ASTR + kk[None, :], mask=rm[:, None], other=0.0
        )
        bv = tl.load(
            b + cols[None, :] * BSTR + kk[:, None], mask=cm[None, :], other=0.0
        )
        dot = tl.dot(av, bv, out_dtype=tl.float32, max_num_imprecise_acc=0)
        tl.store(
            raw + g * M * N + rows[:, None] * N + cols[None, :],
            dot,
            mask=rm[:, None] & cm[None, :],
        )


@triton.jit
def _row_reduce(
    raw,
    as_,
    bs,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    G: tl.constexpr,
    SAM: tl.constexpr,
    SAG: tl.constexpr,
    SBN: tl.constexpr,
    BN: tl.constexpr,
):
    row, pn = tl.program_id(0), tl.program_id(1)
    cols = pn * BN + tl.arange(0, BN)
    cm = cols < N
    total = tl.zeros((BN,), tl.float32)
    for g in range(0, G):
        dot = tl.load(raw + g * M * N + row * N + cols, mask=cm, other=0.0)
        sa = tl.load(as_ + row * SAM + g * SAG)
        # BN=64/128: one tile never spans a 128-N scale block.
        sb = tl.load(bs + ((pn * BN) // 128) * SBN + g)
        total = tl.fma(dot, sa * sb, total)
    tl.store(out + row * N + cols, total.to(tl.bfloat16), mask=cm)


def ordered_block_fp8_mm(a, b, sa, sb, out, raw):
    m, k = a.shape
    groups = k // 128
    _raw_partial[(triton.cdiv(m, 16), triton.cdiv(2560, 64), 8)](
        a,
        b,
        raw,
        M=m,
        N=2560,
        GROUPS=groups,
        GPS=groups // 8,
        ASTR=a.stride(0),
        BSTR=b.stride(0),
        BM=16,
        BN=64,
        num_warps=4,
        num_stages=2,
    )
    bn, warps = (64, 2) if k == 4096 else (128, 4)
    _row_reduce[(m, triton.cdiv(2560, bn))](
        raw,
        sa,
        sb,
        out,
        M=m,
        N=2560,
        G=groups,
        SAM=sa.stride(0),
        SAG=sa.stride(1),
        SBN=sb.stride(0),
        BN=bn,
        num_warps=warps,
        num_stages=2,
    )
    return out
