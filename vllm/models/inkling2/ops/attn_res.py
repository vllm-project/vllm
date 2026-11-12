# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block attention residual: softmax-weighted mix of residual stream blocks.

For each token, sources `x_n` (stored blocks, then the partial block plus an
optional addend, all in fp32) are scored as `(q . x_n) * rsqrt(mean(x_n^2) +
eps)` per pseudo-query `q`; the output is `sum_n softmax_n * x_n`, with
`x_n` RMS-normalized when `normalize_values` is set. The fp32 mix then
goes through the following sublayer's RMSNorm in the same kernel.
"""

import torch

from vllm.models.inkling2.ops.norm import rsqrt
from vllm.triton_utils import tl, tldevice, triton


@triton.jit
def _load_source(
    Stack, stride_s, Last, Addend, row_off, offs, mask, n, N_STACK, HAS_ADDEND
):
    if n < N_STACK:
        x = tl.load(Stack + n * stride_s + row_off + offs, mask, other=0)
        x = x.to(tl.float32)
    else:
        x = tl.load(Last + row_off + offs, mask, other=0).to(tl.float32)
        if HAS_ADDEND:
            x = x + tl.load(Addend + row_off + offs, mask, other=0).to(tl.float32)
    return x


@triton.jit
def _mix(
    Stack,
    stride_s,
    Last,
    Addend,
    row_off,
    offs,
    mask,
    probs,
    n_idx,
    N_STACK,
    HAS_ADDEND,
    BLOCK_D,
):
    acc = tl.zeros((BLOCK_D,), tl.float32)
    for n in tl.static_range(N_STACK + 1):
        x = _load_source(
            Stack, stride_s, Last, Addend, row_off, offs, mask, n, N_STACK, HAS_ADDEND
        )
        acc = tl.fma(tl.sum(tl.where(n_idx == n, probs, 0.0)), x, acc)
    return acc


@triton.jit
def _attn_res_kernel(
    Stack,
    Last,
    Addend,
    Queries,
    NormW,
    Out,
    stride_s,
    T,
    eps,
    norm_eps,
    D: tl.constexpr,
    N_STACK: tl.constexpr,
    HAS_ADDEND: tl.constexpr,
    NORMALIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    # One program per (token, query).
    t = tl.program_id(1).to(tl.int64)
    q_off = tl.program_id(0) * D
    row_off = t * D
    n_idx = tl.arange(0, BLOCK_N)
    d_idx = tl.arange(0, BLOCK_D)

    # Pass 1: per-source inverse RMS and query dot products.
    logits = tl.full((BLOCK_N,), float("-inf"), tl.float32)
    inv_rms = tl.zeros((BLOCK_N,), tl.float32)
    for n in tl.static_range(N_STACK + 1):
        sum_sq = tl.zeros((BLOCK_D,), tl.float32)
        dots = tl.zeros((BLOCK_D,), tl.float32)
        for d0 in tl.range(0, D, BLOCK_D):
            offs = d0 + d_idx
            mask = offs < D
            x = _load_source(
                Stack,
                stride_s,
                Last,
                Addend,
                row_off,
                offs,
                mask,
                n,
                N_STACK,
                HAS_ADDEND,
            )
            q = tl.load(Queries + q_off + offs, mask, other=0)
            sum_sq = tl.fma(x, x, sum_sq)
            dots = tl.fma(q, x, dots)
        rstd = rsqrt(tl.fma(tl.sum(sum_sq), 1.0 / D, eps))
        inv_rms = tl.where(n_idx == n, rstd, inv_rms)
        logits = tl.where(n_idx == n, tl.sum(dots) * rstd, logits)

    probs = tldevice.exp(logits - tl.max(logits))
    inv_sum = tl.div_rn(1.0, tl.sum(probs))
    if NORMALIZE:
        probs = probs * inv_rms

    # Pass 2: RMS of the probability-weighted sum; pass 3: normalized sum.
    sum_sq = 0.0
    for d0 in tl.range(0, D, BLOCK_D):
        offs = d0 + d_idx
        acc = (
            _mix(
                Stack,
                stride_s,
                Last,
                Addend,
                row_off,
                offs,
                offs < D,
                probs,
                n_idx,
                N_STACK,
                HAS_ADDEND,
                BLOCK_D,
            )
            * inv_sum
        )
        sum_sq += tl.sum(acc * acc)
    out_rstd = rsqrt(tl.div_rn(sum_sq, D) + norm_eps)
    out = Out + tl.program_id(0).to(tl.int64) * T * D + row_off
    for d0 in tl.range(0, D, BLOCK_D):
        offs = d0 + d_idx
        mask = offs < D
        acc = (
            _mix(
                Stack,
                stride_s,
                Last,
                Addend,
                row_off,
                offs,
                mask,
                probs,
                n_idx,
                N_STACK,
                HAS_ADDEND,
                BLOCK_D,
            )
            * inv_sum
        )
        w = tl.load(NormW + q_off + offs, mask, other=0).to(tl.float32)
        acc = acc * out_rstd * w
        tl.store(out + offs, acc.to(Out.dtype.element_ty), mask)


@triton.jit
def _attn_res_row_kernel(
    Stack,
    Last,
    Addend,
    Queries,
    NormW,
    Out,
    stride_s,
    T,
    eps,
    norm_eps,
    D: tl.constexpr,
    N_STACK: tl.constexpr,
    HAS_ADDEND: tl.constexpr,
    NORMALIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    # Small batches: one program per (token, query) holds the row in registers
    # as a power-of-2 head [0, BLOCK_A) and a masked tail [BLOCK_A, D), so the
    # sources are read twice instead of three times.
    t = tl.program_id(1).to(tl.int64)
    q_off = tl.program_id(0) * D
    row_off = t * D
    da = tl.arange(0, BLOCK_A)
    db = BLOCK_A + tl.arange(0, BLOCK_B)
    mask_a = da < D
    mask_b = db < D
    n_idx = tl.arange(0, BLOCK_N)
    qa = tl.load(Queries + q_off + da, mask_a, other=0)
    qb = tl.load(Queries + q_off + db, mask_b, other=0)
    src = (Stack, stride_s, Last, Addend, row_off)

    logits = tl.full((BLOCK_N,), float("-inf"), tl.float32)
    inv_rms = tl.zeros((BLOCK_N,), tl.float32)
    for n in tl.static_range(N_STACK + 1):
        xa = _load_source(*src, da, mask_a, n, N_STACK, HAS_ADDEND)
        xb = _load_source(*src, db, mask_b, n, N_STACK, HAS_ADDEND)
        rstd = rsqrt(tl.fma(tl.sum(xa * xa) + tl.sum(xb * xb), 1.0 / D, eps))
        inv_rms = tl.where(n_idx == n, rstd, inv_rms)
        dot = tl.sum(qa * xa) + tl.sum(qb * xb)
        logits = tl.where(n_idx == n, dot * rstd, logits)

    probs = tldevice.exp(logits - tl.max(logits))
    inv_sum = tl.div_rn(1.0, tl.sum(probs))
    if NORMALIZE:
        probs = probs * inv_rms

    acc_a = tl.zeros((BLOCK_A,), tl.float32)
    acc_b = tl.zeros((BLOCK_B,), tl.float32)
    for n in tl.static_range(N_STACK + 1):
        p = tl.sum(tl.where(n_idx == n, probs, 0.0))
        xa = _load_source(*src, da, mask_a, n, N_STACK, HAS_ADDEND)
        xb = _load_source(*src, db, mask_b, n, N_STACK, HAS_ADDEND)
        acc_a = tl.fma(p, xa, acc_a)
        acc_b = tl.fma(p, xb, acc_b)
    acc_a = acc_a * inv_sum
    acc_b = acc_b * inv_sum
    sum_sq = tl.sum(acc_a * acc_a) + tl.sum(acc_b * acc_b)
    out_rstd = rsqrt(tl.div_rn(sum_sq, D) + norm_eps)
    out = Out + tl.program_id(0).to(tl.int64) * T * D + row_off
    wa = tl.load(NormW + q_off + da, mask_a, other=0).to(tl.float32)
    wb = tl.load(NormW + q_off + db, mask_b, other=0).to(tl.float32)
    tl.store(out + da, (acc_a * out_rstd * wa).to(Out.dtype.element_ty), mask_a)
    tl.store(out + db, (acc_b * out_rstd * wb).to(Out.dtype.element_ty), mask_b)


def attn_res(
    stack: torch.Tensor,
    last: torch.Tensor,
    addend: torch.Tensor | None,
    queries: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    normalize_values: bool,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mix `[*stack, last (+ addend)]` with each pseudo-query.

    `queries` are fp32 `(Q, D)`, already scaled by `D**-0.5`; each fp32
    mix is RMS-normalized with its row of `norm_weight` `(Q, D)`. Returns
    `(Q, T, D)` in `last.dtype`.
    """
    num_tokens, d = last.shape
    num_stack = stack.shape[0]
    num_q = queries.shape[0]
    assert last.is_contiguous() and queries.is_contiguous()
    assert queries.dtype == torch.float32
    assert stack.stride(1) == d and stack.stride(2) == 1
    if addend is not None:
        assert addend.is_contiguous() and addend.shape == last.shape
    assert norm_weight.shape == queries.shape and norm_weight.is_contiguous()
    out = torch.empty((num_q, num_tokens, d), dtype=last.dtype, device=last.device)
    args = (
        stack,
        last,
        addend,
        queries,
        norm_weight,
        out,
        stack.stride(0),
        num_tokens,
        eps,
        norm_eps,
    )
    meta = dict(
        D=d,
        N_STACK=num_stack,
        HAS_ADDEND=addend is not None,
        NORMALIZE=normalize_values,
        BLOCK_N=triton.next_power_of_2(num_stack + 1),
        enable_fp_fusion=False,
        launch_pdl=True,
    )
    # The queries of a token run back to back and share its sources in L2.
    grid = (num_q, num_tokens)
    if num_tokens * num_q <= 256:
        block_a = triton.next_power_of_2(d + 1) // 2
        block_b = max(triton.next_power_of_2(d - block_a), 16)
        _attn_res_row_kernel[grid](
            *args, **meta, BLOCK_A=block_a, BLOCK_B=block_b, num_warps=8
        )
    else:
        _attn_res_kernel[grid](*args, **meta, BLOCK_D=1024, num_warps=4)
    return out
