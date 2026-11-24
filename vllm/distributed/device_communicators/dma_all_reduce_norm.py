# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SDMA all-reduce's dequantize-add fused with residual add + Gemma RMSNorm
+ MXFP8 quant (and optionally the bf16 normed rows), for
`DmaAllReduce.add_rmsnorm_mxfp8`.

Unfused, an all-reduce + norm site runs `_dq_sum_kernel` (two INT4 messages ->
bf16 sum, 19.2 KB per row at hidden size 6144) and then
`gemma_fused_add_rmsnorm_mxfp8[_and_bf16]` (sum + residual -> MXFP8 values,
E8M0 scales, residual_out (+ bf16 normed), 43.2 (+ 12.3) KB per row). Fused, one
pass reads the two messages and the residual: 37.8 (+ 12.3) KB per row, 0.63-0.70x
of the two kernels' time on MI350X at 2k-16k rows.

Numerics follow the two kernels: the sum is rounded to the input dtype (the
all-reduce output) before the add, and the MXFP8 quant runs on the input-dtype
normed row. Only the order of the variance's fp32 sum differs, so a few normed
values in 10^6 differ by one ulp.
"""

import torch

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _dq_sum_add_gemma_rmsnorm_mxfp8_kernel(
    qa_ptr,
    sa_ptr,
    qb_ptr,
    sb_ptr,
    res_ptr,
    w_ptr,
    xq_ptr,
    xs_ptr,
    res_out_ptr,
    normed_ptr,
    n_groups,
    stride_rrow,
    eps,
    BLOCK_G: tl.constexpr,
    ALSO_BF16: tl.constexpr,
    FP8_MAX: tl.constexpr,
    TINY: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    g = tl.arange(0, BLOCK_G)
    gm = g < n_groups
    m2 = gm[:, None]
    gq = row * n_groups + g
    qoff = gq[:, None] * 16 + tl.arange(0, 16)[None, :]
    pa = tl.load(qa_ptr + qoff, mask=m2, other=0).to(tl.int32)
    pb = tl.load(qb_ptr + qoff, mask=m2, other=0).to(tl.int32)
    sa = tl.load(sa_ptr + gq, mask=gm, other=0.0).to(tl.float32)[:, None]
    sb = tl.load(sb_ptr + gq, mask=gm, other=0.0).to(tl.float32)[:, None]
    lo = ((pa & 15) - 8).to(tl.float32) * sa + ((pb & 15) - 8).to(tl.float32) * sb
    hi = ((pa >> 4) - 8).to(tl.float32) * sa + ((pb >> 4) - 8).to(tl.float32) * sb
    # Round like the unfused all-reduce, whose output is in the input dtype.
    x = (
        tl.reshape(tl.join(lo, hi), (BLOCK_G, 32))
        .to(res_out_ptr.dtype.element_ty)
        .to(tl.float32)
    )
    n_cols = n_groups * 32
    cols = g[:, None] * 32 + tl.arange(0, 32)[None, :]
    r = tl.load(res_ptr + row * stride_rrow + cols, mask=m2, other=0.0).to(tl.float32)
    s = x + r
    off = row * n_cols + cols
    tl.store(res_out_ptr + off, s.to(res_out_ptr.dtype.element_ty), mask=m2)
    var = tl.sum(tl.sum(s * s, axis=1), axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=m2, other=0.0).to(tl.float32)
    out = s * rstd * (1.0 + w)
    if ALSO_BF16:
        tl.store(normed_ptr + off, out.to(normed_ptr.dtype.element_ty), mask=m2)
    y = out.to(res_out_ptr.dtype.element_ty).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(y), axis=1), TINY)
    e = tl.ceil(tl.log2(amax / FP8_MAX)) + 127.0
    e = tl.minimum(tl.maximum(e, 0.0), 254.0)
    # exp2(e - 127) is subnormal at e == 0 and flushes to zero on CDNA, so
    # scale by the (exact, normal) reciprocal instead of dividing.
    q = y * tl.exp2(127.0 - e)[:, None]
    tl.store(xq_ptr + off, q.to(xq_ptr.dtype.element_ty), mask=m2)
    tl.store(xs_ptr + row * n_groups + g, e.to(xs_ptr.dtype.element_ty), mask=gm)


def dq_sum_add_gemma_rmsnorm_mxfp8(
    qa,
    sa,
    qb,
    sb,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    also_bf16: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Sum two INT4 messages, add `residual`, Gemma RMSNorm, quantize to MXFP8.

    Args:
        qa: Rank 0's packed INT4 values, uint8 `[M * N // 2]` (anything with
            `data_ptr()` and `dtype`).
        sa: Rank 0's fp16 scales, one per 32 values.
        qb: Rank 1's packed INT4 values.
        sb: Rank 1's fp16 scales.
        residual: `[M, N]`, unit column stride.
        weight: The Gemma RMSNorm weight, `[N]`.
        eps: The RMSNorm epsilon.
        also_bf16: Also return the normed rows in the residual's dtype.

    Returns:
        `(x_q, x_scale, normed, residual_out)` as
        `gemma_fused_add_rmsnorm_mxfp8_and_bf16`: float8_e4m3fn `[M, N]`, E8M0
        uint8 `[M, N // 32]` (row-major), the normed rows or None, and the new
        residual.

    """
    m, n = residual.shape
    assert n % MXFP8_BLOCK_SIZE == 0 and residual.stride(1) == 1
    dev = residual.device
    xq = torch.empty((m, n), dtype=MXFP8_VALUE_DTYPE, device=dev)
    xs = torch.empty((m, n // MXFP8_BLOCK_SIZE), dtype=MXFP8_SCALE_DTYPE, device=dev)
    res_out = torch.empty((m, n), dtype=residual.dtype, device=dev)
    normed = torch.empty_like(res_out) if also_bf16 else None
    ng = n // MXFP8_BLOCK_SIZE
    _dq_sum_add_gemma_rmsnorm_mxfp8_kernel[(m,)](
        qa,
        sa,
        qb,
        sb,
        residual,
        weight,
        xq,
        xs,
        res_out,
        normed if normed is not None else res_out,
        ng,
        residual.stride(0),
        eps,
        BLOCK_G=triton.next_power_of_2(ng),
        ALSO_BF16=also_bf16,
        FP8_MAX=float(torch.finfo(MXFP8_VALUE_DTYPE).max),
        TINY=float(torch.finfo(torch.float32).tiny),
        num_warps=4,
    )
    return xq, xs, normed, res_out
