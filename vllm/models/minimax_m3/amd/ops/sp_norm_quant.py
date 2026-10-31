# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma RMSNorm (optionally after the residual add) + per-token MXFP8 quant in one
pass for ``sp_prefill``, with ``gemma_rmsnorm``'s sum order. The scale byte is
clamped to [0, 254] (``mxfp8_e4m3_quantize``) or [85, 254] (aiter's MoE quant)."""

import torch

from vllm.triton_utils import tl, triton

# aiter's MoE input quant: scale at least 2^-42.
MOE_MIN_SCALE_BYTE = 85


@triton.jit
def _rmsnorm_mx_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    out_ptr,
    res_out_ptr,
    q_ptr,
    s_ptr,
    n_cols,
    eps,
    HAS_RES: tl.constexpr,
    STORE_OUT: tl.constexpr,
    MIN_SB: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_MAX: tl.constexpr,
    TINY: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    s = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_RES:
        r = tl.load(res_ptr + row * n_cols + cols, mask=mask, other=0.0)
        s = s + r.to(tl.float32)
        tl.store(
            res_out_ptr + row * n_cols + cols,
            s.to(res_out_ptr.dtype.element_ty),
            mask=mask,
        )
    var = tl.sum(s * s, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (s * rstd * (1.0 + w)).to(out_ptr.dtype.element_ty)
    if STORE_OUT:
        tl.store(out_ptr + row * n_cols + cols, out, mask=mask)
    y = tl.reshape(out.to(tl.float32), (BLOCK_N // 32, 32))
    amax = tl.maximum(tl.max(tl.abs(y), axis=1), TINY)
    sb = tl.ceil(tl.log2(amax / FP8_MAX)) + 127.0
    sb = tl.minimum(tl.maximum(sb, MIN_SB), 254.0)
    q = (y * tl.exp2(127.0 - sb)[:, None]).to(q_ptr.dtype.element_ty)
    tl.store(q_ptr + row * n_cols + cols, tl.reshape(q, (BLOCK_N,)), mask=mask)
    groups = tl.arange(0, BLOCK_N // 32)
    tl.store(
        s_ptr + row * (n_cols // 32) + groups,
        sb.to(tl.uint8),
        mask=groups < n_cols // 32,
    )


def _out_or_empty(out, shape, dtype, device) -> torch.Tensor:
    if out is None:
        return torch.empty(shape, dtype=dtype, device=device)
    assert out.shape == shape and out.dtype == dtype and out.is_contiguous()
    return out


def gemma_rmsnorm_mx(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    eps: float,
    *,
    min_scale_byte: int = 0,
    store_out: bool = False,
    q_out: torch.Tensor | None = None,
    scale_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> (normalized bf16 if ``store_out``, x + residual (``x`` without one),
    MXFP8 codes, e8m0 scales ``[M, N/32]``); the codes / scales into ``q_out`` /
    ``scale_out`` when given."""
    m, n = x.shape
    assert x.is_contiguous() and n % 32 == 0
    assert residual is None or residual.is_contiguous()
    out = torch.empty_like(x) if store_out else None
    res_out = torch.empty_like(x) if residual is not None else x
    q = _out_or_empty(q_out, (m, n), torch.float8_e4m3fn, x.device)
    scale = _out_or_empty(scale_out, (m, n // 32), torch.uint8, x.device)
    block_n = triton.next_power_of_2(n)
    _rmsnorm_mx_kernel[(m,)](
        x,
        x if residual is None else residual,
        weight,
        x if out is None else out,
        res_out,
        q,
        scale,
        n,
        eps,
        HAS_RES=residual is not None,
        STORE_OUT=store_out,
        MIN_SB=float(min_scale_byte),
        BLOCK_N=block_n,
        FP8_MAX=float(torch.finfo(torch.float8_e4m3fn).max),
        TINY=float(torch.finfo(torch.float32).tiny),
        # gemma_rmsnorm._num_warps (its sum order)
        num_warps=16 if block_n >= 4096 else (8 if block_n >= 1024 else 4),
    )
    return out, res_out, q, scale
