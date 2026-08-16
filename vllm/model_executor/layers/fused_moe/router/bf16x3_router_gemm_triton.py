# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton BF16x3 MoE router GEMM for ROCm.

FP32 router weights are split into three BF16 terms and multiplied with BF16
activations using FP32 accumulation.
"""

import torch

from vllm.triton_utils import tl, triton

BF16X3_TERMS = 3
_MAX_N = 256
_MAX_OFFSET = (1 << 31) - 1
_LDS_BUDGET = 144 << 10


@triton.jit
def _bf16x3_router_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    stride_am,
    stride_cm,
    K: tl.constexpr,
    N: tl.constexpr,
    TERMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute ``a @ sum(b).T`` with FP32 accumulation."""
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, N)
    m_mask = offs_m < M
    acc = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :],
            mask=m_mask[:, None],
            other=0.0,
        )
        for t in tl.static_range(TERMS):
            b = tl.load(b_ptr + t * N * K + offs_n[:, None] * K + offs_k[None, :])
            acc += tl.dot(a, tl.trans(b))
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :],
        acc,
        mask=m_mask[:, None],
    )


def split_bf16x3(weight: torch.Tensor) -> torch.Tensor:
    """Split an FP32 weight into three BF16 residual terms."""
    assert weight.dtype == torch.float32, weight.dtype
    terms = []
    resid = weight.detach().contiguous().clone()
    for _ in range(BF16X3_TERMS):
        hi = resid.to(torch.bfloat16)
        terms.append(hi)
        resid -= hi.to(torch.float32)
    return torch.stack(terms).contiguous()


def is_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether this kernel can serve ``x @ weight.T`` -> fp32 profitably."""
    if weight.dim() != 2 or x.dim() != 2:
        return False
    n, k = weight.shape
    m = x.shape[0]
    cfg = _config(m, n)
    return (
        x.dtype == torch.bfloat16
        and weight.dtype == torch.float32
        and x.shape[1] == k
        and x.stride(1) == 1
        # Reject expanded or overlapping rows.
        and x.stride(0) >= k
        and 0 < n <= _MAX_N
        and n & (n - 1) == 0
        and m * x.stride(0) <= _MAX_OFFSET
        and cfg is not None
        and k % cfg[1] == 0
    )


# The FP32 fallback is faster for smaller batches.
MIN_TOKENS = 8192


def _config(m: int, n: int) -> tuple[int, int, int, int] | None:
    """Return the launch configuration, or None for unsupported shapes."""
    if m < MIN_TOKENS:
        return None
    block_m = 128 if m >= 16384 else 64
    block_k = 128
    # Reduce BLOCK_K until the double-buffered tiles fit in LDS.
    while block_k > 32 and 4 * block_k * (block_m + n) > _LDS_BUDGET:
        block_k //= 2
    return block_m, block_k, 8, 2


def bf16x3_router_gemm(
    x: torch.Tensor,
    weight_split: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Router logits in fp32 from bf16 activations and a bf16x3 split weight."""
    m, k = x.shape
    _, n, k_w = weight_split.shape
    assert k == k_w, f"K mismatch: x has {k}, weight_split has {k_w}"
    if out is None:
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    cfg = _config(m, n)
    assert cfg is not None, f"M={m} is below MIN_TOKENS={MIN_TOKENS}"
    block_m, block_k, num_warps, num_stages = cfg
    _bf16x3_router_gemm_kernel[(triton.cdiv(m, block_m),)](
        x,
        weight_split,
        out,
        m,
        x.stride(0),
        out.stride(0),
        K=k,
        N=n,
        TERMS=BF16X3_TERMS,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
