# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm BF16x3 MoE router GEMM for fp32 gate weights.

Gates whose weights ship in fp32 (MiniMax-M2/M3, HunYuan-V3) otherwise land in
the ``F.linear`` fallback, which casts the bf16 activations up to fp32 and runs
the GEMM on the fp32 MFMA pipe. Splitting the weight into three bf16 terms
whose sum reproduces it exactly moves the GEMM onto the bf16 pipe and is more
accurate than the path it replaces, which rounds the activations on the way to
fp32 while this one leaves them untouched.

The terms are stacked along the expert dimension into one ``(TERMS * E, K)``
weight, so the router costs a single hipBLASLt GEMM plus a reduction over the
``TERMS`` column blocks of its output.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

__all__ = [
    "BF16X3_TERMS",
    "MIN_TOKENS",
    "bf16x3_router_gemm",
    "is_supported",
    "platform_supported",
    "split_bf16x3",
]

BF16X3_TERMS = 3

# Below this the fp32 fallback is faster: the GEMM plus reduction has a ~32us
# floor from launch and allocation that small batches cannot amortize.
# Measured on gfx950; the crossover is 1536 at (K=6144, E=128) and (K=3072,
# E=256), and this leaves margin for untested shapes.
MIN_TOKENS = 2048

_MAX_INT32 = (1 << 31) - 1


def split_bf16x3(weight: torch.Tensor) -> torch.Tensor:
    """Split an fp32 weight into bf16 terms whose sum reproduces it exactly.

    Returns a contiguous ``(BF16X3_TERMS, E, K)`` tensor. Three
    round-to-nearest bf16 terms carry 24 mantissa bits, matching fp32's 24, so
    the reconstruction is bit-exact for finite normals down to ``2**-108``.

    Raises ValueError on non-finite input: ``inf`` rounds to bf16 ``inf`` and
    ``inf - inf`` is NaN, which would poison every logit rather than just the
    affected experts.
    """
    if weight.dtype != torch.float32:
        raise ValueError(f"expected fp32 weight, got {weight.dtype}")
    if not torch.isfinite(weight).all():
        raise ValueError("router weight contains inf/NaN")
    terms = []
    resid = weight.detach().clone()
    for _ in range(BF16X3_TERMS):
        hi = resid.to(torch.bfloat16)
        terms.append(hi)
        resid -= hi.to(torch.float32)
    return torch.stack(terms).contiguous()


def platform_supported() -> bool:
    """Whether this platform has a benchmarked bf16x3 router path.

    gfx942 has a different fp32:bf16 MFMA ratio and is not yet evaluated.
    """
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


def is_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether this path can serve ``x @ weight.T`` -> fp32 profitably."""
    if x.dim() != 2 or weight.dim() != 2:
        return False
    n, k = weight.shape
    m = x.shape[0]
    return (
        m >= MIN_TOKENS
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.float32
        and x.shape[1] == k
        # The weight may be offloaded while the activations are not; the
        # fallback this guards would then fail on a device mismatch.
        and x.device == weight.device
        # torch.mm accepts any stride, but a non-K-contiguous activation costs
        # a hidden copy that gives back the win over the fallback.
        and x.stride(1) == 1
        and 0 < m * BF16X3_TERMS * n <= _MAX_INT32
    )


@triton.jit
def _reduce_terms_kernel(
    y_ptr,
    out_ptr,
    M,
    stride_ym,
    stride_om,
    N: tl.constexpr,
    TERMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for t in tl.static_range(TERMS):
        acc += tl.load(
            y_ptr + offs_m[:, None] * stride_ym + t * N + offs_n[None, :],
            mask=mask,
            other=0.0,
        )
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :], acc, mask=mask)


def bf16x3_router_gemm(
    x: torch.Tensor,
    weight_split: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Router logits in fp32 from bf16 activations and a bf16x3 split weight."""
    m, k = x.shape
    terms, n, k_w = weight_split.shape
    assert terms == BF16X3_TERMS, f"expected {BF16X3_TERMS} terms, got {terms}"
    assert k == k_w, f"K mismatch: x has {k}, weight_split has {k_w}"
    assert weight_split.dtype == torch.bfloat16, weight_split.dtype
    assert weight_split.is_contiguous(), "weight_split must be contiguous"
    assert x.dtype == torch.bfloat16, x.dtype
    if out is None:
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    else:
        assert out.shape == (m, n) and out.is_contiguous(), "bad out layout"

    # Rows [t*E, (t+1)*E) of the (TERMS * E, K) view hold term t, so the GEMM
    # emits one column block per term and the reduction sums them.
    y = torch.empty((m, terms * n), dtype=torch.float32, device=x.device)
    torch.mm(x, weight_split.view(terms * n, k).T, out_dtype=torch.float32, out=y)

    # 64 rows/program keeps the accumulator small enough to stay in registers
    # at the largest supported expert count.
    block_m = 64
    _reduce_terms_kernel[(triton.cdiv(m, block_m),)](
        y,
        out,
        m,
        y.stride(0),
        out.stride(0),
        N=n,
        TERMS=terms,
        BLOCK_M=block_m,
        BLOCK_N=triton.next_power_of_2(n),
        num_warps=4,
    )
    return out
