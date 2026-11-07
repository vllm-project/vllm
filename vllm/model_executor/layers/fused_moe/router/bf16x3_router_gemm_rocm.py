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
    "bf16x3_router_gemv",
    "gemv_is_supported",
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

# (max_tokens, (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K)), tuned on MI355 at
# K=6144, E=128. At <= 32 tokens the fp32 router kernel is faster.
_GEMV_MIN_TOKENS = 33
_GEMV_CONFIGS = (
    (64, (32, 16, 128, 16)),
    (256, (64, 16, 128, 16)),
    (512, (64, 64, 128, 16)),
    (MIN_TOKENS - 1, (64, 64, 64, 8)),
)


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


def _gemv_config(m: int) -> tuple[int, int, int, int]:
    for max_m, cfg in _GEMV_CONFIGS:
        if m <= max_m:
            return cfg
    raise ValueError(f"no GEMV config for {m} tokens")


def gemv_is_supported(x: torch.Tensor, weight_split: torch.Tensor) -> bool:
    """Whether `bf16x3_router_gemv` can serve ``x`` with this weight split."""
    if x.dim() != 2 or weight_split.dim() != 3:
        return False
    m, k = x.shape
    if not _GEMV_MIN_TOKENS <= m < MIN_TOKENS:
        return False
    _, block_n, block_k, split_k = _gemv_config(m)
    terms, n, k_w = weight_split.shape
    return (
        terms == BF16X3_TERMS
        and k == k_w
        and x.dtype == torch.bfloat16
        and weight_split.dtype == torch.bfloat16
        and weight_split.is_contiguous()
        and x.device == weight_split.device
        and x.stride(1) == 1
        and n % block_n == 0
        and k % block_k == 0
        and split_k * m * n <= _MAX_INT32
    )


@triton.jit
def _gemv_kernel(
    x_ptr,
    w_ptr,
    partials_ptr,
    M,
    stride_xm,
    K: tl.constexpr,
    N: tl.constexpr,
    TERMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = (offs_m < M)[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(pid_k * BLOCK_K, K, BLOCK_K * SPLIT_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        xv = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
            mask=mask_m,
            other=0.0,
        )
        for t in tl.static_range(TERMS):
            wv = tl.load(w_ptr + t * N * K + offs_n[:, None] * K + offs_k[None, :])
            acc += tl.dot(xv, tl.trans(wv))
    tl.store(
        partials_ptr + pid_k * M * N + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=mask_m,
    )


@triton.jit
def _reduce_splits_kernel(
    partials_ptr,
    out_ptr,
    numel,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    total = tl.zeros((BLOCK,), tl.float32)
    for s in tl.static_range(SPLIT_K):
        total += tl.load(partials_ptr + s * numel + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, total, mask=mask)


def bf16x3_router_gemv(x: torch.Tensor, weight_split: torch.Tensor) -> torch.Tensor:
    """Split-K variant for fewer than MIN_TOKENS tokens. Deterministic: splits
    are summed in a fixed order, not with atomics."""
    m, k = x.shape
    terms, n, _ = weight_split.shape
    block_m, block_n, block_k, split_k = _gemv_config(m)
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    partials = torch.empty((split_k, m, n), dtype=torch.float32, device=x.device)
    _gemv_kernel[(n // block_n, triton.cdiv(m, block_m), split_k)](
        x,
        weight_split,
        partials,
        m,
        x.stride(0),
        K=k,
        N=n,
        TERMS=terms,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        SPLIT_K=split_k,
        num_warps=4,
        num_stages=2,
    )
    reduce_block = 1024
    _reduce_splits_kernel[(triton.cdiv(m * n, reduce_block),)](
        partials,
        out,
        m * n,
        SPLIT_K=split_k,
        BLOCK=reduce_block,
        num_warps=4,
    )
    return out
