# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm BF16x3 MoE router GEMM for fp32 gate weights.

Gates whose weights ship in fp32 (MiniMax-M2/M3, HunYuan-V3) otherwise land in
the ``F.linear`` fallback, which casts the bf16 activations up to fp32 and runs
the GEMM on the fp32 MFMA pipe. Splitting the weight into three bf16 terms
whose sum reproduces it exactly moves the GEMM onto the bf16 pipe and is more
accurate than the path it replaces, which rounds the activations on the way to
fp32 while this one leaves them untouched.

Large batches stack the terms along the expert dimension into one
``(TERMS * E, K)`` weight, so the router costs a single hipBLASLt GEMM plus a
reduction over the ``TERMS`` column blocks of its output. That pair has a
launch and allocation floor that decode batches cannot amortize, so smaller
batches instead run a split-K Triton kernel that accumulates the three terms
in registers and sums the K partitions in a fixed order.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

__all__ = [
    "BF16X3_TERMS",
    "MM_MIN_TOKENS",
    "bf16x3_router_gemm",
    "is_supported",
    "platform_supported",
    "split_bf16x3",
]

BF16X3_TERMS = 3

# From this many tokens on the stacked hipBLASLt GEMM serves the router; below
# it the split-K kernel does. The GEMM plus reduction has a ~10us floor from
# launch and allocation that small batches cannot amortize, where the split-K
# kernel is 1.1-1.8x faster than the GEMM and 1.6-2.1x faster than the fp32
# F.linear fallback on gfx950 at (K=6144, E=128) and (K=3072, E=256).
MM_MIN_TOKENS = 2048

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
    """Whether this path can serve ``x @ weight.T`` -> fp32 profitably.

    Batches small enough for the gfx950 fp32 low-M kernel are faster there;
    the caller is expected to try that kernel first.
    """
    if x.dim() != 2 or weight.dim() != 2:
        return False
    n, k = weight.shape
    m = x.shape[0]
    return (
        m > 0
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
        # The split-K kernel indexes both operands with int32 offsets.
        and BF16X3_TERMS * n * k <= _MAX_INT32
        and m * x.stride(0) <= _MAX_INT32
    )


@triton.jit
def _split_k_partials_kernel(
    x_ptr,
    w_ptr,
    partials_ptr,
    M,
    N,
    K,
    stride_xm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_PER_SPLIT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    split = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    # One accumulator per term: the small terms are summed among themselves
    # before they meet the large one.
    acc0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K_PER_SPLIT, BLOCK_K):
        k = split * K_PER_SPLIT + k_start + offs_k
        mask_k = k < K
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        w_offs = offs_n[None, :] * K + k[:, None]
        w_mask = mask_n[None, :] & mask_k[:, None]
        w0 = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)
        w1 = tl.load(w_ptr + N * K + w_offs, mask=w_mask, other=0.0)
        w2 = tl.load(w_ptr + 2 * N * K + w_offs, mask=w_mask, other=0.0)
        acc0 = tl.dot(x, w0, acc0)
        acc1 = tl.dot(x, w1, acc1)
        acc2 = tl.dot(x, w2, acc2)
    tl.store(
        partials_ptr + (split * M + offs_m[:, None]) * N + offs_n[None, :],
        acc0 + (acc1 + acc2),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _split_k_reduce_kernel(
    partials_ptr,
    out_ptr,
    numel,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    # Fixed split order, so the result does not depend on scheduling.
    for split in tl.static_range(SPLIT_K):
        acc += tl.load(partials_ptr + split * numel + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc, mask=mask)


def _split_k_config(m: int, n: int, k: int) -> tuple[int, int, int, int]:
    """Return (BLOCK_M, BLOCK_N, SPLIT_K, num_warps), swept on gfx950.

    Split K while the grid stays within 768 programs and each split keeps at
    least 192 of K. Over M in [64, 2047] and six router shapes this lands
    within 3% of the best split on average and 16% in the worst case.
    """
    block_m = 16 if m <= 16 else 32 if m <= 64 else 64
    block_n = 32
    tiles = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
    split_k = 1
    while split_k * 2 * tiles <= 768 and k // (split_k * 2) >= 192:
        split_k *= 2
    return block_m, block_n, split_k, 4


def _bf16x3_router_gemm_split_k(
    x: torch.Tensor, weight_split: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, k = x.shape
    n = weight_split.shape[1]
    block_m, block_n, split_k, num_warps = _split_k_config(m, n, k)
    block_k = 64
    k_per_split = triton.cdiv(triton.cdiv(k, split_k), block_k) * block_k
    partials = torch.empty((split_k, m, n), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n), split_k)
    _split_k_partials_kernel[grid](
        x,
        weight_split,
        partials,
        m,
        n,
        k,
        x.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        K_PER_SPLIT=k_per_split,
        num_warps=num_warps,
        num_stages=2,
        matrix_instr_nonkdim=16,
    )
    block = 1024
    _split_k_reduce_kernel[(triton.cdiv(m * n, block),)](
        partials, out, m * n, SPLIT_K=split_k, BLOCK=block, num_warps=4
    )
    return out


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
    if m < MM_MIN_TOKENS:
        return _bf16x3_router_gemm_split_k(x, weight_split, out)

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
