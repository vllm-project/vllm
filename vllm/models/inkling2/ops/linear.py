# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""bf16 linears: small batches on FlashInfer's CuTe DSL low-M kernels, and the
reference `linear` with fp32 accumulation in ordered 1024-wide K groups."""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.flashinfer import is_flashinfer_autotuning

# Up to this many rows, FlashInfer's CuTe DSL low-M kernels run near HBM
# bandwidth (cuBLAS is up to 2x slower); larger batches use cuBLAS.
LOW_M_MAX_TOKENS = 32


def linear(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """bf16 `x @ weight.T` with fp32 accumulation (into `out` if given,
    batches over LOW_M_MAX_TOKENS only)."""
    if out is not None:
        assert x.shape[0] > LOW_M_MAX_TOKENS
        return torch.mm(x, weight.t(), out=out)
    # Autotuning would profile the CuTe DSL kernels past their 32-row limit.
    if x.shape[0] > LOW_M_MAX_TOKENS or is_flashinfer_autotuning():
        return torch.nn.functional.linear(x, weight)
    from flashinfer import mm_bf16

    return mm_bf16(x, weight.t(), pdl=True, backend="cute-dsl")


@triton.jit
def _k_group_tile(
    X,
    W,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    SPLIT: tl.constexpr,
    pid_n,
    pid_m,
    pid_k,
):
    # Row tiles of at most 64 keep tl.dot on the same mma.sync instruction
    # sequence as the reference's 16x32x32 tiling. With SPLIT, each program computes one
    # K group's partial; otherwise it sums the groups in order.
    cols = pid_n * BN + tl.arange(0, BN)
    rows = pid_m * BM + tl.arange(0, BM)
    ks = tl.arange(0, 128)
    first = pid_k if SPLIT else 0
    NUM_GROUPS: tl.constexpr = 1 if SPLIT else (K + 1023) // 1024
    total = tl.zeros((BM, BN), tl.float32)
    for g in tl.static_range(NUM_GROUPS):
        group = first + g
        partial = tl.zeros((BM, BN), tl.float32)
        for tile in range(8):
            k = group * 1024 + tile * 128 + ks
            x = tl.load(
                X + rows[:, None] * K + k[None, :],
                (rows[:, None] < M) & (k[None, :] < K),
                other=0,
            )
            w = tl.load(
                W + cols[None, :] * K + k[:, None],
                (cols[None, :] < N) & (k[:, None] < K),
                other=0,
            )
            partial = tl.dot(x, w, partial)
        total = partial if g == 0 else total + partial
    tl.store(
        Y + (first * M + rows[:, None]) * N + cols[None, :],
        total.to(Y.dtype.element_ty),
        (rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _prefetch_k_group_tile(
    W, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, pid_n, pid_k
):
    # Pulls one K group's weight tile into L2, one 128-byte line per element.
    cols = pid_n * BN + tl.arange(0, BN)
    k = pid_k * 1024 + tl.arange(0, 16) * 64
    mask = (cols[:, None] < N) & (k[None, :] < K)
    tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.ne.b32 p, $2, 0; "
        "@p prefetch.global.L2 [$1]; mov.b32 $0, 0; }",
        "=r,l,r",
        [W + cols[:, None] * K + k[None, :], mask.to(tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _k_group_kernel(
    X,
    W,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    SPLIT: tl.constexpr,
):
    if SPLIT and tl.program_id(1) == 0:
        _prefetch_k_group_tile(W, N, K, BN, tl.program_id(0), tl.program_id(2))
    tl.extra.cuda.gdc_wait()
    _k_group_tile(
        X,
        W,
        Y,
        M,
        N,
        K,
        BM,
        BN,
        SPLIT,
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )


@triton.jit
def _k_group_sum_kernel(Partials, Y, MN, GROUPS: tl.constexpr, BLOCK: tl.constexpr):
    tl.extra.cuda.gdc_wait()
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    total = tl.load(Partials + offs, mask)
    for group in range(1, GROUPS):
        total = total + tl.load(Partials + group * MN + offs, mask)
    tl.store(Y + offs, total, mask)


@triton.jit
def _k_group_tma_kernel(
    XDesc,
    WDesc,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    LAUNCH_DEPENDENTS: tl.constexpr,
):
    # One output tile over all of K: each group's partial accumulates on the
    # tensor cores and is added to the running sum in order.
    TILES: tl.constexpr = (K + BK - 1) // BK
    GROUP_TILES: tl.constexpr = 1024 // BK
    rows = tl.program_id(1) * BM
    cols = tl.program_id(0) * BN
    tl.extra.cuda.gdc_wait()
    if LAUNCH_DEPENDENTS:
        tl.extra.cuda.gdc_launch_dependents()
    total = tl.zeros((BM, BN), tl.float32)
    partial = tl.zeros((BM, BN), tl.float32)
    for tile in tl.range(TILES, warp_specialize=True):
        x = XDesc.load([rows, tile * BK])
        w = WDesc.load([cols, tile * BK])
        partial = tl.dot(x, w.T, partial)
        if tile % GROUP_TILES == GROUP_TILES - 1 or tile == TILES - 1:
            total = total + partial
            partial = tl.zeros((BM, BN), tl.float32)
    r = rows + tl.arange(0, BM)[:, None]
    c = cols + tl.arange(0, BN)[None, :]
    tl.store(Y + r * N + c, total.to(Y.dtype.element_ty), (r < M) & (c < N))


# From this many rows, 128-row TMA tiles over all of K fill the GPU.
TMA_MIN_TOKENS = 2048


def k_group_partials(
    x: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype = torch.float32,
    launch_dependents: bool = False,
) -> torch.Tensor:
    """fp32 partials [G, T, N] of `x @ weight.T` whose in-order sum over G is
    the reference `linear`. G is 1 (already summed, in `out_dtype`) when
    there are enough rows to fill the GPU without splitting K. With
    `launch_dependents`, the next kernel may start (and must wait) while
    the TMA kernel runs."""
    assert x.is_contiguous() and weight.is_contiguous()
    tokens, hidden = x.shape
    n = weight.shape[0]
    if tokens >= TMA_MIN_TOKENS:
        from triton.tools.tensor_descriptor import TensorDescriptor

        out = torch.empty((1, tokens, n), dtype=out_dtype, device=x.device)
        block_n = 64 if tokens < 4096 else 128
        _k_group_tma_kernel[(triton.cdiv(n, block_n), triton.cdiv(tokens, 128))](
            TensorDescriptor.from_tensor(x, [128, 128]),
            TensorDescriptor.from_tensor(weight, [block_n, 128]),
            out,
            tokens,
            n,
            hidden,
            128,
            block_n,
            128,
            launch_dependents,
            num_warps=4 if block_n == 64 else 8,
            num_stages=4 if block_n == 64 else 3,
            enable_fp_fusion=False,
            launch_pdl=True,
        )
        return out
    block_m = 16 if tokens <= 128 else 64
    block_n = 32 if tokens <= 16 else 64 if tokens <= 256 else 128
    groups = triton.cdiv(hidden, 1024)
    split = groups > 1 and tokens < 4096
    num_warps = 4 if block_n <= 64 else 8
    num_stages = 4 if not split else 3 if block_n <= 64 else 2
    out = torch.empty(
        (groups if split else 1, tokens, n),
        dtype=torch.float32 if split else out_dtype,
        device=x.device,
    )
    grid = (
        triton.cdiv(n, block_n),
        triton.cdiv(tokens, block_m),
        groups if split else 1,
    )
    _k_group_kernel[grid](
        x,
        weight,
        out,
        tokens,
        n,
        hidden,
        block_m,
        block_n,
        split,
        num_warps=num_warps,
        num_stages=num_stages,
        enable_fp_fusion=False,
        launch_pdl=True,
    )
    return out


def linear_k_groups(
    x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """`x @ weight.T` for bf16 `x` [T, K] and `weight` [N, K].

    Each 1024-wide K group is accumulated in fp32 and the partials are summed in
    order, as in the reference `linear`.
    """
    partials = k_group_partials(x, weight, out_dtype)
    if partials.shape[0] == 1:
        return partials[0]
    out = torch.empty(partials.shape[1:], dtype=out_dtype, device=x.device)
    _k_group_sum_kernel[(triton.cdiv(out.numel(), 1024),)](
        partials, out, out.numel(), partials.shape[0], 1024, launch_pdl=True
    )
    return out
