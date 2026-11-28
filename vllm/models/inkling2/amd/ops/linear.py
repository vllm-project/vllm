# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""bf16 linears as the reference `linear`: fp32 accumulation in ordered
1024-wide K groups."""

import torch

from vllm.triton_utils import tl, triton


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
    # With SPLIT, each program computes one K group's partial; otherwise it
    # sums the groups in order.
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
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    total = tl.load(Partials + offs, mask)
    for group in range(1, GROUPS):
        total = total + tl.load(Partials + group * MN + offs, mask)
    tl.store(Y + offs, total, mask)


def k_group_partials(
    x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """fp32 partials [G, T, N] of `x @ weight.T` whose in-order sum over G is
    the reference `linear`. G is 1 (already summed, in `out_dtype`) when
    there are enough rows to fill the GPU without splitting K."""
    assert x.is_contiguous() and weight.is_contiguous()
    tokens, hidden = x.shape
    n = weight.shape[0]
    block_m = 16 if tokens <= 128 else 64
    block_n = 32 if tokens <= 16 else 64 if tokens <= 256 else 128
    groups = triton.cdiv(hidden, 1024)
    split = groups > 1 and tokens < 4096
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
        num_warps=4 if block_n <= 64 else 8,
        # gfx950's 160 KB of LDS fits two stages of the largest tiles.
        num_stages=2,
        enable_fp_fusion=False,
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
        partials, out, out.numel(), partials.shape[0], 1024
    )
    return out
