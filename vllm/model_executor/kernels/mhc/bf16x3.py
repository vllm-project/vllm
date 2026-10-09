# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Experimental compensated mHC projection, targeting Hopper.

Adapted from SGLang's hc_mix_stats_bf16x3 (PRs #39664 and #41251).
This primitive is not selected by serving dispatch pending GPU validation.
"""

import torch

from vllm.triton_utils import tl, triton


@torch.no_grad()
def split_bf16_mhc_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare three BF16 components of an FP32 mHC weight outside the hot path.

    The caller owns these derived weights and must rebuild them after updates.
    This reduces BF16 conversion error; it does not make the GEMM bit-exact.
    """
    assert weight.ndim == 2 and weight.dtype == torch.float32
    assert weight.is_contiguous()
    high = weight.bfloat16()
    residual = weight - high.float()
    middle = residual.bfloat16()
    low = (residual - middle.float()).bfloat16()
    return high, middle, low


@triton.jit(do_not_specialize=["M"])
def _mhc_prenorm_bf16x3_kernel(
    X,
    W_HIGH,
    W_MIDDLE,
    W_LOW,
    MIX,
    SQUARE_SUM,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    K_PER_SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    split = tl.program_id(1)
    ks = split * K_PER_SPLIT + tl.arange(0, BLOCK_K)
    high = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    middle = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    low = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    square_sum = tl.zeros((BLOCK_M,), tl.float32)
    for block in range(K_PER_SPLIT // BLOCK_K):
        k = ks + block * BLOCK_K
        x = tl.load(
            X + rows[:, None].to(tl.int64) * K + k[None, :],
            (rows[:, None] < M) & (k[None, :] < K),
            0,
        )
        offsets = cols[None, :].to(tl.int64) * K + k[:, None]
        mask = (cols[None, :] < N) & (k[:, None] < K)
        w_high = tl.load(W_HIGH + offsets, mask, 0)
        w_middle = tl.load(W_MIDDLE + offsets, mask, 0)
        w_low = tl.load(W_LOW + offsets, mask, 0)
        high = tl.dot(x, w_high, high)
        middle = tl.dot(x, w_middle, middle)
        low = tl.dot(x, w_low, low)
        xf = x.to(tl.float32)
        square_sum += tl.sum(xf * xf, axis=1)
    output_rows = split.to(tl.int64) * M + rows
    tl.store(
        MIX + output_rows[:, None] * N + cols[None, :],
        (high + middle) + low,
        (rows[:, None] < M) & (cols[None, :] < N),
    )
    tl.store(SQUARE_SUM + output_rows, square_sum, rows < M)


def mhc_prenorm_bf16x3(
    x: torch.Tensor,
    weight_parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    mix: torch.Tensor,
    square_sum: torch.Tensor,
    *,
    block_m: int = 64,
) -> None:
    """Write split-K projection and squared-norm partials without allocation.

    Args:
        x: Contiguous BF16 input, [M, K].
        weight_parts: Three contiguous BF16 weights, each [N, K].
        mix: FP32 output [splits, M, N]; sum dim 0 for x @ weight.T.
        square_sum: FP32 output [splits, M]; sum dim 0 for squared norms.
        block_m: Token tile, exposed for offline Hopper tuning.

    Outputs follow the existing mHC prenorm epilogue's layout. Different split
    counts can change rounding; batch invariance is not promised.

    """
    assert x.ndim == 2 and x.dtype == torch.bfloat16
    assert x.is_cuda and x.is_contiguous()
    assert len(weight_parts) == 3
    m, k = x.shape
    n = weight_parts[0].shape[0]
    assert k > 0 and k % 64 == 0 and 0 < n <= 32
    assert all(
        w.shape == (n, k)
        and w.dtype == torch.bfloat16
        and w.device == x.device
        and w.is_contiguous()
        for w in weight_parts
    )
    assert mix.ndim == 3
    splits = mix.shape[0]
    assert splits in (1, 4, 16) and block_m in (32, 64, 128)
    assert mix.shape == (splits, m, n) and square_sum.shape == (splits, m)
    assert all(
        t.dtype == torch.float32 and t.device == x.device and t.is_contiguous()
        for t in (mix, square_sum)
    )
    if m == 0:
        return
    _mhc_prenorm_bf16x3_kernel[(triton.cdiv(m, block_m), splits)](
        x,
        *weight_parts,
        mix,
        square_sum,
        m,
        K=k,
        N=n,
        K_PER_SPLIT=triton.cdiv(k, splits * 64) * 64,
        BLOCK_M=block_m,
        BLOCK_N=max(16, triton.next_power_of_2(n)),
        BLOCK_K=64,
        num_warps=4,
        num_stages=3,
    )
