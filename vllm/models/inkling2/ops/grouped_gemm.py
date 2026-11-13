# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 grouped GEMM over expert-sorted rows (the expert LoRA projections)."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _grouped_gemm_kernel(
    X,
    W,
    Y,
    Perm,
    Offsets,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    EXPERTS_POW2: tl.constexpr,
    TOPK: tl.constexpr,
    GATHER: tl.constexpr,
    ACCUMULATE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # Program (n tile, row block): the row block is found from the per-expert
    # block counts, so the grid is a static upper bound and nothing is read
    # back to the host (CUDA graph capture).
    pid = tl.program_id(1)
    experts = tl.arange(0, EXPERTS_POW2)
    starts = tl.load(Offsets + experts, experts < NUM_EXPERTS, other=0)
    ends = tl.load(Offsets + experts + 1, experts < NUM_EXPERTS, other=0)
    blocks = tl.cumsum(tl.cdiv(ends - starts, BM), 0)
    expert = tl.sum((blocks <= pid).to(tl.int32), 0)
    if expert >= NUM_EXPERTS:
        return
    block = pid - tl.max(tl.where(blocks <= pid, blocks, 0), 0)
    rows = (tl.load(Offsets + expert) + block * BM).to(tl.int64) + tl.arange(0, BM)
    row_mask = rows < tl.load(Offsets + expert + 1)
    x_rows = tl.load(Perm + rows, row_mask, other=0) // TOPK if GATHER else rows
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    col_mask = cols < N
    ks = tl.arange(0, BK)
    x_ptrs = X + x_rows[:, None] * K + ks[None, :]
    w_ptrs = W + (expert.to(tl.int64) * N + cols[None, :]) * K + ks[:, None]
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        k_mask = (k + ks) < K
        x = tl.load(x_ptrs, row_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, k_mask[:, None] & col_mask[None, :], other=0.0)
        acc = tl.dot(x, w, acc)
        x_ptrs += BK
        w_ptrs += BK
    y_ptrs = Y + rows[:, None] * N + cols[None, :]
    mask = row_mask[:, None] & col_mask[None, :]
    if ACCUMULATE:
        acc += tl.load(y_ptrs, mask, other=0.0).to(tl.float32)
    tl.store(y_ptrs, acc.to(Y.dtype.element_ty), mask)


def grouped_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    perm: torch.Tensor,
    offsets: torch.Tensor,
    num_rows: int,
    topk: int,
    gather: bool,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """`out[r] (+)= x[r] @ weight[e].T` for this rank's expert-sorted rows.

    Same row contract as `ops.moe.expert_gemm`: rows `[offsets[0],
    offsets[-1])` are this rank's, expert `e`'s rows are `[offsets[e],
    offsets[e + 1])`; with `gather` row `r` reads `x[perm[r] // topk]`.
    `weight` is BF16 `[E, N, K]`. Without `out`, returns a new
    `[num_rows, N]` tensor whose rows outside this rank's range are
    uninitialized; with `out`, adds into it (FP32 accumulation either way).
    """
    num_experts, n, k = weight.shape
    accumulate = out is not None
    if out is None:
        out = x.new_empty(num_rows, n)
    assert out.is_contiguous() and out.shape == (num_rows, n)
    block_m, block_n, block_k = 64, min(128, max(16, triton.next_power_of_2(n))), 64
    # Every non-empty expert holds at least one row.
    max_blocks = triton.cdiv(num_rows, block_m) + min(num_experts, num_rows) - 1
    if max_blocks > 0:
        _grouped_gemm_kernel[(triton.cdiv(n, block_n), max_blocks)](
            x,
            weight,
            out,
            perm,
            offsets,
            n,
            k,
            num_experts,
            triton.next_power_of_2(num_experts),
            topk,
            gather,
            accumulate,
            block_m,
            block_n,
            block_k,
            num_warps=4,
            num_stages=3,
        )
    return out
