# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-K BF16 GEMM for few-token activations against a narrow weight.

``y = x @ w.T`` with ``x`` ``[M, K]`` (M small) and ``w`` ``[N, K]``. Each split
accumulates its K range in fp32; a second kernel adds the splits in fixed order,
in fp32, and rounds to BF16 once. No partial sum is ever rounded to BF16, so the
error matches an fp32-accumulating GEMM such as ``F.linear`` (the results differ
from it only through the summation order), and the output is deterministic for
a given M.

A single-kernel "last split reduces" variant is slower on MI355X: the
agent-scope acquire/release it needs flushes and invalidates the per-XCD L2.
"""

import torch

from vllm.triton_utils import tl, triton

# (BLOCK_M, BLOCK_N, BLOCK_K, splits, num_warps)
_Config = tuple[int, int, int, int, int]

# (N, K) -> ((max tokens, *_Config), ...), tuned on MI355X with the weight
# streamed from HBM and the calls replayed from a HIP graph.
_CONFIGS: dict[tuple[int, int], tuple[tuple[int, int, int, int, int, int], ...]] = {
    # Muse Spark pooler, column-sharded over TP8.
    (512, 8192): (
        (16, 16, 16, 256, 16, 4),
        (32, 32, 16, 256, 16, 2),
        (64, 32, 16, 256, 8, 4),
        (128, 16, 64, 256, 8, 4),
    ),
    # Muse Spark unpooler, column-sharded over TP8.
    (1024, 4096): (
        (8, 16, 16, 256, 8, 2),
        (16, 16, 16, 256, 8, 4),
        (32, 32, 16, 256, 8, 2),
        (64, 32, 16, 256, 4, 4),
        (128, 64, 64, 256, 8, 4),
    ),
}


def skinny_gemm_config(n: int, k: int, num_tokens: int) -> _Config | None:
    """The tuned config for ``[num_tokens, k] @ [n, k].T``, or None."""
    for max_tokens, *cfg in _CONFIGS.get((n, k), ()):
        if num_tokens <= max_tokens:
            return tuple(cfg)  # type: ignore[return-value]
    return None


@triton.jit
def _skinny_gemm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    part_ptr,
    M,
    stride_xm,
    stride_ym,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    K_PER_SPLIT: tl.constexpr = K // SPLITS
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for kk in range(0, K_PER_SPLIT, BLOCK_K):
        offs_k = pid_k * K_PER_SPLIT + kk + tl.arange(0, BLOCK_K)
        a = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
            mask=m_mask[:, None],
            other=0.0,
        )
        b = tl.load(w_ptr + offs_n[None, :] * K + offs_k[:, None])
        acc = tl.dot(a, b, acc)
    if SPLITS == 1:
        tl.store(
            y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :],
            acc.to(y_ptr.dtype.element_ty),
            mask=m_mask[:, None],
        )
    else:
        tl.store(
            part_ptr + (pid_k * M + offs_m[:, None]) * N + offs_n[None, :],
            acc,
            mask=m_mask[:, None],
        )


@triton.jit
def _reduce_splits_kernel(
    part_ptr,
    y_ptr,
    M,
    stride_ym,
    N: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M * N
    acc = tl.load(part_ptr + offs, mask=mask, other=0.0)
    for s in tl.static_range(1, SPLITS):
        acc += tl.load(part_ptr + s * M * N + offs, mask=mask, other=0.0)
    tl.store(
        y_ptr + (offs // N) * stride_ym + offs % N,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask,
    )


def skinny_gemm(
    x: torch.Tensor, weight: torch.Tensor, config: _Config | None = None
) -> torch.Tensor:
    """``x @ weight.T`` for 2-D bf16 ``x`` and a contiguous ``[N, K]`` weight.

    ``config`` defaults to the tuned entry, which must exist for this shape.
    """
    assert x.dim() == 2 and x.stride(1) == 1 and weight.is_contiguous()
    m, k = x.shape
    n = weight.shape[0]
    assert weight.shape[1] == k
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if m == 0:
        return out
    if config is None:
        config = skinny_gemm_config(n, k, m)
        assert config is not None, f"no skinny GEMM config for {(m, n, k)}"
    block_m, block_n, block_k, splits, num_warps = config
    # The kernel loads the weight unmasked.
    assert n % block_n == 0 and k % (splits * block_k) == 0
    part = (
        torch.empty((splits, m, n), dtype=torch.float32, device=x.device)
        if splits > 1
        else out
    )
    _skinny_gemm_kernel[(n // block_n, splits, triton.cdiv(m, block_m))](
        x,
        weight,
        out,
        part,
        m,
        x.stride(0),
        out.stride(0),
        N=n,
        K=k,
        SPLITS=splits,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    if splits > 1:
        block = 1024
        _reduce_splits_kernel[(triton.cdiv(m * n, block),)](
            part, out, m, out.stride(0), N=n, SPLITS=splits, BLOCK=block, num_warps=4
        )
    return out
