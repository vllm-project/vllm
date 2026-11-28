# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from sxuff/inkling-2-mini-spark, gemv_q8.py (Apache-2.0),
# blob ef0b1d7acc447c7eba0d07a1292e78d706bf9fb1.
"""BF16 draft projections with INT8 weights and per-output-row FP32 scales."""

import torch

from vllm.triton_utils import tl, triton

MAX_M = 64


@torch.no_grad()
def quantize_rows(
    weight: torch.Tensor,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refill in place when CUDA graphs already reference the quantized weights."""
    n = weight.shape[0]
    if out is None:
        out = (
            weight.new_empty(weight.shape, dtype=torch.int8),
            weight.new_empty((n,), dtype=torch.float32),
        )
    q, scale = out
    for start in range(0, n, 8192):
        rows = slice(start, start + 8192)
        w = weight[rows].float()
        s = w.abs().amax(1).clamp_min(1e-30) / 127
        q[rows].copy_((w / s[:, None]).round_().clamp_(-127, 127))
        scale[rows].copy_(s)
    return out


@triton.jit
def _gemv_q8_kernel(
    X,
    W,
    S,
    B,
    Y,
    M,
    N,
    K,
    SX,
    SW,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
):
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    rows = tl.program_id(1) * MP + tl.arange(0, MP)
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(
            W + cols[None, :].to(tl.int64) * SW + ks[:, None],
            (cols[None, :] < N) & (ks[:, None] < K),
            other=0.0,
        )
        x = tl.load(
            X + rows[:, None] * SX + ks[None, :],
            (rows[:, None] < M) & (ks[None, :] < K),
            other=0,
        )
        acc += tl.dot(x, w.to(tl.bfloat16))
    acc = acc * tl.load(S + cols, cols < N, other=0)[None, :]
    if B is not None:
        acc += tl.load(B + cols, cols < N, other=0).to(tl.float32)[None, :]
    tl.store(
        Y + rows[:, None] * N + cols[None, :],
        acc,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def _config(m: int, n: int, k: int) -> tuple[int, int, int]:
    if m > 16:
        return 32, 256, 2
    if k <= 256:
        return 64, 256, 3
    if n >= 16384:
        return 32, 512, 3
    return 32, 256, 4


def gemv_q8(
    x: torch.Tensor,
    q: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute `x @ q.T` with BF16 products, then apply row scales and bias."""
    x2 = x.reshape(-1, x.shape[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m, k = x2.shape
    n = q.shape[0]
    assert q.shape[1] == k and q.stride(-1) == 1
    out = x.new_empty((*x.shape[:-1], n))
    if m:
        bn, bk, stages = _config(m, n, k)
        mp = 16 if m <= 16 else 32 if m <= 32 else 64
        _gemv_q8_kernel[(triton.cdiv(n, bn), triton.cdiv(m, mp))](
            x2,
            q,
            scale,
            bias,
            out,
            m,
            n,
            k,
            x2.stride(0),
            q.stride(0),
            BN=bn,
            BK=bk,
            MP=mp,
            num_stages=stages,
        )
    return out
