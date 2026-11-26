# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from the inkling-2-mini-gb10 recipe (2026-11-21).
# Adapted for vLLM: INT8-only, configured through Inkling2DSparkConfig.
# Original: sxuff/inkling-2-mini-spark, gemv_q8.py
# Blob: ef0b1d7acc447c7eba0d07a1292e78d706bf9fb1
"""Weight-only INT8 (per-output-row fp32 scale) low-M
projections for SM12x drafts: bf16 `x` [M, K] @ dequant(w) [N, K].T.

The 8-bit values convert exactly to bf16, so each tile runs the same bf16
tensor-core dot as the bf16 GEMV with half the weight bytes; the row scale
(and optional bias) is applied once to the fp32 accumulator.
"""

import torch

from vllm.triton_utils import tl, triton

MAX_M = 64


@torch.no_grad()
def quantize_rows(
    weight: torch.Tensor,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
    chunk: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric quantization of bf16 `weight` [N, K]; refills `out`
    in place when given (CUDA graphs keep pointing at it)."""
    n, k = weight.shape
    if out is None:
        out = (
            torch.empty((n, k), dtype=torch.int8, device=weight.device),
            torch.empty((n,), dtype=torch.float32, device=weight.device),
        )
    q, scale = out
    qmax = 127.0
    for start in range(0, n, chunk):
        rows = weight[start : start + chunk].float()
        s = rows.abs().amax(1).clamp_min(1e-30) / qmax
        v = rows / s[:, None]
        v = v.round_().clamp_(-127, 127)
        q[start : start + chunk].copy_(v.to(q.dtype))
        scale[start : start + chunk].copy_(s)
    return q, scale


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
    SY,
    HAS_BIAS: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
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
    if HAS_BIAS:
        acc += tl.load(B + cols, cols < N, other=0).to(tl.float32)[None, :]
    tl.store(
        Y + rows[:, None] * SY + cols[None, :],
        acc.to(Y.dtype.element_ty),
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def _config(n: int, k: int) -> tuple[int, int, int, int]:
    if k <= 256:
        return 64, 256, 4, 3
    if n >= 16384:
        return 32, 512, 4, 3
    return 32, 256, 4, 4


def gemv_q8(
    x: torch.Tensor,
    q: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """`(x @ (q * scale[:, None]).T + bias)` in `out_dtype` (default x's)."""
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m, k = x2.shape
    n = q.shape[0]
    assert q.shape[1] == k and q.stride(-1) == 1
    out = torch.empty((m, n), device=x.device, dtype=out_dtype or x.dtype)
    if m:
        bn, bk, warps, stages = _config(n, k)
        mp = 16 if m <= 16 else 32 if m <= 32 else 64
        if mp > 16:
            bn, bk, warps, stages = 32, min(bk, 256), 4, 2
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
            out.stride(0),
            HAS_BIAS=bias is not None,
            BN=bn,
            BK=bk,
            MP=mp,
            num_warps=warps,
            num_stages=stages,
        )
    return out.view(*lead, n)
