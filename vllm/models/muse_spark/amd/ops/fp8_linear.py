# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 W8A8 / W8A16 GEMMs of `[N, K]` FP8 weights with per-output-channel scales,
for Muse Spark's dense decoder linears on ROCm gfx950.

The kernel is a table lookup on the static shape and the M bucket (tuned for the
TP4 shapes on MI355X, cold weights): a Triton one-launch W8A8 / W8A16 kernel
(`sk1`), Triton split-K plus a scaled sum (`sk`), or row-wise `torch._scaled_mm`.
Other shapes, and M above the table, use `torch._scaled_mm`. The lookup runs
inside the custom op `torch.ops.vllm.muse_fp8_linear` (CUDA-graph and
`torch.compile` safe, no host syncs). A BF16 activation runs W8A16 where the
table has it, else per-token quantization plus W8A8.
"""

from collections.abc import Sequence
from typing import Any

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

F8 = torch.float8_e4m3fn

# (N, K) per TP4 rank -> shape name
SHAPES = {
    (4608, 8192): "qkvg",
    (4096, 8192): "pooler",
    (8192, 4096): "unpooler",
}
# W8A8 kernel per M bucket, tuned on MI355X: (max M, kind, cfg).
# cfg = (BN, BK, SK, warps, stages); BM comes from M.
W8A8_TABLE = {
    "qkvg": [
        (8, "sk", (32, 256, 8, 4, 2)),
        (40, "sk1", (32, 512, 1, 8, 3)),
        (48, "sk1", (32, 512, 1, 4, 3)),
        (64, "sk1", (32, 512, 1, 4, 3)),
        (104, "sk", (128, 256, 4, 4, 2)),
        (128, "sk", (128, 256, 4, 4, 2)),
        (192, "scaled_mm", None),
        (256, "sk1", (64, 256, 1, 4, 3)),
        (320, "sk1", (64, 256, 1, 8, 3)),
    ],
    "pooler": [
        (8, "sk1", (16, 1024, 1, 4, 3)),
        (40, "sk", (32, 256, 4, 4, 2)),
        (48, "sk1", (16, 512, 1, 4, 3)),
        (64, "sk1", (16, 512, 1, 4, 3)),
        (104, "sk", (64, 256, 4, 4, 2)),
        (128, "sk", (64, 256, 4, 4, 2)),
        (192, "sk1", (32, 256, 1, 4, 3)),
        (256, "scaled_mm", None),
        (320, "sk1", (64, 256, 1, 8, 3)),
    ],
    "unpooler": [
        (8, "sk1", (32, 512, 1, 4, 3)),
        (40, "sk1", (32, 256, 1, 4, 3)),
        (48, "sk1", (32, 512, 1, 4, 3)),
        (64, "sk1", (32, 512, 1, 4, 3)),
        (104, "sk1", (32, 256, 1, 8, 3)),
        (128, "sk1", (32, 256, 1, 4, 3)),
        (192, "sk1", (64, 256, 1, 8, 3)),
        (256, "sk1", (64, 256, 1, 8, 3)),
        (320, "scaled_mm", None),
    ],
}
# W8A16 (BF16 activation, no quant) where it beats quant + W8A8: the unpooler at M <= 64
W8A16_TABLE = {
    "unpooler": [
        (8, "sk1", (64, 512, 1, 4, 3)),
        (40, "sk1", (32, 256, 1, 4, 3)),
        (48, "sk1", (32, 256, 1, 4, 3)),
        (64, "sk1", (32, 256, 1, 8, 3)),
    ],
}
LARGE_M_KIND = "scaled_mm"  # M above the table (prefill chunks, Ulysses shards)


def _bm(m: int) -> int:
    return min(128, max(16, 1 << (m - 1).bit_length()))


def route(table: Sequence[tuple[int, str, Any]], m: int) -> tuple[int | None, str, Any]:
    """The `(bucket max M, kernel kind, cfg)` of the first bucket holding `m`."""
    for max_m, kind, cfg in table:
        if m <= max_m:
            return max_m, kind, cfg
    return None, LARGE_M_KIND, None


@triton.jit
def _sk1_kernel(
    X,
    W,
    WS,
    XS,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    A8: tl.constexpr,
):
    """Y[M, N] = (X * xs) @ (W * ws)^T, one launch, no split-K. W FP8 [N, K] + FP32 row
    scale; X FP8 with FP32 per-row scale (A8, FP8 MFMA) or BF16 (W tile up-converted,
    BF16 MFMA).
    """
    pn, pm = tl.program_id(0), tl.program_id(2)
    m = pm * BM + tl.arange(0, BM)
    kk = tl.arange(0, BK)
    n = pn * BN + tl.arange(0, BN)
    bp = W + n[None, :].to(tl.int64) * K + kk[:, None]
    ap = X + m[:, None].to(tl.int64) * K + kk[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(K // BK):
        a = tl.load(ap, mask=m[:, None] < M, other=0.0)
        if A8:
            b = tl.load(bp, cache_modifier=".cg")
        else:
            b = tl.load(bp, cache_modifier=".cg").to(tl.bfloat16)
        acc = tl.dot(a, b, acc)
        ap += BK
        bp += BK
    yv = acc * tl.load(WS + n)[None, :]
    if A8:
        yv = yv * tl.load(XS + m, mask=m < M, other=0.0)[:, None]
    tl.store(
        Y + m[:, None] * N + n[None, :], yv.to(Y.dtype.element_ty), mask=m[:, None] < M
    )


@triton.jit
def _sk_partial_kernel(
    X,
    W,
    P,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SK: tl.constexpr,
):
    """Split-K FP8 x FP8 partials (FP32) [SK, M, N]."""
    pn, pk = tl.program_id(0), tl.program_id(1)
    KT_PER: tl.constexpr = K // BK // SK
    m = tl.program_id(2) * BM + tl.arange(0, BM)
    kk = tl.arange(0, BK)
    n = pn * BN + tl.arange(0, BN)
    bp = W + n[None, :].to(tl.int64) * K + (pk * KT_PER * BK + kk)[:, None]
    ap = X + m[:, None].to(tl.int64) * K + (pk * KT_PER * BK + kk)[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(KT_PER):
        a = tl.load(ap, mask=m[:, None] < M, other=0.0)
        b = tl.load(bp, cache_modifier=".cg")
        acc = tl.dot(a, b, acc)
        ap += BK
        bp += BK
    tl.store(P + pk * M * N + m[:, None] * N + n[None, :], acc, mask=m[:, None] < M)


@triton.jit
def _sk_sum_kernel(P, XS, WS, Y, M, N: tl.constexpr, SK: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    ok = i < M * N
    s = tl.arange(0, SK)
    p = tl.load(P + s[:, None] * M * N + i[None, :], mask=ok[None, :], other=0.0)
    y = (
        tl.sum(p, axis=0)
        * tl.load(XS + i // N, mask=ok, other=0.0)
        * tl.load(WS + i % N, mask=ok, other=0.0)
    )
    tl.store(Y + i, y.to(Y.dtype.element_ty), mask=ok)


def _run_sk1(x, xs, wq, ws, cfg, a8: bool):
    m, k = x.shape
    n = wq.shape[0]
    bn, bk, _, warps, stages = cfg
    bm = _bm(m)
    y = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    _sk1_kernel[(n // bn, 1, triton.cdiv(m, bm))](
        x,
        wq,
        ws,
        xs if a8 else ws,
        y,
        m,
        n,
        k,
        bm,
        bn,
        bk,
        a8,
        num_warps=warps,
        num_stages=stages,
        matrix_instr_nonkdim=16,
    )
    return y


def _run_sk(xq, xs, wq, ws, cfg):
    m, k = xq.shape
    n = wq.shape[0]
    bn, bk, sk, warps, stages = cfg
    bm = _bm(m)
    p = torch.empty((sk, m, n), device=xq.device, dtype=torch.float32)
    _sk_partial_kernel[(n // bn, sk, triton.cdiv(m, bm))](
        xq,
        wq,
        p,
        m,
        n,
        k,
        bm,
        bn,
        bk,
        sk,
        num_warps=warps,
        num_stages=stages,
        matrix_instr_nonkdim=16,
    )
    y = torch.empty((m, n), device=xq.device, dtype=torch.bfloat16)
    _sk_sum_kernel[(triton.cdiv(m * n, 512),)](p, xs, ws, y, m, n, sk, 512, num_warps=4)
    return y


def scaled_mm(xq, xs, wq, ws, out_dtype: torch.dtype = torch.bfloat16):
    """Row-wise scaled FP8 GEMM (hipBLASLt): `xs` holds M scales, `ws` N."""
    return torch._scaled_mm(
        xq,
        wq.t(),
        scale_a=xs.view(-1, 1),
        scale_b=ws.view(1, -1),
        out_dtype=out_dtype,
    )


def quant_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token FP8 of a BF16 `[M, K]` (AITER): `(xq, xs [M, 1] float32)`."""
    return rocm_aiter_ops.per_token_quant(x.contiguous(), F8)


def routes(n: int, k: int) -> str:
    """The kernel per M bucket of an `[N, K]` FP8 weight, for logging."""
    name = SHAPES.get((n, k), "")
    parts = [
        f"{'w8a16' if table is W8A16_TABLE else 'w8a8'} M<={max_m}: {kind}"
        for table in (W8A16_TABLE, W8A8_TABLE)
        for max_m, kind, _ in table.get(name, [])
    ]
    return ", ".join([*parts, f"else {LARGE_M_KIND}"])


def _fp8_linear_impl(
    x: torch.Tensor, xs: torch.Tensor | None, wq: torch.Tensor, ws: torch.Tensor
) -> torch.Tensor:
    m, k = x.shape
    n = wq.shape[0]
    if m == 0:
        return torch.empty((0, n), device=x.device, dtype=torch.bfloat16)
    name = SHAPES.get((n, k), "")
    if x.dtype != F8:
        w16 = W8A16_TABLE.get(name)
        if w16 is not None and m <= w16[-1][0]:
            _, _, cfg = route(w16, m)
            return _run_sk1(x.contiguous(), None, wq, ws, cfg, a8=False)
        x, xs = quant_rows(x)
    if not x.is_contiguous():
        x = x.contiguous()
    _, kind, cfg = route(W8A8_TABLE.get(name, ()), m)
    if kind == "sk1":
        return _run_sk1(x, xs, wq, ws, cfg, a8=True)
    if kind == "sk":
        return _run_sk(x, xs, wq, ws, cfg)
    return scaled_mm(x, xs, wq, ws)


def _fp8_linear_fake(
    x: torch.Tensor, xs: torch.Tensor | None, wq: torch.Tensor, ws: torch.Tensor
) -> torch.Tensor:
    return torch.empty((x.shape[0], wq.shape[0]), device=x.device, dtype=torch.bfloat16)


direct_register_custom_op(
    op_name="muse_fp8_linear",
    op_func=_fp8_linear_impl,
    fake_impl=_fp8_linear_fake,
)


def linear(
    layer: torch.nn.Module, x: torch.Tensor, xs: torch.Tensor | None = None
) -> torch.Tensor:
    """Local (un-reduced) output of an FP8 linear: `x` FP8 with per-row `xs`, or
    BF16.
    """
    shape = x.shape
    if x.dim() != 2:
        x = x.reshape(-1, shape[-1])
    y = torch.ops.vllm.muse_fp8_linear(x, xs, layer.weight, layer.weight_scale)
    return y if len(shape) == 2 else y.view(*shape[:-1], y.shape[-1])
