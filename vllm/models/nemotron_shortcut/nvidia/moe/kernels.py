# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton expert GEMMs for Nemotron Shortcut's clamped ReLU² experts.

`routed_experts` reproduces the training numerics: the clamp, the ReLU², and
the routing weight are applied after the up projection and rounded to the
activation dtype before the down projection, and the top-k sum is kept in
FP32. Both projections optionally read MXFP8 weights and quantize their
activations in-kernel.
"""

import torch

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
)
from vllm.triton_utils import tl, tldevice, triton


@triton.jit
def mxfp8_quantize_block(x):
    """Quantize rows of 32 values to E4M3 with E8M0 scales."""
    x = x.to(tl.float32)
    # The scale puts the block amax at the top of the E4M3 range rather than
    # at 1.0, or small elements of the block end up in the subnormals.
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1.1754943508222875e-38)
    sb = tl.ceil(tl.log2(amax / 448.0)) + 127.0
    sb = tl.minimum(tl.maximum(sb, 0.0), 254.0)
    # Scale by the reciprocal rather than dividing by exp2(sb - 127): for an
    # all-zero block sb == 0 makes that divisor subnormal in FP32.
    rescale = tl.exp2(127.0 - sb)
    xq = (x * rescale[:, None]).to(tl.float8e4nv)
    return xq, sb.to(tl.uint8)


@triton.jit
def _expert_mm(
    x,
    weight,
    weight_scale,
    ids,
    routing,
    output,
    TOP: tl.constexpr,
    IN: tl.constexpr,
    OUT: tl.constexpr,
    FIRST: tl.constexpr,
    BEGIN: tl.constexpr,
    END: tl.constexpr,
    CLAMP: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MX_BLOCK_SIZE: tl.constexpr,
    MXFP8: tl.constexpr,
):
    # Smaller M tiles make Triton emulate dot_scaled with BF16 on SM100.
    BM: tl.constexpr = 128 if MXFP8 else 16
    # int64: with DP, every rank sees all ranks' tokens, so offsets such as
    # assignment * OUT can pass 2**31.
    assignment, tile = tl.program_id(0).to(tl.int64), tl.program_id(1)
    token = assignment // TOP
    expert = tl.load(ids + assignment)
    # int64: the weight offset local * OUT * IN passes 2**31 when one GPU holds
    # many experts (few EP ranks).
    local = (expert - BEGIN).to(tl.int64)
    valid = (expert >= BEGIN) & (expert < END)
    rows = tl.arange(0, BM)
    cols = tile * BN + tl.arange(0, BN)
    reduction = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    row = token if FIRST else assignment
    for block in range(tl.cdiv(IN, BK)):
        k = block * BK + reduction
        a = tl.load(
            x + row * IN + k[None, :],
            (rows[:, None] == 0) & (k[None, :] < IN) & valid,
            other=0,
        )
        b = tl.load(
            weight + (local * OUT + cols[None, :]) * IN + k[:, None],
            (cols[None, :] < OUT) & (k[:, None] < IN) & valid,
            other=0.0,
        )
        if MXFP8:
            # Quantize each activation block at the input of its projection.
            aq, a_scale = mxfp8_quantize_block(
                a.reshape(BM * (BK // MX_BLOCK_SIZE), MX_BLOCK_SIZE)
            )
            ks = block * (BK // MX_BLOCK_SIZE) + tl.arange(0, BK // MX_BLOCK_SIZE)
            bs = tl.load(
                weight_scale
                + (local * OUT + cols[:, None]) * (IN // MX_BLOCK_SIZE)
                + ks[None, :],
                (cols[:, None] < OUT) & (ks[None, :] < IN // MX_BLOCK_SIZE) & valid,
                other=0,
            )
            acc = tl.dot_scaled(
                aq.reshape(BM, BK),
                a_scale.reshape(BM, BK // MX_BLOCK_SIZE),
                "e4m3",
                b,
                bs,
                "e4m3",
                acc,
            )
        else:
            acc = tl.dot(a, b, acc, input_precision="ieee")
    value = acc.to(x.dtype.element_ty).to(tl.float32)
    if FIRST:
        if CLAMP is not None:
            value = (
                (CLAMP * tldevice.tanh(value / CLAMP))
                .to(x.dtype.element_ty)
                .to(tl.float32)
            )
        value = tl.maximum(value, 0)
        value = (value * value).to(x.dtype.element_ty).to(tl.float32)
        value *= tl.load(routing + assignment)
    tl.store(
        output + assignment * OUT + cols[None, :] + tl.zeros((BM, 1), tl.int32),
        value,
        (rows[:, None] == 0) & (cols[None, :] < OUT),
    )


@triton.jit
def _expert_sum(
    parts,
    ids,
    output,
    TOP: tl.constexpr,
    WIDTH: tl.constexpr,
    BEGIN: tl.constexpr,
    END: tl.constexpr,
    B: tl.constexpr,
):
    # int64: with DP, (token * TOP + route) * WIDTH can pass 2**31.
    token, tile = tl.program_id(0).to(tl.int64), tl.program_id(1)
    cols = tile * B + tl.arange(0, B)
    value = tl.full((B,), 0, tl.float32)
    # The caller sorts routes by expert id to keep a fixed accumulation order.
    for route in tl.static_range(TOP):
        expert = tl.load(ids + token * TOP + route)
        part = tl.load(
            parts + (token * TOP + route) * WIDTH + cols,
            (cols < WIDTH) & (expert >= BEGIN) & (expert < END),
            other=0,
        ).to(tl.float32)
        value += part
    tl.store(output + token * WIDTH + cols, value, cols < WIDTH)


def routed_experts(
    x: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    begin: int,
    clamp: float | None,
    up_scale: torch.Tensor | None = None,
    down_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the local experts on every route and sum the top-k in FP32.

    `up` and `down` hold the experts `[begin, begin + len(up))`; routes to
    other experts contribute zero, so EP ranks return partial sums.
    """
    ids, order = ids.sort(dim=-1)
    weights = weights.gather(1, order)
    tokens, top = ids.shape
    intermediate, width = up.shape[1:]
    output = torch.empty_like(x, dtype=torch.float32)
    if tokens == 0:
        return output
    hidden = x.new_empty((tokens * top, intermediate))
    parts = x.new_empty((tokens * top, width))
    for first, inp, matrix, out, scale in (
        (True, x, up, hidden, up_scale),
        (False, hidden, down, parts, down_scale),
    ):
        inner, outer = (width, intermediate) if first else (intermediate, width)
        # Routed assignments can exceed CUDA's 65535-block grid Y limit, so
        # they go on the X axis.
        _expert_mm[(tokens * top, triton.cdiv(outer, 64))](
            inp,
            matrix,
            scale,
            ids,
            weights,
            out,
            top,
            inner,
            outer,
            first,
            begin,
            begin + up.shape[0],
            clamp,
            64,
            128 if scale is not None else 32,
            MXFP8_BLOCK_SIZE,
            scale is not None,
            enable_fp_fusion=False,
        )
    _expert_sum[(tokens, triton.cdiv(width, 256))](
        parts,
        ids,
        output,
        top,
        width,
        begin,
        begin + up.shape[0],
        256,
        enable_fp_fusion=False,
    )
    return output
