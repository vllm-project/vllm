# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Normalize MTP embedding and hidden states directly into the projection input."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _norm_concat(
    E,
    H,
    EW,
    HW,
    Output,
    ES: tl.constexpr,
    HS: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    ZERO_CENTERED: tl.constexpr,
    ROUND_BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    e = tl.load(E + row * ES + col, col < D, other=0)
    h = tl.load(H + row * HS + col, col < D, other=0)
    if ROUND_BF16:
        e = e.to(tl.bfloat16)
        h = h.to(tl.bfloat16)
    e = e.to(tl.float32)
    h = h.to(tl.float32)
    ew = tl.load(EW + col, col < D, other=0).to(tl.float32)
    hw = tl.load(HW + col, col < D, other=0).to(tl.float32)
    if ZERO_CENTERED:
        ew = ew + 1.0
        hw = hw + 1.0
    e = (e * tl.rsqrt(tl.sum(e * e, 0) / D + EPS)) * ew
    h = (h * tl.rsqrt(tl.sum(h * h, 0) / D + EPS)) * hw
    tl.store(Output + row * (2 * D) + col, e, col < D)
    tl.store(Output + row * (2 * D) + D + col, h, col < D)


def norm_concat(
    embeds: torch.Tensor,
    hidden: torch.Tensor,
    embed_weight: torch.Tensor,
    hidden_weight: torch.Tensor,
    eps: float,
    zero_centered: bool,
    fp32_residual_connection: bool,
) -> torch.Tensor:
    """Preserve the BF16 input rounding used with FP32 residual connections."""
    dtype = (
        torch.bfloat16
        if fp32_residual_connection
        else torch.promote_types(embeds.dtype, hidden.dtype)
    )
    output = torch.empty(
        (embeds.shape[0], 2 * embeds.shape[1]), device=embeds.device, dtype=dtype
    )
    if embeds.shape[0]:
        _norm_concat[(embeds.shape[0],)](
            embeds,
            hidden,
            embed_weight,
            hidden_weight,
            output,
            embeds.stride(0),
            hidden.stride(0),
            embeds.shape[1],
            eps,
            zero_centered,
            fp32_residual_connection,
            triton.next_power_of_2(embeds.shape[1]),
            enable_fp_fusion=False,
        )
    return output
