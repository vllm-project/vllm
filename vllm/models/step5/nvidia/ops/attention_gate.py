# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Headwise sigmoid output gating with the original low-precision rounding."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _headwise_sigmoid_gate(
    Attention,
    Gate,
    Output,
    Active,
    ACTIVE_STRIDE: tl.constexpr,
    MASK_INACTIVE: tl.constexpr,
    MASK_IS_SLOT: tl.constexpr,
    MASK_ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ATTENTION_ROW: tl.constexpr,
    ATTENTION_COL: tl.constexpr,
    GATE_ROW: tl.constexpr,
    GATE_COL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(
        Gate + row * GATE_ROW + (col // HEAD_DIM) * GATE_COL,
        col < WIDTH,
        other=0,
    ).to(tl.float32)
    sigmoid = tl.div_rn(1.0, 1.0 + tl.extra.cuda.libdevice.exp(-gate))
    # PyTorch materializes sigmoid in gate dtype before the multiply.
    sigmoid = sigmoid.to(Gate.dtype.element_ty).to(tl.float32)
    value = tl.load(
        Attention + row.to(tl.int64) * ATTENTION_ROW + col * ATTENTION_COL,
        col < WIDTH,
        other=0,
    ).to(tl.float32)
    if MASK_INACTIVE:
        active = tl.load(
            Active + row * ACTIVE_STRIDE,
            row < MASK_ROWS,
            other=-1 if MASK_IS_SLOT else 0,
        )
        if MASK_IS_SLOT:
            active = active >= 0
        # Clear attention before multiplication: 0 * sigmoid(NaN) remains NaN.
        value = tl.where(active, value, 0.0)
    tl.store(Output + row.to(tl.int64) * WIDTH + col, value * sigmoid, col < WIDTH)


def headwise_sigmoid_gate(
    attention: torch.Tensor,
    gate: torch.Tensor,
    active: torch.Tensor | None = None,
    *,
    slot_mapping: torch.Tensor | None = None,
) -> torch.Tensor:
    if active is not None and slot_mapping is not None:
        raise ValueError("Pass either active flags or decode slots")
    mask = active if active is not None else slot_mapping
    output = attention.new_empty(attention.shape)
    head_dim = attention.shape[1] // gate.shape[1]
    _headwise_sigmoid_gate[(attention.shape[0], triton.cdiv(attention.shape[1], 1024))](
        attention,
        gate,
        output,
        mask,
        mask.stride(0) if mask is not None else 0,
        mask is not None,
        slot_mapping is not None,
        mask.numel() if mask is not None else 0,
        attention.shape[1],
        head_dim,
        *attention.stride(),
        *gate.stride(),
        1024,
        enable_fp_fusion=False,
    )
    return output
