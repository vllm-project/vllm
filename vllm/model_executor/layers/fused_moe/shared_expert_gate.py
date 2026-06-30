# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernel for the Qwen2/3-MoE shared-expert sigmoid gate.

Replaces the three-kernel `F.sigmoid(linear(x)) * out` tail of
`Qwen2MoeMLP.forward` / `Qwen3MoeMLP.forward` with a single row-fused
pass that removes the two HBM-resident intermediates.

The wrapper is shape-guarded and silently falls back to the PyTorch
reference (`F.sigmoid(F.linear(x, weight)) * out`) for any input shape
this kernel does not handle, so it is safe to use behind the existing
`expert_gate` call sites without further checks.
"""

import torch
import torch.nn.functional as F

from vllm.triton_utils import tl, triton


@triton.jit
def _fused_shared_expert_gate_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    y_ptr,
    stride_x_n,
    stride_out_n,
    stride_y_n,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K

    x = tl.load(x_ptr + row * stride_x_n + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.sigmoid(tl.sum(x * weight, axis=0))

    out = tl.load(out_ptr + row * stride_out_n + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    tl.store(y_ptr + row * stride_y_n + offsets, out * gate, mask=mask)


def fused_shared_expert_gate(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Compute ``F.sigmoid(F.linear(x, weight)) * out`` in a single pass.

    Specialised for a one-row gate weight (``weight.shape == [1, K]``), as
    produced by ``ReplicatedLinear(hidden_size, 1)`` in the Qwen2/3-MoE
    shared-expert blocks. The kernel handles arbitrary row strides on ``x``,
    ``out``, and the output ``y`` (so views with non-K row stride are fine),
    but assumes unit stride along the inner ``K`` dimension. For any input
    that violates these requirements -- including the unit-inner-stride
    requirement on ``weight`` -- the function falls back to the PyTorch
    reference so callers can use it unconditionally.

    Args:
        x: Shared-expert input, shape ``[N, K]``.
        weight: Gate weight, shape ``[1, K]``.
        out: Shared-expert MLP output, shape ``[N, K]``.

    Returns:
        Contiguous ``[N, K]`` tensor equal to ``sigmoid(x @ weight.T) * out``
        within bf16/fp16 tolerance.
    """
    if (
        x.ndim != 2
        or out.ndim != 2
        or weight.ndim != 2
        or weight.shape[0] != 1
        or x.shape != out.shape
        or weight.shape[1] != x.shape[1]
        or x.stride(1) != 1
        or out.stride(1) != 1
        or weight.stride(1) != 1
    ):
        return F.sigmoid(F.linear(x, weight)) * out

    y = torch.empty_like(out, memory_format=torch.contiguous_format)
    _fused_shared_expert_gate_kernel[(x.shape[0],)](
        x,
        weight,
        out,
        y,
        x.stride(0),
        out.stride(0),
        y.stride(0),
        K=x.shape[1],
        BLOCK_K=triton.next_power_of_2(x.shape[1]),
        num_warps=8,
    )
    return y
