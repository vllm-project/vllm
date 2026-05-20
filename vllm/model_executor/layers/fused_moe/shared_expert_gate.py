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
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K

    x = tl.load(x_ptr + row * K + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.sigmoid(tl.sum(x * weight, axis=0))

    out = tl.load(out_ptr + row * K + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * K + offsets, out * gate, mask=mask)


def fused_shared_expert_gate(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Compute ``F.sigmoid(F.linear(x, weight)) * out`` in a single pass.

    Specialised for a one-row gate weight (``weight.shape == [1, K]``), as
    produced by ``ReplicatedLinear(hidden_size, 1)`` in the Qwen2/3-MoE
    shared-expert blocks. For any other shape, the function falls back to
    the PyTorch reference so callers can use it unconditionally.

    Args:
        x: Shared-expert input, shape ``[N, K]``.
        weight: Gate weight, shape ``[1, K]``.
        out: Shared-expert MLP output, shape ``[N, K]``.

    Returns:
        ``[N, K]`` tensor equal to ``sigmoid(x @ weight.T) * out`` within
        bf16/fp16 tolerance.
    """
    if (
        x.ndim != 2
        or out.ndim != 2
        or weight.ndim != 2
        or weight.shape[0] != 1
        or x.shape != out.shape
        or weight.shape[1] != x.shape[1]
    ):
        return F.sigmoid(F.linear(x, weight)) * out

    y = torch.empty_like(out)
    _fused_shared_expert_gate_kernel[(x.shape[0],)](
        x,
        weight,
        out,
        y,
        K=x.shape[1],
        BLOCK_K=triton.next_power_of_2(x.shape[1]),
        num_warps=8,
    )
    return y
