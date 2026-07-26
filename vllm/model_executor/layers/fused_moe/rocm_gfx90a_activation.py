# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MoE activations tuned for AMD gfx90a."""

import torch

from vllm.triton_utils import tl, triton

# The fused kernel becomes faster than separate ReLU and square kernels once
# the activation is large enough to amortize its additional store. Triton uses
# 32-bit offsets here, so retain the eager implementation for larger tensors.
_MIN_RELU2_ELEMENTS = 30_000_000
_MAX_RELU2_ELEMENTS = torch.iinfo(torch.int32).max


@triton.jit
def _relu2_kernel(
    input_ptr,
    output_ptr,
    elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elements
    values = tl.load(input_ptr + offsets, mask=mask)
    activated = tl.maximum(values, 0.0)
    tl.store(input_ptr + offsets, activated, mask=mask)
    tl.store(output_ptr + offsets, activated * activated, mask=mask)


def can_use_fused_relu2(output: torch.Tensor, input: torch.Tensor) -> bool:
    """Return whether the gfx90a fused ReLU-squared kernel supports the tensors."""
    elements = input.numel()
    return (
        input.dtype == torch.bfloat16
        and output.dtype == input.dtype
        and output.shape == input.shape
        and input.is_cuda
        and input.is_contiguous()
        and output.is_contiguous()
        and input.device == output.device
        and input.data_ptr() != output.data_ptr()
        and _MIN_RELU2_ELEMENTS <= elements <= _MAX_RELU2_ELEMENTS
    )


def fused_relu2(output: torch.Tensor, input: torch.Tensor) -> None:
    """Compute ReLU-squared while preserving the eager path in-place ReLU."""
    elements = input.numel()
    _relu2_kernel[(triton.cdiv(elements, 1024),)](
        input,
        output,
        elements,
        BLOCK_SIZE=1024,
        num_warps=4,
    )
