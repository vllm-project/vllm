# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm._custom_ops as ops

__all__ = ["HadamardTransform"]


class HadamardTransform(torch.nn.Module):
    """Apply a block-wise Hadamard transform to the last dimension."""

    def __init__(self, block_size: int) -> None:
        super().__init__()
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.unflatten(-1, (-1, self.block_size)).contiguous().clone()
        x = ops.hadacore_transform(x)
        return x.flatten(-2, -1).reshape(original_shape)
