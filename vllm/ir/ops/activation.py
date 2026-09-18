# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F
from torch import Tensor

from ..op import register_op


@register_op
def gelu_and_mul_sparse(
    x: Tensor, std_multiplier: float, approximate: str = "none"
) -> Tensor:
    """Apply Gaussian sparsification, GELU, and gated multiplication."""
    d = x.shape[-1] // 2
    gate = x[..., :d]
    # Statistics intentionally remain local to each tensor-parallel shard.
    mean = torch.mean(gate, dim=-1, keepdim=True)
    std = torch.std(gate, dim=-1, keepdim=True, unbiased=False)
    sparse_gate = F.relu(gate - (mean + std * std_multiplier))
    return F.gelu(sparse_gate, approximate=approximate) * x[..., d:]


@gelu_and_mul_sparse.register_input_generator
def _gelu_and_mul_sparse_input_generator(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    std_multiplier: float = 1.6448536269514722,
    approximate: str = "tanh",
) -> tuple:
    x = torch.randn(num_tokens, hidden_size * 2, dtype=dtype)
    return x, std_multiplier, approximate
