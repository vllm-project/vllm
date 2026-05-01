# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F
from torch import Tensor

from ..op import register_op


@register_op
def mul_and_silu(x: Tensor) -> Tensor:
    """Activation function for SwiGLU with multiplication before SiLU."""
    d = x.shape[-1] // 2
    return x[..., :d] * F.silu(x[..., d:])


@mul_and_silu.register_input_generator
def _mul_and_silu_input_generator(
    num_tokens: int, hidden_size: int, dtype: torch.dtype
) -> tuple:
    x = torch.randn(num_tokens, 2 * hidden_size, dtype=dtype)
    return (x,)
