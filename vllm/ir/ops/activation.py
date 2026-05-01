# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F
from torch import Tensor

from ..op import register_op


@register_op
def relu2(x: Tensor) -> Tensor:
    """Squared ReLU activation."""
    return torch.square(F.relu(x))


@relu2.register_input_generator
def _relu2_input_generator(
    num_tokens: int, hidden_size: int, dtype: torch.dtype
) -> tuple:
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    return (x,)
