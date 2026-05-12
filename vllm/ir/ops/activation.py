# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch.nn.functional as F
from torch import Tensor

from ..op import register_op


@register_op
def mul_and_silu(x: Tensor) -> Tensor:
    """Activation function for SwiGLU (mul-then-silu variant)."""
    d = x.shape[-1] // 2
    return x[..., :d] * F.silu(x[..., d:])
