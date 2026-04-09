# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch.nn.functional as F
from torch import Tensor

from ..op import register_op


@register_op
def silu_and_mul(
    x: Tensor
) -> Tensor:
    """Activation function for SwiGLU"""
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]
