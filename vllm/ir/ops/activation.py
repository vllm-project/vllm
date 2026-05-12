# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F
from torch import Tensor

from ..op import register_op


@register_op
def silu_and_mul_with_clamp(x: Tensor, swiglu_limit: float) -> Tensor:
    """SwiGLU activation with input clamping (used by some MoE shared experts).

    Computes:
        gate = clamp(x[..., :d], max=swiglu_limit)
        up   = clamp(x[..., d:], min=-swiglu_limit, max=swiglu_limit)
        out  = silu(gate) * up
    where d = x.shape[-1] // 2.
    """
    d = x.shape[-1] // 2
    gate = torch.clamp(x[..., :d], max=swiglu_limit)
    up = torch.clamp(x[..., d:], min=-swiglu_limit, max=swiglu_limit)
    return F.silu(gate) * up
