# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F
from torch import Tensor

from ..op import register_op


@register_op
def silu_and_mul_fp8(input: Tensor, scale: Tensor) -> Tensor:
    """SiLU(x[.., :d]) * x[.., d:], quantized to float8_e4m3fn with scalar scale."""
    d = input.shape[-1] // 2
    result = F.silu(input[..., :d]) * input[..., d:]
    return (result.to(torch.float32) / scale).to(torch.float8_e4m3fn)
