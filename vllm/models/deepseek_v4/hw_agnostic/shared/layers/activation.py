# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """SwiGLU: ``x -> silu(x[:d]) * x[d:]`` where ``d = x.shape[-1] // 2``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]


class SiluAndMulWithClamp(nn.Module):
    """SwiGLU with input clamping. ``d = x.shape[-1] // 2``;
    ``out = silu(clamp(x[:d], max=L)) * clamp(x[d:], min=-L, max=L)``."""

    def __init__(self, swiglu_limit: float):
        super().__init__()
        self.swiglu_limit = float(swiglu_limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = torch.clamp(x[..., :d], max=self.swiglu_limit)
        up = torch.clamp(x[..., d:], min=-self.swiglu_limit, max=self.swiglu_limit)
        return F.silu(gate) * up
