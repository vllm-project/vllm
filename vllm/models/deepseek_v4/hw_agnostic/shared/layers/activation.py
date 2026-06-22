# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F

from vllm.models.deepseek_v4.hw_agnostic.shared.custom_op import CustomOp


@CustomOp.register("silu_and_mul")
class SiluAndMul(CustomOp):
    """SwiGLU: ``x -> silu(x[:d]) * x[d:]`` where ``d = x.shape[-1] // 2``."""

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]


@CustomOp.register("silu_and_mul_with_clamp")
class SiluAndMulWithClamp(CustomOp):
    """SwiGLU with input clamping. ``d = x.shape[-1] // 2``;
    ``out = silu(clamp(x[:d], max=L)) * clamp(x[d:], min=-L, max=L)``."""

    def __init__(self, swiglu_limit: float):
        super().__init__()
        self.swiglu_limit = float(swiglu_limit)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = torch.clamp(x[..., :d], max=self.swiglu_limit)
        up = torch.clamp(x[..., d:], min=-self.swiglu_limit, max=self.swiglu_limit)
        return F.silu(gate) * up
