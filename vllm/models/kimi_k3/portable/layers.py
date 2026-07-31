# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small, unfused PyTorch layers used by the portable Kimi models."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        dtype = inputs.dtype
        inputs = inputs.float()
        inputs = inputs * torch.rsqrt(
            inputs.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (inputs * self.weight.float()).to(dtype)


def situ(
    gate: torch.Tensor,
    up: torch.Tensor,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    dtype = gate.dtype
    gate = gate.float()
    up = up.float()
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (gate * up).to(dtype)


class KimiMLP(nn.Module):
    """A readable tensor-parallel gated MLP."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        reduce_results: bool = True,
        situ_beta: float | None = None,
        situ_linear_beta: float | None = None,
    ) -> None:
        super().__init__()
        self.hidden_act = hidden_act
        self.situ_beta = situ_beta or 1.0
        self.situ_linear_beta = situ_linear_beta
        self.gate_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_proj",
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )

    def activate(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if self.hidden_act == "silu":
            return F.silu(gate) * up
        if self.hidden_act == "situ":
            return situ(gate, up, self.situ_beta, self.situ_linear_beta)
        raise ValueError(f"Unsupported Kimi activation {self.hidden_act!r}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, _ = self.gate_proj(hidden_states)
        up, _ = self.up_proj(hidden_states)
        hidden_states = self.activate(gate, up)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class AttentionResidual(nn.Module):
    """K3's attention-residual weighted sum in ordinary PyTorch."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        prefix: str,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps)
        self.proj = ReplicatedLinear(
            hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=prefix,
        )

    def forward(
        self,
        prefix_sum: torch.Tensor,
        block_residuals: torch.Tensor,
    ) -> torch.Tensor:
        values = torch.cat((block_residuals, prefix_sum.unsqueeze(-2)), dim=-2)
        scores, _ = self.proj(self.norm(values))
        probabilities = scores.float().softmax(dim=-2)
        return (probabilities * values.float()).sum(dim=-2).to(values.dtype)
