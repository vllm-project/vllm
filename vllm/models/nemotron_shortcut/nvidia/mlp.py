# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nemotron Shortcut dense MLP with the soft-clamped squared-ReLU activation.

Shared by the dense MLP blocks and by the MoE's shared experts, so it lives
apart from both the model and the MoE package.
"""

import torch
from torch import nn
from transformers import PreTrainedConfig

from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig


class NemotronShortcutClampedActivation(nn.Module):
    """Soft-clamps the pre-activation to `activation_clamp_value` before relu2."""

    def __init__(self, config: PreTrainedConfig) -> None:
        super().__init__()
        self.clamp_value = config.activation_clamp_value
        self.act_fn = get_act_fn(config.mlp_hidden_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.clamp_value is not None:
            x = (self.clamp_value * torch.tanh(x.float() / self.clamp_value)).to(
                x.dtype
            )
        return self.act_fn(x)


class NemotronShortcutMLP(nn.Module):
    def __init__(
        self,
        config: PreTrainedConfig,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.up_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=intermediate_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = NemotronShortcutClampedActivation(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x
