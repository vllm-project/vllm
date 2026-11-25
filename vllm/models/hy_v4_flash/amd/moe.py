# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash latent experts with HYV4's routing, shared experts and weight format."""

import torch
from torch import nn

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.models.hy_v4.nvidia.moe import HYV4FeedForward, HYV4MoEFused
from vllm.models.hy_v4_flash.configs import HYV4FlashConfig


class HYV4FlashFeedForward(HYV4FeedForward):
    def __init__(
        self,
        config: HYV4FlashConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.post_fc2_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.sandwich_norm
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = super().forward(x)
        return self.post_fc2_norm(result) if self.post_fc2_norm is not None else result


class HYV4FlashMoE(HYV4MoEFused):
    def __init__(
        self,
        config: HYV4FlashConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        latent = config.routed_expert_hidden_size
        down: nn.Module | None = None
        up: nn.Module | None = None
        if latent is not None:
            down = ReplicatedLinear(
                config.hidden_size,
                latent,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.routed_expert_down_proj",
            )
            up = ReplicatedLinear(
                latent,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.routed_expert_up_proj",
            )
        super().__init__(
            config,
            quant_config=quant_config,
            prefix=prefix,
            routed_hidden_size=latent,
            routed_input_transform=down,
            routed_output_transform=up,
        )
        self.routed_expert_down_proj = down
        self.routed_expert_up_proj = up
        self.combine_output_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.sandwich_norm
            else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        result = super().forward(hidden_states)
        return (
            self.combine_output_norm(result)
            if self.combine_output_norm is not None
            else result
        )
