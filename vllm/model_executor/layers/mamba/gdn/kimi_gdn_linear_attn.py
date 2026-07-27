# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config import VllmConfig
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.mamba.gdn.kda_linear_attn import KDAAttentionBase
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig


@PluggableLayer.register("kimi_gated_delta_net_attention")
class KimiGatedDeltaNetAttention(KDAAttentionBase):
    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        kda_config = config.linear_attn_config  # type: ignore[attr-defined]
        assert kda_config is not None, "linear_attn_config must be set"

        super().__init__(
            config,
            vllm_config,
            prefix,
            num_heads=kda_config["num_heads"],
            head_dim=kda_config["head_dim"],
            conv_size=kda_config["short_conv_kernel_size"],
            conv_params_dtype=torch.float32,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        output[:] = self._forward_kda(hidden_states)
