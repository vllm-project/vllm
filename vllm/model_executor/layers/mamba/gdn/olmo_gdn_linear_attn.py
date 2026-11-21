# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
)
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.quantization import QuantizationConfig


class _SeparateBAProjection(nn.ModuleDict):
    """Preserve OLMo's gate rounding with separate B/A GEMMs."""

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        b, _ = self["b_proj"](hidden_states)
        a, _ = self["a_proj"](hidden_states)
        return torch.cat((b, a), dim=-1), None


@PluggableLayer.register("olmo_hybrid_gated_delta_net_attention")
class OlmoHybridGatedDeltaNetAttention(QwenGatedDeltaNetAttention):
    def __init__(self, config, vllm_config: VllmConfig, prefix: str = "") -> None:
        assert getattr(config, "linear_use_gate", True), (
            "OlmoHybridGatedDeltaNet requires linear_use_gate=True"
        )
        super().__init__(
            config,
            vllm_config,
            prefix=prefix,
            norm_eps=1e-5,
            a_log_dtype=vllm_config.model_config.dtype,
        )

    def create_ba_proj(
        self,
        hidden_size: int,
        num_v_heads: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> nn.Module:
        return _SeparateBAProjection(
            {
                name: ColumnParallelLinear(
                    input_size=hidden_size,
                    output_size=num_v_heads,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.{name}",
                    disable_tp=self.maybe_disable_tp(quant_config),
                )
                for name in ("b_proj", "a_proj")
            }
        )

    def create_conv_proj(self, prefix: str) -> MergedColumnParallelLinear:
        conv = MergedColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_sizes=[self.key_dim, self.key_dim, self.value_dim],
            bias=False,
            prefix=f"{prefix}.conv1d",
        )
        conv.weight.data = conv.weight.data.unsqueeze(1)
        return conv
