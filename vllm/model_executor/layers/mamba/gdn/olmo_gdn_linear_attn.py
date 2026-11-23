# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
)
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils.torch_utils import direct_register_custom_op


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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(hidden_states)
        torch.ops.vllm.olmo_hybrid_gdn_full_forward(hidden_states, output, self.prefix)
        return output

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


@eager_break_during_capture
def olmo_hybrid_gdn_full_forward(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    # Preserve OLMo's projection/norm rounding under torch.compile.
    layer = get_forward_context().no_compile_layers[layer_name]
    output.copy_(QwenGatedDeltaNetAttention.forward(layer, hidden_states))


direct_register_custom_op(
    op_name="olmo_hybrid_gdn_full_forward",
    op_func=olmo_hybrid_gdn_full_forward,
    mutates_args=["output"],
)
