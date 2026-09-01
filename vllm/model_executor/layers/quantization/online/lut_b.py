# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Online LUT-B quantization for routed MoE expert weights."""

import torch

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
)
from vllm.model_executor.layers.fused_moe.experts.lut_b_moe import (
    make_lut_b_moe_kernel,
)
from vllm.model_executor.layers.quantization.online.moe_base import (
    OnlineMoEMethodBase,
)
from vllm.model_executor.layers.quantization.utils.lut_b_utils import (
    LUT_B_BLOCK_K,
    LUT_B_BLOCK_N,
    quantize_lut_b,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform


class LutBOnlineMoEMethod(OnlineMoEMethodBase):
    """Quantize BF16/FP16 routed experts to LUT-B during model loading."""

    def __init__(self, *, layer: torch.nn.Module):
        if not current_platform.is_cuda_alike():
            raise ValueError("LUT-B online MoE quantization requires a GPU.")
        if layer.moe_config.has_bias:
            raise ValueError("LUT-B online MoE does not currently support bias.")
        super().__init__(layer.moe_config)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if (
            hidden_size % LUT_B_BLOCK_K != 0
            or intermediate_size_per_partition % LUT_B_BLOCK_K != 0
        ):
            raise ValueError(
                "LUT-B online MoE requires hidden and sharded intermediate "
                f"sizes divisible by {LUT_B_BLOCK_K}; got hidden={hidden_size}, "
                f"intermediate={intermediate_size_per_partition}."
            )
        if hidden_size % LUT_B_BLOCK_N != 0:
            raise ValueError(
                f"LUT-B online MoE requires hidden size divisible by "
                f"{LUT_B_BLOCK_N}; got {hidden_size}."
            )
        super().create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

    def get_fused_moe_quant_config(
        self,
        layer: torch.nn.Module,
    ) -> FusedMoEQuantConfig:
        return FusedMoEQuantConfig(
            _a1=FusedMoEQuantDesc(),
            _a2=FusedMoEQuantDesc(),
            _w1=FusedMoEQuantDesc(
                dtype="lut_b",
                scale=layer.w13_weight_codebook,
            ),
            _w2=FusedMoEQuantDesc(
                dtype="lut_b",
                scale=layer.w2_weight_codebook,
            ),
        )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        self._zero_padding(layer)
        w13, w13_codebook = quantize_lut_b(layer.w13_weight)
        w2, w2_codebook = quantize_lut_b(layer.w2_weight)
        replace_parameter(layer, "w13_weight", w13.contiguous())
        replace_parameter(layer, "w13_weight_codebook", w13_codebook.contiguous())
        replace_parameter(layer, "w2_weight", w2.contiguous())
        replace_parameter(layer, "w2_weight_codebook", w2_codebook.contiguous())

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        self.moe_kernel = make_lut_b_moe_kernel(
            moe_config=self.moe,
            quant_config=self.moe_quant_config,
            routing_tables=layer._expert_routing_tables(),
        )
        layer._already_called_process_weights_after_loading = True
