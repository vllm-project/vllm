# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch
from torch.nn import Module

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEQuantConfig,
    )

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.oracle.int8 import (
    convert_to_int8_moe_kernel_format,
    make_int8_moe_kernel,
    make_int8_moe_quant_config,
    select_int8_moe_backend,
)
from vllm.model_executor.layers.quantization.online.moe_base import (
    OnlineMoEMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    amax_for_moe_weight_quant,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
    weight_amax,
)
from vllm.model_executor.utils import replace_parameter


class Int8OnlineMoEMethod(OnlineMoEMethodBase):
    """Online per-channel INT8 MoE quantization.
    Loads fp16/bf16 weights and quantizes them per-row to int8 during loading.
    """

    def __init__(
        self,
        *,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.int8_backend, self.experts_cls = select_int8_moe_backend(
            config=self.moe,
            weight_key=kInt8StaticChannelSym,
            activation_key=kInt8DynamicTokenSym,
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        w13_weight, w2_weight = self.get_weights_for_quantization(layer)
        self._quantize_weights(layer, w13_weight, w2_weight)
        self._setup_kernel(layer)
        self.release_requantization_source_weights(layer)

        layer._already_called_process_weights_after_loading = True

    def arrival_row_group(self, layer: Module, name: str) -> int | None:
        if name == "w2_weight" and self.moe.tp_size > 1:
            return None  # its scale is reduced over the ranks sharding it
        return self._row_local(layer, name, 1)

    def quantize_param(
        self, layer: Module, name: str, weight: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Per-row int8; the scale as ``(experts, rows, 1)``."""
        vmax = torch.iinfo(torch.int8).max
        qweight = torch.empty_like(weight, dtype=torch.int8)
        scale = torch.zeros(
            *weight.shape[:2], 1, device=weight.device, dtype=torch.float32
        )
        if name == "w2_weight":
            amax = weight_amax(weight, dim=-1)
            amax = amax_for_moe_weight_quant(amax, self.moe.tp_size)
        for expert in range(weight.shape[0]):
            w = weight[expert, :, :]
            if name == "w2_weight":
                scales = amax[expert] / vmax  # over the whole intermediate dim
            else:
                scales = w.abs().amax(dim=1) / vmax  # over the hidden dim
            q = w.div(scales.unsqueeze(1)).round().clamp(-vmax, vmax)
            qweight[expert, :, :] = q.to(torch.int8)
            scale[expert, :, 0] = scales
        return qweight, scale

    def _quantize_weights(
        self,
        layer: Module,
        w13_weight: torch.Tensor | None = None,
        w2_weight: torch.Tensor | None = None,
    ) -> None:
        w13_weight = layer.w13_weight if w13_weight is None else w13_weight
        w2_weight = layer.w2_weight if w2_weight is None else w2_weight
        w13, w13_scale = self.quantized_param(layer, "w13_weight", w13_weight)
        w2, w2_scale = self.quantized_param(layer, "w2_weight", w2_weight)
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w13_scale", w13_scale.squeeze(-1))
        replace_parameter(layer, "w2_scale", w2_scale.squeeze(-1))

    def _setup_kernel(self, layer: RoutedExperts) -> None:
        w13, w2 = convert_to_int8_moe_kernel_format(
            int8_backend=self.int8_backend,
            w13=layer.w13_weight,
            w2=layer.w2_weight,
            layer=layer,
            w13_scale=layer.w13_scale,
        )
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.moe_quant_config is not None
        assert self.experts_cls is not None
        self.moe_kernel = make_int8_moe_kernel(
            int8_backend=self.int8_backend,
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            routing_tables=layer._expert_routing_tables(),
        )
        self.moe_kernel.fused_experts.process_weights_after_loading(layer)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> "FusedMoEQuantConfig | None":
        return make_int8_moe_quant_config(
            int8_backend=self.int8_backend,
            w1_scale=getattr(layer, "w13_scale", None),
            w2_scale=getattr(layer, "w2_scale", None),
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
            layer=layer,
        )
