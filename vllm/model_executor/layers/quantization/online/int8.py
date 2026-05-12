# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch
from torch.nn import Module

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe import FusedMoE
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEQuantConfig,
    )

from vllm.model_executor.layers.fused_moe.oracle.int8 import (
    make_int8_moe_kernel,
    make_int8_moe_quant_config,
    select_int8_moe_backend,
)
from vllm.model_executor.layers.quantization.online.moe_base import (
    OnlineMoEMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
)
from vllm.model_executor.utils import replace_parameter


class Int8OnlineMoEMethod(OnlineMoEMethodBase):
    """Online per-channel INT8 MoE quantization.
    Loads fp16/bf16 weights and quantizes them per-row to int8 during loading.
    """

    def __init__(
        self,
        *,
        layer: torch.nn.Module,
    ):
        super().__init__(layer.moe_config)
        self.int8_backend, self.experts_cls = select_int8_moe_backend(
            config=self.moe,
            weight_key=kInt8StaticChannelSym,
            activation_key=kInt8DynamicTokenSym,
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        self._quantize_weights(layer)
        self._setup_kernel(layer)

        layer._already_called_process_weights_after_loading = True

    def _quantize_weights(self, layer: Module) -> None:
        vmax = torch.iinfo(torch.int8).max

        w13 = torch.empty_like(layer.w13_weight, dtype=torch.int8)
        w2 = torch.empty_like(layer.w2_weight, dtype=torch.int8)
        w13_scale = torch.zeros(
            layer.num_experts,
            layer.w13_weight.shape[1],
            device=w13.device,
            dtype=torch.float32,
        )
        w2_scale = torch.zeros(
            layer.num_experts,
            layer.w2_weight.shape[1],
            device=w2.device,
            dtype=torch.float32,
        )

        for expert in range(layer.local_num_experts):
            # w13: per-row quantization over hidden_size dim
            w = layer.w13_weight[expert, :, :]
            scales = w.abs().amax(dim=1) / vmax
            q = w.div(scales.unsqueeze(1)).round().clamp(-vmax, vmax)
            w13[expert, :, :] = q.to(torch.int8)
            w13_scale[expert, :] = scales

            # w2: per-row quantization over intermediate_size dim
            w = layer.w2_weight[expert, :, :]
            scales = w.abs().amax(dim=1) / vmax
            q = w.div(scales.unsqueeze(1)).round().clamp(-vmax, vmax)
            w2[expert, :, :] = q.to(torch.int8)
            w2_scale[expert, :] = scales

        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w13_scale", w13_scale)
        replace_parameter(layer, "w2_scale", w2_scale)

    def _setup_kernel(self, layer: "FusedMoE") -> None:
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.moe_quant_config is not None
        assert self.experts_cls is not None
        self.moe_kernel = make_int8_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            routing_tables=layer._expert_routing_tables(),
            shared_experts=layer.shared_experts,
        )

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> "FusedMoEQuantConfig | None":
        quant_config = make_int8_moe_quant_config(
            w1_scale=layer.w13_scale,
            w2_scale=layer.w2_scale,
        )
        self._maybe_inject_biases(quant_config, layer)
        return quant_config
