# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from torch.nn import Module

from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEConfig,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    int8_w8a16_moe_quant_config,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.online_moe import (
    OnlineMoEMethodBase,
)


class ExpertsInt8Config(QuantizationConfig):
    """Online int8 quantization for MoE expert weights.
    Linear layers are left unquantized."""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "experts_int8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ExpertsInt8Config":
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        elif isinstance(layer, FusedMoE):
            return ExpertsInt8MoEMethod(self, layer.moe_config)
        return None


class ExpertsInt8MoEMethod(OnlineMoEMethodBase):
    """Online int8 MoE quantization. Loads full-precision weights and
    quantizes to int8 with per-row scales during model loading."""

    def __init__(
        self,
        quant_config: QuantizationConfig,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.quant_config = quant_config

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

        # Replace full-precision weights with quantized versions
        layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
        layer.w13_scale = torch.nn.Parameter(w13_scale, requires_grad=False)
        layer.w2_scale = torch.nn.Parameter(w2_scale, requires_grad=False)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return int8_w8a16_moe_quant_config(
            w1_scale=layer.w13_scale,
            w2_scale=layer.w2_scale,
            w1_zp=None,
            w2_zp=None,
        )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe import fused_experts

        return fused_experts(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=not self.moe.disable_inplace,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            quant_config=self.moe_quant_config,
        )
