# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEQuantConfig,
    )

from vllm.model_executor.kernels.linear import init_int8_linear_kernel
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.oracle.int8 import (
    convert_to_int8_moe_kernel_format,
    make_int8_moe_kernel,
    make_int8_moe_quant_config,
    select_int8_moe_backend,
)
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.online.moe_base import (
    OnlineMoEMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs


class Int8OnlineLinearMethod(LinearMethodBase):
    """Online per-channel INT8 weights with dynamic per-token activations."""

    def create_weights(
        self,
        layer: Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        layer.logical_widths = output_partition_sizes
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)
        self.kernel = init_int8_linear_kernel(
            is_channelwise=True,
            is_static_input_scheme=False,
            input_symmetric=True,
            module_name=self.__class__.__name__,
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        weight = layer.weight.float()
        weight_scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127
        weight = (weight / weight_scale).round().clamp(-127, 127).to(torch.int8)

        replace_parameter(layer, "weight", weight)
        layer.register_parameter(
            "weight_scale", Parameter(weight_scale, requires_grad=False)
        )
        layer.register_parameter("input_scale", None)
        layer.register_parameter("input_zero_point", None)
        layer.register_parameter("azp_adj", None)
        self.kernel.process_weights_after_loading(layer)
        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)


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
            layer=layer,
        )

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
