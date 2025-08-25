# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Callable, Optional, cast

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.quantization.quark.schemes import QuarkScheme
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp, normalize_e4m3fn_to_e4m3fnuz, requantize_with_max_scale)
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)
from vllm.platforms import current_platform

__all__ = ["QuarkW8A8Fp8"]


class QuarkW8A8Fp8(QuarkScheme):

    def __init__(self, weight_config: dict[str, Any],
                 input_config: Optional[dict[str, Any]]):
        self.weight_qscheme = cast(str, weight_config.get("qscheme"))
        self.is_static_input_scheme: bool = False
        self.input_qscheme: Optional[str] = None
        if input_config is not None:
            self.is_static_input_scheme = not cast(
                bool, input_config.get("is_dynamic"))
            self.input_qscheme = cast(str, input_config.get("qscheme"))

        per_token = (not self.is_static_input_scheme
                     and self.input_qscheme == "per_channel")
        self.act_quant_group_shape = GroupShape.PER_TOKEN \
            if per_token else GroupShape.PER_TENSOR
        self.fp8_linear = Fp8LinearOp(
            act_quant_static=self.is_static_input_scheme,
            act_quant_group_shape=self.act_quant_group_shape)
        self.out_dtype = torch.get_default_dtype()

    @classmethod
    def get_min_capability(cls) -> int:
        # lovelace and up
        return 89

    def process_weights_after_loading(self, layer) -> None:
        # If per tensor, when we have a fused module (e.g. QKV) with per
        # tensor scales (thus N scales being passed to the kernel),
        # requantize so we can always run per tensor
        if self.weight_qscheme == "per_tensor":
            if current_platform.is_fp8_fnuz():
                input_scale = getattr(layer, 'input_scale', None)
                weight, max_w_scale, input_scale = normalize_e4m3fn_to_e4m3fnuz(
                    weight=layer.weight,
                    weight_scale=layer.weight_scale,
                    input_scale=input_scale)
                if input_scale is not None:
                    layer.input_scale = Parameter(input_scale,
                                                  requires_grad=False)
            else:
                max_w_scale = layer.weight_scale
                weight = layer.weight

            max_w_scale, weight = requantize_with_max_scale(
                weight=weight,
                weight_scale=max_w_scale,
                logical_widths=layer.logical_widths,
            )

            layer.weight = Parameter(weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(max_w_scale, requires_grad=False)

        # If channelwise, scales are already lined up, so just transpose.
        elif self.weight_qscheme == "per_channel":
            weight = layer.weight

            if current_platform.is_fp8_fnuz():
                input_scale = getattr(layer, 'input_scale', None)
                weight, weight_scale, input_scale = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        weight=weight,
                        weight_scale=layer.weight_scale,
                        input_scale=input_scale)
                if input_scale is not None:
                    layer.input_scale = Parameter(input_scale,
                                                  requires_grad=False)
            else:
                weight_scale = layer.weight_scale.data
            if self.act_quant_group_shape == GroupShape.PER_TOKEN:
                weight_scale = weight_scale.view(-1, 1)
            layer.weight = Parameter(weight.t(), requires_grad=False)
            # required by torch.compile to be torch.nn.Parameter
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)

        else:
            raise ValueError(
                f"Unknown quantization scheme {self.weight_qscheme}")

        # INPUT SCALE
        if self.is_static_input_scheme:
            layer.input_scale = Parameter(layer.input_scale.max(),
                                          requires_grad=False)
        else:
            layer.input_scale = None

    def create_weights(self, layer: torch.nn.Module,
                       output_partition_sizes: list[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes

        # WEIGHT
        weight = ModelWeightParameter(data=torch.empty(
            output_size_per_partition,
            input_size_per_partition,
            dtype=torch.float8_e4m3fn),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        # TODO: update create_xxx_parameter functions to return
        # the newly added parameters
        if self.weight_qscheme == "per_channel":
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes)),
                                 dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader)
        else:
            assert self.weight_qscheme == "per_tensor"
            weight_scale = PerTensorScaleParameter(data=torch.empty(
                len(output_partition_sizes), dtype=torch.float32),
                                                   weight_loader=weight_loader)

        # min requirement for fp8 kernels
        weight_scale[:] = torch.finfo(torch.float32).min
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE
        if self.is_static_input_scheme:
            input_scale = PerTensorScaleParameter(data=torch.empty(
                len(output_partition_sizes), dtype=torch.float32),
                                                  weight_loader=weight_loader)
            input_scale[:] = torch.finfo(torch.float32).min
            layer.register_parameter("input_scale", input_scale)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        return self.fp8_linear.apply(input=x,
                                     weight=layer.weight,
                                     weight_scale=layer.weight_scale,
                                     out_dtype=self.out_dtype,
                                     input_scale=layer.input_scale,
                                     bias=bias)
