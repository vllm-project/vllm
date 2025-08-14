# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Callable, Optional

import torch
from compressed_tensors.quantization import (QuantizationArgs,
                                             QuantizationStrategy,
                                             QuantizationType)
from torch.nn import Parameter

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp, maybe_create_device_identity, normalize_e4m3fn_to_e4m3fnuz,
    requantize_with_max_scale)
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)
from vllm.platforms import current_platform

__all__ = [
    "is_fp8_w8a8", "is_fp8_w8a8_sm90", "is_fp8_w8a8_sm100",
    "CompressedTensorsW8A8Fp8"
]


def is_fp8_w8a8(self, weight_quant: QuantizationArgs,
                input_quant: Optional[QuantizationArgs]) -> bool:
    # Confirm weights and activations quantized.
    if weight_quant is None or input_quant is None:
        return False
    # Confirm weight scheme is supported.
    is_floating_point = (weight_quant.type == QuantizationType.FLOAT
                         and input_quant.type == QuantizationType.FLOAT)
    is_symmetric_weight = weight_quant.symmetric
    is_static_weight = not weight_quant.dynamic
    is_per_tensor_or_channel_weight = (weight_quant.strategy in [
        QuantizationStrategy.TENSOR, QuantizationStrategy.CHANNEL
    ])
    if not (is_floating_point and is_symmetric_weight and is_static_weight
            and is_per_tensor_or_channel_weight):
        return False
    # Dynamic quantization is always supported if weights supported.
    if input_quant.dynamic:
        return True
    # Confirm activation scheme is supported.
    is_symmetric_activation = input_quant.symmetric
    is_per_tensor_activation = (
        input_quant.strategy == QuantizationStrategy.TENSOR)
    return is_symmetric_activation and is_per_tensor_activation


# TODO: resolve _check_scheme_supported
def is_fp8_w8a8_sm90(self, weight_quant: QuantizationArgs,
                     input_quant: Optional[QuantizationArgs]) -> bool:
    return (CompressedTensorsScheme.check_scheme_supported(
        90, error=False, match_exact=True)
            and is_fp8_w8a8(weight_quant, input_quant))


def is_fp8_w8a8_sm100(self, weight_quant: QuantizationArgs,
                      input_quant: Optional[QuantizationArgs]) -> bool:
    return (CompressedTensorsScheme.check_scheme_supported(
        100, error=False, match_exact=True)
            and is_fp8_w8a8(weight_quant, input_quant))


class CompressedTensorsW8A8Fp8(CompressedTensorsScheme):

    def __init__(self, strategy: str, is_static_input_scheme: bool):
        self.strategy = strategy
        self.out_dtype = torch.get_default_dtype()
        self.is_static_input_scheme = is_static_input_scheme
        self.act_q_group_shape = GroupShape.PER_TENSOR \
            if is_static_input_scheme else GroupShape.PER_TOKEN
        self.fp8_linear = Fp8LinearOp(
            act_quant_static=self.is_static_input_scheme,
            act_quant_group_shape=self.act_q_group_shape)

    @classmethod
    def get_min_capability(cls) -> int:
        # lovelace and up
        return 89

    def process_weights_after_loading(self, layer) -> None:
        # If per tensor, when we have a fused module (e.g. QKV) with per
        # tensor scales (thus N scales being passed to the kernel),
        # requantize so we can always run per tensor
        if self.strategy == QuantizationStrategy.TENSOR:
            max_w_scale, weight = requantize_with_max_scale(
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                logical_widths=layer.logical_widths,
            )

            if current_platform.is_fp8_fnuz():
                input_scale = getattr(layer, 'input_scale', None)

                weight, max_w_scale, input_scale = normalize_e4m3fn_to_e4m3fnuz(
                    weight=weight,
                    weight_scale=max_w_scale,
                    input_scale=input_scale)
                if input_scale is not None:
                    layer.input_scale = Parameter(input_scale,
                                                  requires_grad=False)

            layer.weight = Parameter(weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(max_w_scale, requires_grad=False)

        # If channelwise, scales are already lined up, so just transpose.
        elif self.strategy == QuantizationStrategy.CHANNEL:
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

            layer.weight = Parameter(weight.t(), requires_grad=False)
            # required by torch.compile to be torch.nn.Parameter
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)

        else:
            raise ValueError(f"Unknown quantization strategy {self.strategy}")

        # INPUT SCALE
        if self.is_static_input_scheme and hasattr(layer, 'input_scale'):
            layer.input_scale = Parameter(layer.input_scale.max(),
                                          requires_grad=False)
        else:
            layer.input_scale = None

    def create_weights(self, layer: torch.nn.Module,
                       output_partition_sizes: list[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):
        maybe_create_device_identity()

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
        if self.strategy == QuantizationStrategy.CHANNEL:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1),
                                 dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader)
        else:
            assert self.strategy == QuantizationStrategy.TENSOR
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
