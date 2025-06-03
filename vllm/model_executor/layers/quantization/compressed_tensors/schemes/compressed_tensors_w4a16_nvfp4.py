# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Callable, Optional

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear, prepare_fp4_layer_for_marlin)
from vllm.model_executor.parameter import (GroupQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)

__all__ = ["CompressedTensorsW4A16Fp4"]


class CompressedTensorsW4A16Fp4(CompressedTensorsScheme):

    def __init__(self):
        self.group_size = 16

    @classmethod
    def get_min_capability(cls) -> int:
        # dont restrict as emulations
        return 80

    def create_weights(self, layer: torch.nn.Module,
                       output_partition_sizes: list[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # Weight
        weight = ModelWeightParameter(data=torch.empty(
            sum(output_partition_sizes),
            input_size_per_partition // 2,
            dtype=torch.uint8),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)
        layer.register_parameter("weight_packed", weight)

        # Global Weight Scale
        weight_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader)
        layer.register_parameter("weight_global_scale", weight_global_scale)

        # Per Group Weight Scale
        weight_scale = GroupQuantScaleParameter(data=torch.empty(
            sum(output_partition_sizes),
            input_size_per_partition // self.group_size,
            dtype=torch.float8_e4m3fn,
        ),
                                                input_dim=1,
                                                output_dim=0,
                                                weight_loader=weight_loader)

        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer) -> None:
        # Process parameters for marlin repacking

        # Rename weight_packed to weight that marlin expects
        layer.weight = Parameter(layer.weight_packed.data, requires_grad=False)
        del layer.weight_packed
        # Rename weight_global_scale to weight_scale_2 that marlin expects
        # Note: ct stores the inverse of what is expected by the marlin kernel
        layer.weight_scale_2 = Parameter(
            1 / layer.weight_global_scale.max().to(torch.float32),
            requires_grad=False)
        del layer.weight_global_scale

        prepare_fp4_layer_for_marlin(layer)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        return apply_fp4_marlin_linear(input=x,
                                       weight=layer.weight,
                                       weight_scale=layer.weight_scale,
                                       weight_scale_2=layer.weight_scale_2,
                                       workspace=layer.workspace,
                                       size_n=layer.output_size_per_partition,
                                       size_k=layer.input_size_per_partition,
                                       bias=bias)
