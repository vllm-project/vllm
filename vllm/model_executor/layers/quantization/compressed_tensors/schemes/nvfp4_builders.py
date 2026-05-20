# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)

__all__ = [
    "NvFp4DynamicActivationBuilder",
    "NvFp4StaticWeightBuilder",
]


class NvFp4StaticWeightBuilder:
    """Register and normalize compressed-tensors NVFP4 static weight params."""

    group_size = 16

    def __init__(self, *, wrap_weight: bool = False):
        self.wrap_weight = wrap_weight

    def create(
        self,
        *,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        weight_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_global_scale", weight_global_scale)

        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def post_load(self, layer: torch.nn.Module) -> None:
        if self.wrap_weight:
            # Marlin W4A16 expects an ordinary Parameter after checkpoint loading.
            layer.weight = Parameter(layer.weight_packed.data, requires_grad=False)
        else:
            # Re-register the original ModelWeightParameter under the runtime name.
            layer.weight = layer.weight_packed
        del layer.weight_packed

        # CT stores global scales as divisors, i.e. 1 / runtime scale.
        weight_global_scale = layer.weight_global_scale.max().to(torch.float32)
        layer.weight_global_scale = Parameter(
            1.0 / weight_global_scale, requires_grad=False
        )


class NvFp4DynamicActivationBuilder:
    """Register and normalize compressed-tensors NVFP4 dynamic activation params."""

    def create(
        self,
        *,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ) -> None:
        input_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("input_global_scale", input_global_scale)

    def post_load(self, layer: torch.nn.Module) -> None:
        # CT stores global scales as divisors, i.e. 1 / runtime scale.
        input_global_scale_inv = layer.input_global_scale.max().to(torch.float32)
        layer.input_global_scale = Parameter(
            (1.0 / input_global_scale_inv).to(torch.float32), requires_grad=False
        )
        layer.input_global_scale_inv = Parameter(
            input_global_scale_inv, requires_grad=False
        )
