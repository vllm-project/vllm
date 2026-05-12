# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_mixfp4 import (
    apply_mixfp4_marlin_linear,
    prepare_mixfp4_layer_for_marlin,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)

__all__ = ["CompressedTensorsW4A16MixFP4"]


class CompressedTensorsW4A16MixFP4(CompressedTensorsScheme):
    def __init__(self):
        self.group_size = 16

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        if input_size_per_partition % (2 * self.group_size) != 0:
            raise ValueError(
                "MixFP4 Marlin requires input_size_per_partition to be "
                f"divisible by {2 * self.group_size}, got "
                f"{input_size_per_partition}."
            )
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

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

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = Parameter(layer.weight_packed.data, requires_grad=False)
        del layer.weight_packed

        weight_global_scale = layer.weight_global_scale.to(torch.float32).flatten()
        if weight_global_scale.numel() > 1:
            common_global_scale = weight_global_scale.min()
            layer.weight_scale = Parameter(
                _fold_global_scales_into_weight_scale(
                    weight_scale=layer.weight_scale,
                    global_scales=weight_global_scale,
                    common_global_scale=common_global_scale,
                    logical_widths=layer.logical_widths,
                ),
                requires_grad=False,
            )
        else:
            common_global_scale = weight_global_scale[0]

        layer.weight_global_scale = Parameter(
            1.0 / common_global_scale,
            requires_grad=False,
        )
        prepare_mixfp4_layer_for_marlin(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return apply_mixfp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )


def _fold_global_scales_into_weight_scale(
    weight_scale: torch.Tensor,
    global_scales: torch.Tensor,
    common_global_scale: torch.Tensor,
    logical_widths: list[int],
) -> torch.Tensor:
    if len(logical_widths) != global_scales.numel():
        raise ValueError(
            "MixFP4 expected one weight_global_scale per fused output partition, "
            f"got {global_scales.numel()} scales for {len(logical_widths)} partitions."
        )
    if not torch.isfinite(global_scales).all() or not (global_scales > 0).all():
        raise ValueError(
            "MixFP4 weight_global_scale values must be finite and positive."
        )

    raw = weight_scale.detach().contiguous().view(torch.uint8)
    flags = raw & 0x80
    magnitudes = (raw & 0x7F).view(torch.float8_e4m3fn).to(torch.float32)

    row_start = 0
    for width, global_scale in zip(logical_widths, global_scales):
        row_end = row_start + width
        magnitudes[row_start:row_end] *= common_global_scale / global_scale
        row_start = row_end
    if row_start != weight_scale.size(0):
        raise ValueError(
            "MixFP4 fused output partition widths do not match weight_scale rows."
        )

    magnitude_raw = magnitudes.to(torch.float8_e4m3fn).contiguous().view(torch.uint8)
    magnitude_raw &= 0x7F
    flags = torch.where(magnitude_raw != 0, flags, torch.zeros_like(flags))
    return (magnitude_raw | flags).view(torch.float8_e4m3fn)
