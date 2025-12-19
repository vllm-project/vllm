# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    is_fp4_marlin_supported,
    prepare_fp4_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    swizzle_blockscale,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)

logger = init_logger(__name__)

__all__ = ["CompressedTensorsW4A16Fp4"]


class CompressedTensorsW4A16Fp4(CompressedTensorsScheme):
    def __init__(self, has_input_global_scale: bool = False):
        self.has_input_global_scale = has_input_global_scale
        self.group_size = 16

        self.use_marlin = is_fp4_marlin_supported()
        self.use_emulation = not self.use_marlin

        if self.use_emulation:
            logger.warning_once(
                "Marlin FP4 kernels not available. "
                "Falling back to emulation mode (dequantize weights to FP16). "
                "For better performance, use NVIDIA Ampere+ GPUs."
            )
        else:
            logger.info_once("Using Marlin backend for NVFP4 W4A16 linear layers.")

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

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
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # Weight
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        # Global Weight Scale
        weight_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_global_scale", weight_global_scale)

        # Per Group Weight Scale
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight_scale", weight_scale)

        if self.has_input_global_scale:
            input_global_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            layer.register_parameter("input_global_scale", input_global_scale)

    def process_weights_after_loading(self, layer) -> None:
        if self.use_emulation:
            layer.weight = Parameter(layer.weight_packed.data, requires_grad=False)
            del layer.weight_packed

            layer.weight_global_scale = Parameter(
                layer.weight_global_scale.max().to(torch.float32), requires_grad=False
            )

            swizzled_weight_scale = swizzle_blockscale(layer.weight_scale)
            layer.weight_scale = Parameter(swizzled_weight_scale, requires_grad=False)

            if self.has_input_global_scale:
                layer.input_global_scale = Parameter(
                    layer.input_global_scale.data, requires_grad=False
                )
        else:
            layer.weight = Parameter(layer.weight_packed.data, requires_grad=False)
            del layer.weight_packed

            layer.weight_scale_2 = Parameter(
                1 / layer.weight_global_scale.max().to(torch.float32),
                requires_grad=False,
            )
            del layer.weight_global_scale

            if self.has_input_global_scale:
                layer.input_global_scale = Parameter(
                    layer.input_global_scale.data, requires_grad=False
                )

            prepare_fp4_layer_for_marlin(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_emulation:
            from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
                dequantize_to_dtype,
            )

            weight_dequant = dequantize_to_dtype(
                layer.weight,
                layer.weight_scale,
                layer.weight_global_scale,
                x.dtype,
                x.device,
                self.group_size,
            )
            out = torch.matmul(x, weight_dequant.t())

            if bias is not None:
                out = out + bias
            return out
        else:
            return apply_fp4_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                weight_scale_2=layer.weight_scale_2,
                workspace=layer.workspace,
                size_n=layer.output_size_per_partition,
                size_k=layer.input_size_per_partition,
                bias=bias,
            )
