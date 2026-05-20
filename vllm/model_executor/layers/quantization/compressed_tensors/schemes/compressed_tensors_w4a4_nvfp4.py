# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.nvfp4_builders import (  # noqa: E501
    NvFp4DynamicActivationBuilder,
    NvFp4StaticWeightBuilder,
)

logger = init_logger(__name__)


__all__ = ["CompressedTensorsW4A4Fp4"]


class CompressedTensorsW4A4Fp4(CompressedTensorsScheme):
    def __init__(self):
        self.kernel = init_nvfp4_linear_kernel()
        self.weight_builder = NvFp4StaticWeightBuilder()
        self.activation_builder = NvFp4DynamicActivationBuilder()
        self.group_size = self.weight_builder.group_size

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

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

        self.weight_builder.create(
            layer=layer,
            output_partition_sizes=output_partition_sizes,
            input_size_per_partition=input_size_per_partition,
            params_dtype=params_dtype,
            weight_loader=weight_loader,
            **kwargs,
        )

        self.activation_builder.create(
            layer=layer,
            output_partition_sizes=output_partition_sizes,
            input_size_per_partition=input_size_per_partition,
            params_dtype=params_dtype,
            weight_loader=weight_loader,
            **kwargs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if (
            torch.unique(layer.input_global_scale).numel() != 1
            or torch.unique(layer.weight_global_scale).numel() != 1
        ):
            logger.warning_once(
                "In NVFP4 linear, the global scale for input or weight are different"
                " for parallel layers (e.g. q_proj, k_proj, v_proj). This "
                " will likely result in reduced accuracy. Please verify the model"
                " accuracy. Consider using a checkpoint with a shared global NVFP4"
                " scale for fused layers."
            )

        self.weight_builder.post_load(layer)
        self.activation_builder.post_load(layer)
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer=layer, x=x, bias=bias)
