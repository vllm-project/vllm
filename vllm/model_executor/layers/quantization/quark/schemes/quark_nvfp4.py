# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel
from vllm.model_executor.kernels.linear.nvfp4.emulation import (
    EmulationNvFp4LinearKernel,
)
from vllm.model_executor.layers.quantization.quark.schemes.quark_scheme import (
    QuarkScheme,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)

__all__ = ["QuarkNVFP4"]

logger = init_logger(__name__)


class QuarkNVFP4(QuarkScheme):
    """
    Quark NVFP4 quantization scheme.

    Supports loading NVFP4 checkpoints with the following structure:
    - weight: uint8, shape [out_features, in_features // 2] (packed FP4)
    - weight_scale: float8_e4m3fn, shape [out_features, in_features // group_size]
    - weight_scale_2: bfloat16/float32, scalar (global weight scale)
    - input_scale_2: bfloat16/float32, scalar (global input scale)
    """

    def __init__(
        self,
    ):
        self.kernel = init_nvfp4_linear_kernel()
        self.group_size = 16

        if not isinstance(self.kernel, EmulationNvFp4LinearKernel):
            logger.warning_once(
                "Only EmulationNvFp4LinearKernel NVFP4 dense implementation is "
                "tested with QuarkNVFP4, got kernel=%s. Correctness is not validated.",
                type(self.kernel).__name__,
            )

    @classmethod
    def get_min_capability(cls) -> int:
        # FP4 requires Turing (75) or newer
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

        if input_size_per_partition % self.group_size != 0:
            raise ValueError(
                f"Input size per partition ({input_size_per_partition}) must be "
                f"divisible by group size ({self.group_size})"
            )

        # Weight: FP4 packed as uint8 (2 FP4 values per uint8)
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
        layer.register_parameter("weight", weight)

        # Per-group weight scale (FP8 E4M3)
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

        # Global weight scale (scalar, per partition)
        weight_scale_2 = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale_2", weight_scale_2)

        # Global input scale (scalar, per partition)
        input_scale_2 = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("input_scale_2", input_scale_2)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        input_global_scale = layer.input_scale_2.max().to(torch.float32)
        layer.input_global_scale = Parameter(input_global_scale, requires_grad=False)
        del layer.input_scale_2

        weight_global_scale = layer.weight_scale_2.to(torch.float32)

        if torch.unique(weight_global_scale).numel() != 1:
            logger.warning_once(
                "In NVFP4 linear, the global scale for weight are different"
                " for parallel layers (e.g. q_proj, k_proj, v_proj). This"
                " will likely result in reduced accuracy. Please verify the"
                " model accuracy. Consider using a checkpoint with a shared"
                " global NVFP4 scale for fused layers."
            )

        weight_global_scale = weight_global_scale.max()

        layer.weight_global_scale = Parameter(weight_global_scale, requires_grad=False)
        del layer.weight_scale_2

        layer.alpha = Parameter(
            layer.input_global_scale * layer.weight_global_scale, requires_grad=False
        )
        layer.input_global_scale_inv = Parameter(
            (1.0 / layer.input_global_scale).to(torch.float32), requires_grad=False
        )

        # Convert layer to NVFP4 linear kernel format
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer=layer, x=x, bias=bias)
