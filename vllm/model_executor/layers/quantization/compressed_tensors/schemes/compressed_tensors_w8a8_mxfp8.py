# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch

from vllm.model_executor.kernels.linear import (
    MXFP8LinearLayerConfig,
    choose_mxfp8_linear_kernel,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_weight_parameter,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
)
from vllm.platforms import current_platform

__all__ = ["CompressedTensorsW8A8MXFp8"]


class CompressedTensorsW8A8MXFp8(CompressedTensorsScheme):
    def __init__(self):
        self.group_size = MXFP8_BLOCK_SIZE

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        if input_size_per_partition % MXFP8_BLOCK_SIZE != 0:
            raise ValueError(
                f"MXFP8 requires input dimension to be divisible by "
                f"{MXFP8_BLOCK_SIZE}, got {input_size_per_partition}"
            )

        mxfp8_linear_kernel_config = MXFP8LinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(
                input_size_per_partition,
                output_size_per_partition,
            ),
            weight_type=current_platform.fp8_dtype(),
            act_type=params_dtype,
        )
        kernel_type = choose_mxfp8_linear_kernel(mxfp8_linear_kernel_config)

        # WEIGHT
        weight = create_fp8_weight_parameter(
            output_size_per_partition, input_size_per_partition, weight_loader
        )
        layer.register_parameter("weight", weight)

        # Per Group Weight Scale (MXFP8 uses E8M0 format for scales)
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        self.kernel = kernel_type(
            mxfp8_linear_kernel_config,
            w_q_param_name="weight",
            w_s_param_name="weight_scale",
        )

    def process_weights_after_loading(self, layer) -> None:
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)
