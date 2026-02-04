# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm import (
    FP4ScaledMMLinearKernel,
    init_fp4_linear_kernel,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    NvFp4LinearBackend,
    apply_nvfp4_linear,
    convert_to_nvfp4_linear_kernel_format,
    select_nvfp4_linear_backend,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)

__all__ = ["CompressedTensorsW4A4Fp4"]


class CompressedTensorsW4A4Fp4(CompressedTensorsScheme):
    def __init__(self):
        self.backend = select_nvfp4_linear_backend()
        self.group_size = 16

        # Initialize the appropriate FP4 kernel (unless using Marlin)
        self.kernel: FP4ScaledMMLinearKernel | None
        if self.backend != NvFp4LinearBackend.MARLIN:
            # Extract backend name for FlashInfer variants
            backend_name = None
            if self.backend.value.startswith("flashinfer-"):
                backend_name = self.backend.value[len("flashinfer-") :]

            self.kernel = init_fp4_linear_kernel(
                group_size=self.group_size,
                is_checkpoint_fp4_serialized=True,
                out_dtype=None,
                backend=backend_name,
                module_name="CompressedTensorsW4A4Fp4",
            )
        else:
            self.kernel = None

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

        input_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("input_global_scale", input_global_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Rename CT checkpoint names to standardized names
        layer.weight = layer.weight_packed
        del layer.weight_packed
        # Process global scales (CT stores as divisors, i.e. 1/scale)
        input_global_scale_inv = layer.input_global_scale.max().to(torch.float32)
        layer.input_global_scale = Parameter(
            (1.0 / input_global_scale_inv).to(torch.float32), requires_grad=False
        )
        weight_global_scale = layer.weight_global_scale.max().to(torch.float32)
        layer.weight_global_scale = Parameter(
            1.0 / weight_global_scale, requires_grad=False
        )

        # Pre-compute alpha and inverse for runtime quantization
        layer.input_global_scale_inv = Parameter(
            input_global_scale_inv, requires_grad=False
        )
        layer.alpha = Parameter(
            layer.input_global_scale * layer.weight_global_scale, requires_grad=False
        )

        # Convert layer to NVFP4 linear kernel format
        convert_to_nvfp4_linear_kernel_format(self.backend, layer)

        # Initialize kernel weights if using kernel abstraction
        if self.kernel is not None:
            self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Marlin uses a special path
        if self.backend == NvFp4LinearBackend.MARLIN:
            return apply_nvfp4_linear(
                backend=self.backend,
                layer=layer,
                x=x,
                bias=bias,
            )

        # Use kernel abstraction for other backends
        if self.kernel is None:
            raise RuntimeError("FP4 kernel not initialized for non-Marlin backend")
        return self.kernel.apply_weights(layer, x, bias)
