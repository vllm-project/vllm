# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel
from vllm.model_executor.layers.fusion.quant_activation import (
    expose_input_quant_key,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)

logger = init_logger(__name__)


__all__ = ["CompressedTensorsW4A4Fp4"]


class CompressedTensorsW4A4Fp4(CompressedTensorsScheme):
    def __init__(self, use_a16: bool = False):
        self.use_a16 = use_a16
        self.kernel = init_nvfp4_linear_kernel(use_a16=use_a16)
        self.group_size = 16

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

        if not self.use_a16:
            input_global_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            layer.register_parameter("input_global_scale", input_global_scale)

        expose_input_quant_key(layer, self.kernel)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Rename CT checkpoint names to standardized names
        layer.weight = layer.weight_packed
        del layer.weight_packed

        weight_global_scales = layer.weight_global_scale.detach().to(torch.float32)
        has_non_uniform_scales = not torch.allclose(
            weight_global_scales,
            weight_global_scales[0].expand_as(weight_global_scales),
        )

        output_partition_rescale = None

        if len(layer.logical_widths) == 2 and has_non_uniform_scales:
            # Use the minimum divisor as the common base so that all
            # post-GEMM correction factors are <= 1.
            base_divisor = weight_global_scales.min()

            output_partition_rescale = torch.empty(
                sum(layer.logical_widths),
                dtype=torch.float32,
                device=weight_global_scales.device,
            )
            offset = 0
            for width, partition_divisor in zip(
                layer.logical_widths, weight_global_scales
            ):
                output_partition_rescale[offset : offset + width] = (
                    base_divisor / partition_divisor
                )
                offset += width

            logger.warning_once(
                "Detected non-uniform NVFP4 weight global scales in a "
                "two-partition fused linear layer. Applying per-partition "
                "output rescaling."
            )
            logger.debug(
                "NVFP4 partition scales=%s, rescale=%s",
                weight_global_scales.tolist(),
                output_partition_rescale.tolist(),
            )

            weight_global_scale = base_divisor
        else:
            if has_non_uniform_scales:
                logger.warning_once(
                    "In NVFP4 linear, the weight global scale differs "
                    "across fused logical partitions. This may reduce "
                    "model accuracy."
                )
            weight_global_scale = weight_global_scales.max()

        layer._output_partition_rescale = (
            Parameter(output_partition_rescale, requires_grad=False)
            if output_partition_rescale is not None
            else None
        )

        # Process weight global scale (CT stores as divisors, i.e. 1/scale)
        layer.weight_global_scale = Parameter(
            1.0 / weight_global_scale, requires_grad=False
        )

        if not self.use_a16:
            if torch.unique(layer.input_global_scale).numel() != 1:
                logger.warning_once(
                    "In NVFP4 linear, the input global scale is different"
                    " for parallel layers (e.g. q_proj, k_proj, v_proj). This "
                    " will likely result in reduced accuracy. Please verify the model"
                    " accuracy. Consider using a checkpoint with a shared global NVFP4"
                    " scale for fused layers."
                )
            # Process input global scale and pre-compute alpha for W4A4 mode
            input_global_scale_inv = layer.input_global_scale.max().to(torch.float32)
            layer.input_global_scale = Parameter(
                (1.0 / input_global_scale_inv).to(torch.float32), requires_grad=False
            )

            # Pre-compute alpha and inverse for runtime quantization
            layer.input_global_scale_inv = Parameter(
                input_global_scale_inv, requires_grad=False
            )
            layer.alpha = Parameter(
                layer.input_global_scale * layer.weight_global_scale,
                requires_grad=False,
            )

        # Convert layer to NVFP4 linear kernel format
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rescale = getattr(layer, "_output_partition_rescale", None)
        if rescale is not None:
            out = self.kernel.apply_weights(layer=layer, x=x, bias=None)
            out = out * rescale.to(dtype=out.dtype)
            if bias is not None:
                out = out + bias
            return out
        return self.kernel.apply_weights(layer=layer, x=x, bias=bias)
