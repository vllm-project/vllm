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

        # Process weight global scale (CT stores as divisors, i.e. 1/scale).
        # Fused layers (e.g. qkv_proj, gate_up_proj) run one GEMM with the
        # largest divisor; output columns of shards quantized with a smaller
        # divisor are rescaled by their exact ratio in apply_weights.
        shard_divisors = layer.weight_global_scale.to(torch.float32)
        weight_global_scale = shard_divisors.max()
        layer.shard_output_scales = _shard_output_scales(
            layer.logical_widths, shard_divisors.tolist(), weight_global_scale.item()
        )
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
        if not layer.shard_output_scales:
            return self.kernel.apply_weights(layer=layer, x=x, bias=bias)
        out = self.kernel.apply_weights(layer=layer, x=x)
        for start, end, scale in layer.shard_output_scales:
            out[..., start:end] *= scale
        if bias is not None:
            out += bias
        return out


def _shard_output_scales(
    logical_widths: list[int], shard_divisors: list[float], divisor: float
) -> list[tuple[int, int, float]]:
    """Column ranges [start, end) whose shard divisor differs from `divisor`,
    with the factor that restores their own global scale."""
    ranges: list[tuple[int, int, float]] = []
    start = 0
    for width, shard_divisor in zip(logical_widths, shard_divisors):
        end = start + width
        if shard_divisor != divisor:
            scale = divisor / shard_divisor
            if ranges and ranges[-1][1] == start and ranges[-1][2] == scale:
                ranges[-1] = (ranges[-1][0], end, scale)
            else:
                ranges.append((start, end, scale))
        start = end
    return ranges
