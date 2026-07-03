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

        # Handle per-half weight global scales for fused gate_up_proj layers.
        # When gate_proj and up_proj have independent global scales (e.g. from
        # llmcompressor with per-tensor observers), using a single max() would
        # over-scale one half. We pre-compute a rescale vector instead.
        ws2 = layer.weight_global_scale.data.float()
        num_partitions = len(layer.logical_widths)

        if num_partitions == 2 and not torch.allclose(ws2[0:1], ws2[1:2]):
            # CT stores divisors: min(divisor) corresponds to max(absmax),
            # which is the correct unified base for the fused GEMM alpha.
            weight_global_scale = ws2.min().to(torch.float32)
            gate_size = layer.logical_widths[0]
            up_size = layer.logical_widths[1]
            rescale = torch.ones(
                gate_size + up_size,
                dtype=torch.float32,
                device=layer.weight_global_scale.device,
            )
            rescale[:gate_size] = ws2.min() / ws2[0]
            rescale[gate_size:gate_size + up_size] = ws2.min() / ws2[1]
            layer._per_half_rescale = Parameter(rescale, requires_grad=False)
            logger.info(
                "NVFP4 per-half scale detected: gate_s2=%.1f, up_s2=%.1f, "
                "rescale factors=[%.4f, %.4f]",
                ws2[0].item(),
                ws2[1].item(),
                rescale[0].item(),
                rescale[gate_size].item(),
            )
        else:
            if torch.unique(ws2).numel() != 1:
                logger.warning_once(
                    "In NVFP4 linear, the weight global scale is different"
                    " for parallel layers (e.g. q_proj, k_proj, v_proj). This"
                    " will likely result in reduced accuracy. Please verify the"
                    " model accuracy. Consider using a checkpoint with a shared"
                    " global NVFP4 scale for fused layers."
                )
            weight_global_scale = ws2.max().to(torch.float32)
            layer._per_half_rescale = None

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
        rescale = getattr(layer, "_per_half_rescale", None)
        if rescale is not None:
            out = self.kernel.apply_weights(layer=layer, x=x, bias=None)
            out = out * rescale.to(out.dtype)
            if bias is not None:
                out = out + bias
            return out
        return self.kernel.apply_weights(layer=layer, x=x, bias=bias)
