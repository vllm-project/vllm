# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch
from torch.nn.parameter import Parameter

# from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
#     swizzle_mxfp4_scales,
# )
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils_fp4 import (
    apply_mxfp4_flashinfer_linear,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    prepare_fp4_layer_for_marlin,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

__all__ = ["CompressedTensorsW4A4Mxfp4"]


class CompressedTensorsW4A4Mxfp4(CompressedTensorsScheme):
    """
    Compressed tensors scheme for MXFP4.

    Supports models quantized with the compressed-tensors mxfp4-pack-quantized
    format.

    MXFP4 format:
    - 4-bit float weights (E2M1) packed into uint8
    - Per-group E8M0 scales with group_size=32
    - No global scale (unlike NVFP4)

    On SM100+ with FlashInfer: true W4A4 (activations dynamically quantized).
    Otherwise: W4A16 weight-only via Marlin.
    """

    def __init__(self):
        self.group_size = 32
        p = current_platform
        self.use_flashinfer = (
            p.is_cuda() and p.is_device_capability_family(100) and has_flashinfer()
        )

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
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.params_dtype = params_dtype

        # Packed FP4 weights (2 values per byte)
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

        # Per-group E8M0 scales
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

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = Parameter(layer.weight_packed.data, requires_grad=False)
        del layer.weight_packed

        if self.use_flashinfer:
            # TODO: verify whether FlashInfer cute-dsl needs a specific
            # swizzle for checkpoint weight scales (flat [N, K//32] E8M0).
            # swizzle_mxfp4_scales targets the CUTLASS MoE tiled layout and
            # may not match FlashInfer's 128x4 layout — test first.
            # N, scale_K = layer.weight_scale.shape
            # K = scale_K * self.group_size
            # layer.weight_scale = Parameter(
            #     swizzle_mxfp4_scales(layer.weight_scale.data, N, K).reshape(N, -1),
            #     requires_grad=False,
            # )
            layer.weight_scale = Parameter(layer.weight_scale.data, requires_grad=False)
        else:
            prepare_fp4_layer_for_marlin(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_flashinfer:
            return apply_mxfp4_flashinfer_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                size_n=layer.output_size_per_partition,
                bias=bias,
            )
        return apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=None,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )
