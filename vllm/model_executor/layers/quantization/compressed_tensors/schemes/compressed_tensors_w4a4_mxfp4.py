# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    prepare_fp4_layer_for_marlin,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
)

__all__ = ["CompressedTensorsW4A4MXFp4"]


class CompressedTensorsW4A4MXFp4(CompressedTensorsScheme):
    """
    Compressed tensors scheme for MXFP4 weight-only quantization.

    Supports models quantized with the compressed-tensors mxfp4-pack-quantized
    format.

    MXFP4 format:
    - 4-bit float weights (E2M1) packed into uint8
    - Per-group E8M0 scales with group_size=32
    - No global scale (unlike NVFP4)
    """

    def __init__(self, use_marlin: bool = False):
        self.group_size = 32
        self.use_marlin = use_marlin
     
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
        # Rename weight_packed to weight that marlin expects
        layer.weight = Parameter(layer.weight_packed.data, requires_grad=False)
        del layer.weight_packed

        """
        if self.use_marlin:
            prepare_fp4_layer_for_marlin(layer)
        else:
            # Pre-compile flashinfer modules to avoid JIT compilation during torch.compile
        """
    
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        if self.use_marlin:
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
        """

        from flashinfer import mxfp4_quantize

        from vllm.utils.flashinfer import flashinfer_mm_fp4

        # Reshape input to 2D
        input_shape = x.shape
        x_2d = x.view(-1, input_shape[-1])

        # Step 1: Quantize activations to MXFP4 (4-bit E2M1, block_size=32)
        # Returns: (packed_uint8 [M, K//2], E8M0_scales_uint8 [M, K//32])
        x_mxfp4_packed, x_scales_e8m0 = mxfp4_quantize(x_2d)

 
        # Need to transpose weight to [K//2, N]
        output = flashinfer_mm_fp4(
            A=x_mxfp4_packed,  # [M, K//2]
            B=layer.weight.t(),  # [K//2, N]
            A_scale=x_scales_e8m0,  # [M, K//32]
            B_scale=layer.weight_scale.t(),  # [K//32, N]
            g_scale=None,  # No global scale for MXFP4
            dtype=torch.bfloat16,
            backend="auto",
            block_size=self.group_size,
            use_8x4_sf_layout=False,
            use_nvfp4=False
        )

        # Add bias if present
        if bias is not None:
            output = output + bias

        # Reshape output back to original batch dimensions
        return output.view(*input_shape[:-1], -1)
