# SPDX-License-Identifier: Apache-2.0
from typing import Callable, Optional

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (  # noqa: E501
    dequantize_to_dtype, ref_nvfp4_quant)
from vllm.model_executor.parameter import (GroupQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)

__all__ = ["CompressedTensorsW4A4Fp4"]


class CompressedTensorsW4A4Fp4(CompressedTensorsScheme):

    def __init__(self):
        self.group_size = 16

    @classmethod
    def get_min_capability(cls) -> int:
        # dont restrict as emulations
        return 80

    def create_weights(self, layer: torch.nn.Module,
                       output_partition_sizes: list[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # Weight
        weight = ModelWeightParameter(data=torch.empty(
            sum(output_partition_sizes),
            input_size_per_partition // 2,
            dtype=torch.uint8),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)
        layer.register_parameter("weight_packed", weight)

        # Global Weight Scale
        weight_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader)
        layer.register_parameter("weight_global_scale", weight_global_scale)

        # Per Group Weight Scale
        weight_scale = GroupQuantScaleParameter(data=torch.empty(
            sum(output_partition_sizes),
            input_size_per_partition // self.group_size,
            dtype=torch.float8_e4m3fn,
        ),
                                                input_dim=1,
                                                output_dim=0,
                                                weight_loader=weight_loader)

        layer.register_parameter("weight_scale", weight_scale)

        input_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader)
        layer.register_parameter("input_global_scale", input_global_scale)

    def swizzle_blockscale(self, scale: torch.tensor):
        assert (scale.dtype == torch.float8_e4m3fn)
        # Pad and blockwise interleave weight_scale
        scale_ndim = scale.ndim
        if scale.ndim == 2:
            scale = scale.unsqueeze(0)
        assert scale.ndim == 3
        B, M, K = scale.shape
        round_up_multiple = lambda x, m: (x + m - 1) // m * m
        M_padded = round_up_multiple(M, 128)
        K_padded = round_up_multiple(K, 4)
        padded_scale = torch.zeros((B, M_padded, K_padded), dtype=scale.dtype)
        padded_scale[:B, :M, :K] = scale
        batches, rows, cols = padded_scale.shape
        assert rows % 128 == 0
        assert cols % 4 == 0
        padded_scale = padded_scale.reshape(batches, rows // 128, 4, 32,
                                            cols // 4, 4)
        swizzled_scale = padded_scale.permute((0, 1, 4, 3, 2, 5))
        swizzled_scale = swizzled_scale.contiguous().cuda()
        return (swizzled_scale.reshape(M, K)
                if scale_ndim == 2 else swizzled_scale.reshape(B, M, K))

    def process_weights_after_loading(self, layer) -> None:

        global_input_scale = layer.input_global_scale.max().to(torch.float32)
        layer.input_global_scale = Parameter(global_input_scale,
                                             requires_grad=False)

        layer.weight_global_scale = Parameter(
            layer.weight_global_scale.max().to(torch.float32),
            requires_grad=False)

        swizzled_weight_scale = self.swizzle_blockscale(layer.weight_scale)
        layer.weight_scale_swizzled = Parameter(swizzled_weight_scale,
                                                requires_grad=False)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        x_m, x_k = x.shape
        output_dtype = x.dtype

        # quantize input to (FP4 and interleaved block scale)
        x_global_scale = layer.input_global_scale
        x_fp4, x_blockscale = ref_nvfp4_quant(x, x_global_scale,
                                              self.group_size)

        # dequantize input
        x_fp4 = x_fp4.reshape(x_m, x_k // self.group_size, self.group_size)
        x_blockscale = x_blockscale.unsqueeze(-1) / x_global_scale
        x_dq = (x_fp4 * x_blockscale).reshape(x_m, x_k).to(output_dtype)
        del x_fp4, x_blockscale

        # dequantize weight
        w_fp4 = layer.weight_packed.data.view(torch.uint8)
        w_blockscale = layer.weight_scale_swizzled.data
        w_global_scale = layer.weight_global_scale
        w_dq = dequantize_to_dtype(w_fp4, w_blockscale, w_global_scale,
                                   output_dtype, x.device, self.group_size)

        # matmul
        out = torch.matmul(x_dq, w_dq.t())
        del w_dq, x_dq
        return out
