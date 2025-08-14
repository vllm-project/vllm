# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Callable, Optional

import torch
from compressed_tensors.quantization import (QuantizationArgs,
                                             QuantizationStrategy,
                                             QuantizationType)
from torch.nn.parameter import Parameter

import vllm.envs as envs
from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (  # noqa: E501
    run_nvfp4_emulations)
from vllm.model_executor.parameter import (GroupQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm, has_flashinfer

logger = init_logger(__name__)

__all__ = ["is_fp4a4_nvfp4", "CompressedTensorsW4A4Fp4"]


def is_fp4a4_nvfp4(weight_quant: QuantizationArgs,
                   input_quant: Optional[QuantizationArgs]):
    if input_quant is None:
        return False

    is_tensor_group_quant = (
        weight_quant.strategy == QuantizationStrategy.TENSOR_GROUP.value
        and input_quant.strategy == QuantizationStrategy.TENSOR_GROUP.value)
    is_symmetric = weight_quant.symmetric and input_quant.symmetric

    is_group_size_16 = (weight_quant.group_size == 16
                        and input_quant.group_size == 16)
    is_float_type = (weight_quant.type == QuantizationType.FLOAT
                     and input_quant.type == QuantizationType.FLOAT.value)
    is_4_bits = weight_quant.num_bits == 4 and input_quant.num_bits == 4

    return (is_tensor_group_quant and is_float_type and is_4_bits
            and is_group_size_16 and is_symmetric)


class CompressedTensorsW4A4Fp4(CompressedTensorsScheme):

    def __init__(self):
        if envs.VLLM_USE_TRTLLM_FP4_GEMM:
            assert has_flashinfer(), "TRTLLM FP4 GEMM requires FlashInfer"
            self.backend = "flashinfer-trtllm"
        elif has_flashinfer():
            self.backend = "flashinfer-cutlass"
        else:
            self.backend = "cutlass"
        self.group_size = 16

    @classmethod
    def get_min_capability(cls) -> int:
        if envs.VLLM_USE_NVFP4_CT_EMULATIONS:
            return 80
        return 100

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

        if self.backend == "flashinfer-trtllm":
            # FlashInfer TRTLLM FP4 GEMM requires a different weight layout.
            # FlashInfer provides nvfp4_quantize to quantize + shuffle the
            # layout but we use our own quantization so we have to call
            # shuffles ourselves.
            from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

            weight = layer.weight_packed.data
            weight_scale = layer.weight_scale.data

            epilogue_tile_m = 128
            weight = shuffle_matrix_a(weight.view(torch.uint8),
                                      epilogue_tile_m)
            weight_scale = (shuffle_matrix_sf_a(weight_scale.view(
                torch.uint8), epilogue_tile_m).reshape(
                    weight_scale.shape).view(torch.float8_e4m3fn))

            layer.weight_scale_swizzled = Parameter(weight_scale,
                                                    requires_grad=False)
            layer.weight_packed = Parameter(weight, requires_grad=False)
        else:
            swizzled_weight_scale = self.swizzle_blockscale(layer.weight_scale)
            layer.weight_scale_swizzled = Parameter(swizzled_weight_scale,
                                                    requires_grad=False)
            layer.weight_packed = Parameter(layer.weight_packed.data,
                                            requires_grad=False)

        layer.alpha = Parameter(
            1 / (layer.input_global_scale * layer.weight_global_scale),
            requires_grad=False)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        if envs.VLLM_USE_NVFP4_CT_EMULATIONS:
            out = run_nvfp4_emulations(
                x=x,
                input_global_scale=layer.input_global_scale,
                weight=layer.weight_packed,
                weight_scale_swizzled=layer.weight_scale_swizzled,
                weight_global_scale=layer.weight_global_scale)
            if bias is not None:
                out = out + bias
            return out

        output_dtype = x.dtype
        output_shape = [x.shape[0], layer.weight_packed.shape[0]]

        # quantize BF16 or FP16 to (FP4 and interleaved block scale)
        x_fp4, x_blockscale = scaled_fp4_quant(x, layer.input_global_scale)

        mm_args = (x_fp4, layer.weight_packed, x_blockscale,
                   layer.weight_scale_swizzled, layer.alpha, output_dtype)
        if self.backend == "flashinfer-trtllm":
            out = flashinfer_scaled_fp4_mm(*mm_args, backend="trtllm")
        elif self.backend == "flashinfer-cutlass":
            out = flashinfer_scaled_fp4_mm(*mm_args, backend="cutlass")
        else:
            out = cutlass_scaled_fp4_mm(*mm_args)

        if bias is not None:
            out = out + bias
        return out.view(*output_shape)
