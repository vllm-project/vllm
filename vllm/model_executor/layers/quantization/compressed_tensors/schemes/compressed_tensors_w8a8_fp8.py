# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from compressed_tensors.quantization import (QuantizationArgs,
                                             QuantizationStrategy)
from torch.nn import Parameter

import vllm.envs as envs
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp, cutlass_block_fp8_supported, maybe_create_device_identity,
    normalize_e4m3fn_to_e4m3fnuz, requantize_with_max_scale)
from vllm.model_executor.parameter import (BlockQuantScaleParameter,
                                           ChannelQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)
from vllm.platforms import current_platform

__all__ = ["CompressedTensorsW8A8Fp8"]


class CompressedTensorsW8A8Fp8(CompressedTensorsScheme):

    def __init__(self, weight_quant: QuantizationArgs,
                 is_static_input_scheme: bool):
        self.weight_quant = weight_quant
        self.strategy = weight_quant.strategy
        self.out_dtype = torch.get_default_dtype()
        self.is_static_input_scheme = is_static_input_scheme
        self.fp8_linear = Fp8LinearOp(use_per_token_if_dynamic=True)

        if self.weight_quant.block_structure is None:
            block_size = None
        elif isinstance(self.weight_quant.block_structure, str):
            block_size = [
                int(x) for x in self.weight_quant.block_structure.split("x")
            ]
        else:
            block_size = self.weight_quant.block_structure
        self.weight_block_size = block_size
        self.cutlass_block_fp8_supported = cutlass_block_fp8_supported()
        # AITER is only supported on ROCm and only for FP8_FNUZ
        # and at the moment are MI300 series
        self.use_aiter_and_is_supported = (current_platform.is_rocm()
                                           and envs.VLLM_ROCM_USE_AITER
                                           and envs.VLLM_ROCM_USE_AITER_LINEAR
                                           and current_platform.is_fp8_fnuz())

    @classmethod
    def get_min_capability(cls) -> int:
        # lovelace and up
        return 89

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       weight_loader: Callable, **kwargs):
        maybe_create_device_identity()

        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.weight_block_size = None

        if self.strategy == QuantizationStrategy.BLOCK:
            tp_size = get_tensor_model_parallel_world_size()
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            block_n, block_k = (
                layer.weight_block_size[0],
                layer.weight_block_size[1],
            )
            # Required by row parallel
            if (tp_size > 1
                    and input_size // input_size_per_partition == tp_size
                    and input_size_per_partition % block_k != 0):
                raise ValueError(
                    f"Weight input_size_per_partition = "
                    f"{input_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}.")
            # Required by column parallel or enabling merged weights
            if (tp_size > 1 and output_size // output_size_per_partition
                    == tp_size) or len(output_partition_sizes) > 1:
                for output_partition_size in output_partition_sizes:
                    if output_partition_size % block_n != 0:
                        raise ValueError(
                            f"Weight output_partition_size = "
                            f"{output_partition_size} is not divisible by "
                            f"weight quantization block_n = {block_n}.")

        # WEIGHT
        weight = ModelWeightParameter(data=torch.empty(
            output_size_per_partition,
            input_size_per_partition,
            dtype=torch.float8_e4m3fn),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        # TODO: update create_xxx_parameter functions to return
        # the newly added parameters
        if self.strategy == QuantizationStrategy.CHANNEL:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1),
                                 dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader)
        elif self.strategy == QuantizationStrategy.BLOCK:
            assert self.is_static_input_scheme is False
            weight_scale = BlockQuantScaleParameter(
                data=torch.empty(
                    (output_size_per_partition + block_n - 1) // block_n,
                    (input_size_per_partition + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
        else:
            assert self.strategy == QuantizationStrategy.TENSOR
            weight_scale = PerTensorScaleParameter(data=torch.empty(
                len(output_partition_sizes), dtype=torch.float32),
                                                   weight_loader=weight_loader)

        # min requirement for fp8 kernels
        weight_scale[:] = torch.finfo(torch.float32).min
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE
        if self.is_static_input_scheme:
            input_scale = PerTensorScaleParameter(data=torch.empty(
                len(output_partition_sizes), dtype=torch.float32),
                                                  weight_loader=weight_loader)
            input_scale[:] = torch.finfo(torch.float32).min
            layer.register_parameter("input_scale", input_scale)

    def _maybe_pad_weight(self, weight: torch.Tensor) -> torch.Tensor:
        # Pad the weight tensor. This is an optimization on ROCm platform, which
        # can benefit from tensors located far enough from one another in memory
        if (envs.VLLM_ROCM_FP8_PADDING and current_platform.is_rocm()
                and weight.stride(-1) == 1
                and (weight.stride(-2) * weight.element_size()) % 512 == 0):
            num_pad = 256 // weight.element_size()
            weight = F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]
            torch.cuda.empty_cache()
        return weight

    def process_weights_after_loading(self, layer) -> None:
        # If per tensor, when we have a fused module (e.g. QKV) with per
        # tensor scales (thus N scales being passed to the kernel),
        # requantize so we can always run per tensor
        if self.strategy == QuantizationStrategy.TENSOR:
            max_w_scale, weight = requantize_with_max_scale(
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                logical_widths=layer.logical_widths,
            )

            if current_platform.is_fp8_fnuz():
                input_scale = getattr(layer, 'input_scale', None)

                weight, max_w_scale, input_scale = normalize_e4m3fn_to_e4m3fnuz(
                    weight=weight,
                    weight_scale=max_w_scale,
                    input_scale=input_scale)
                if input_scale is not None:
                    layer.input_scale = Parameter(input_scale,
                                                  requires_grad=False)

            layer.weight = Parameter(weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(max_w_scale, requires_grad=False)

        # If channelwise, scales are already lined up, so just transpose.
        elif self.strategy == QuantizationStrategy.CHANNEL:
            weight = layer.weight

            if current_platform.is_fp8_fnuz():
                input_scale = getattr(layer, 'input_scale', None)

                weight, weight_scale, input_scale = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        weight=weight,
                        weight_scale=layer.weight_scale,
                        input_scale=input_scale)
                if input_scale is not None:
                    layer.input_scale = Parameter(input_scale,
                                                  requires_grad=False)
            else:
                weight_scale = layer.weight_scale.data

            # required by torch.compile to be torch.nn.Parameter
            layer.weight = Parameter(weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)

        elif self.strategy == QuantizationStrategy.BLOCK:
            assert self.is_static_input_scheme is False
            weight = layer.weight.data

            if current_platform.is_fp8_fnuz():
                weight, weight_scale, _ = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        weight=weight,
                        weight_scale=layer.weight_scale)
            else:
                weight_scale = layer.weight_scale.data

            weight = self._maybe_pad_weight(weight)

            # required by torch.compile to be torch.nn.Parameter
            layer.weight = Parameter(weight, requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)

        else:
            raise ValueError(f"Unknown quantization strategy {self.strategy}")

        # INPUT SCALE
        if self.is_static_input_scheme and hasattr(layer, 'input_scale'):
            layer.input_scale = Parameter(layer.input_scale.max(),
                                          requires_grad=False)
        else:
            layer.input_scale = None

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        if layer.weight_block_size is not None:
            return torch.ops.vllm.apply_w8a8_block_fp8_linear(
                input=x,
                weight=layer.weight,
                block_size=layer.weight_block_size,
                weight_scale=layer.weight_scale,
                input_scale=layer.input_scale,
                bias=bias,
                cutlass_block_fp8_supported=self.cutlass_block_fp8_supported,
                use_aiter_and_is_supported=self.use_aiter_and_is_supported,
            )

        return self.fp8_linear.apply(input=x,
                                     weight=layer.weight,
                                     weight_scale=layer.weight_scale,
                                     out_dtype=self.out_dtype,
                                     input_scale=layer.input_scale,
                                     bias=bias)
