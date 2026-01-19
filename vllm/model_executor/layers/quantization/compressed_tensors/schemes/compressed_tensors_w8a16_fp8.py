# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
from compressed_tensors.quantization import QuantizationStrategy

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    prepare_fp8_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise,
)
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.platforms import current_platform

__all__ = ["CompressedTensorsW8A16Fp8"]

SUPPORTED_STRATEGIES = [QuantizationStrategy.CHANNEL, QuantizationStrategy.TENSOR]


class CompressedTensorsW8A16Fp8(CompressedTensorsScheme):
    def __init__(self, strategy: str, is_static_input_scheme: bool):
        self.strategy = strategy
        self.is_static_input_scheme = is_static_input_scheme

    @classmethod
    def get_min_capability(cls) -> int:
        if current_platform.is_xpu():
            return 0
        # ampere and up
        return 80

    # W8A8-Fp8 kernels support only per-tensor and per-channel cases.
    # So if we have a fused module (QKV, MLP) with per tensor scales,
    # we expand each scale to its shard's channels.
    def process_weights_after_loading(self, layer) -> None:
        if current_platform.is_xpu():
            return self.process_weights_after_loading_xpu(layer)
        if self.strategy == QuantizationStrategy.TENSOR:
            ws_channelwise = convert_to_channelwise(
                layer.weight_scale, layer.logical_widths
            )
            layer.weight_scale = torch.nn.Parameter(ws_channelwise, requires_grad=False)
        else:
            # required by torch.compile to be torch.nn.Parameter
            layer.weight_scale = torch.nn.Parameter(
                layer.weight_scale.data, requires_grad=False
            )

        # Weights must be transposed for marlin
        layer.weight = torch.nn.Parameter(layer.weight.t(), requires_grad=False)

        if self.is_static_input_scheme:
            # required by torch.compile to be torch.nn.Parameter
            layer.input_scale = torch.nn.Parameter(
                layer.input_scale.data, requires_grad=False
            )
        prepare_fp8_layer_for_marlin(layer)

    def process_weights_after_loading_xpu(self, layer) -> None:
        # Give xpu only support to per-tensor strategy for now
        # So if we have a fused module (QKV, MLP) with per tensor scales,
        # requantize the weights w/ max scale
        def requant_weight_per_tensor(layer):
            device = layer.weight.device
            # Get the max scale on the weight's device
            max_scale = torch.max(layer.weight_scale.data.to(device))
            orig_dtype = layer.weight.dtype
            start_idx = 0
            for index, width in enumerate(layer.logical_widths):
                end_idx = start_idx + width
                if width == 0:
                    continue

                shard_weight = layer.weight.data[start_idx:end_idx, :]
                shard_scale = layer.weight_scale.data[index].to(device)

                # Dequantize and then requantize with max_scale
                dequantized_shard = shard_weight.to(torch.float32) * shard_scale
                requantized_shard = (dequantized_shard / max_scale).to(orig_dtype)

                layer.weight.data[start_idx:end_idx, :] = requantized_shard
                start_idx = end_idx

            layer.weight_scale = torch.nn.Parameter(max_scale, requires_grad=False).to(
                device
            )

        if (
            self.strategy == QuantizationStrategy.TENSOR
            and len(layer.logical_widths) > 1
        ):
            requant_weight_per_tensor(layer)
        layer.weight = torch.nn.Parameter(layer.weight.data.t(), requires_grad=False)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size: int,
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
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        # WEIGHT
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        if self.strategy == QuantizationStrategy.CHANNEL:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader,
            )
        elif self.strategy == QuantizationStrategy.TENSOR:
            weight_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
        else:
            raise ValueError(
                f"Unsupported weight strategy={self.strategy}, "
                f"supported strategies are {SUPPORTED_STRATEGIES}"
            )

        weight_scale[:] = torch.finfo(torch.float32).min
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE (to deal with converted checkpoints)
        if self.is_static_input_scheme:
            input_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            layer.register_parameter("input_scale", input_scale)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if current_platform.is_xpu():
            return torch.ops._xpu_C.fp8_gemm_w8a16(
                x, layer.weight, layer.weight_scale, bias
            )
        return apply_fp8_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )
