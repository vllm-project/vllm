from typing import Callable, List, Optional

import torch

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    QuantizationStrategy)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear, prepare_fp8_layer_for_marlin)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise, create_per_channel_scale_param,
    create_per_tensor_scale_param)
from vllm.model_executor.utils import set_weight_attrs

__all__ = ["CompressedTensorsW8A16Fp8"]


class CompressedTensorsW8A16Fp8(CompressedTensorsScheme):

    def __init__(self, strategy: str):
        self.strategy = strategy

    def get_min_capability(self):
        # ampere and up
        return 80

    # W8A8-Fp8 kernels support only per-tensor and per-channel cases.
    # So if we have a fused module (QKV, MLP) with per tensor scales (thus N
    # scales being passed to the kernel), we requantize with a single scale.
    def process_weights_after_loading(self, layer) -> None:
        if self.strategy == QuantizationStrategy.TENSOR:
            ws_channelwise = convert_to_channelwise(layer.weight_scale,
                                                    layer.logical_widths)
            layer.weight_scale = torch.nn.Parameter(ws_channelwise,
                                                    requires_grad=False)

        prepare_fp8_layer_for_marlin(layer, strategy="channel")

    def create_weights(self, layer: torch.nn.Module,
                       output_partition_sizes: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):

        del params_dtype

        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # WEIGHT
        weight = torch.nn.Parameter(torch.empty(output_size_per_partition,
                                                input_size_per_partition,
                                                dtype=torch.float8_e4m3fn),
                                    requires_grad=False)
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, {
            "input_dim": 1,
            "output_dim": 0,
            "weight_loader": weight_loader,
        })

        # WEIGHT SCALE
        if self.strategy == QuantizationStrategy.CHANNEL:
            weight_scale = create_per_channel_scale_param(
                output_partition_sizes, weight_loader=weight_loader)
        else:
            assert self.strategy == QuantizationStrategy.TENSOR
            weight_scale = create_per_tensor_scale_param(
                output_partition_sizes, weight_loader=weight_loader)
        layer.register_parameter("weight_scale", weight_scale)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        return apply_fp8_marlin_linear(input=x,
                                       weight=layer.weight,
                                       weight_scale=layer.weight_scale,
                                       workspace=layer.workspace,
                                       size_n=layer.output_size_per_partition,
                                       size_k=layer.input_size_per_partition,
                                       bias=bias)
