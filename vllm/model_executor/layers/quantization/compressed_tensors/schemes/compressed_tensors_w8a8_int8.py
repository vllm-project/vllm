from typing import Callable, List, Optional

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    QuantizationStrategy)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    apply_int8_linear, convert_to_channelwise, create_per_channel_scale_param,
    create_per_tensor_scale_param)
from vllm.model_executor.utils import set_weight_attrs


class CompressedTensorsW8A8Int8(CompressedTensorsScheme):

    def __init__(self, strategy: str, is_static_input_scheme: bool):
        self.strategy = strategy
        self.is_static_input_scheme = is_static_input_scheme

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # WEIGHT
        # Cutlass kernels need transposed weight.
        weight = layer.weight
        layer.weight = Parameter(weight.t(), requires_grad=False)

        # WEIGHT SCALE
        # Cutlass kernels support only per-tensor and per-channel.
        # If we have a fused module (QKV, MLP) with per tensor scales (thus N
        # scales being passed to the kernel), convert to the per-channel case.
        is_fused_module = len(self.logical_widths) > 1
        if is_fused_module and self.strategy == QuantizationStrategy.TENSOR:
            ws_channelwise = convert_to_channelwise(layer.weight_scale,
                                                    self.logical_widths)
            layer.weight_scale = Parameter(ws_channelwise, requires_grad=False)

        # INPUT SCALE
        if self.is_static_input_scheme:
            layer.input_scale = Parameter(layer.input_scale.max(),
                                          requires_grad=False)
        else:
            layer.input_scale = None

    def create_weights(self, layer: torch.nn.Module,
                       output_partition_sizes: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):
        self.logical_widths = output_partition_sizes

        # WEIGHT
        weight = Parameter(torch.empty(sum(output_partition_sizes),
                                       input_size_per_partition,
                                       dtype=torch.int8),
                           requires_grad=False)
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, {
            "input_dim": 1,
            "output_dim": 0,
            "weight_loader": weight_loader,
        })

        # WEIGHT SCALE
        layer_kwargs = {"weight_loader": weight_loader}
        if self.strategy == QuantizationStrategy.CHANNEL:
            scale = create_per_channel_scale_param(output_partition_sizes,
                                                   **layer_kwargs)
        else:
            assert self.strategy == QuantizationStrategy.TENSOR
            scale = create_per_tensor_scale_param(output_partition_sizes,
                                                  **layer_kwargs)
        layer.register_parameter("weight_scale", scale)

        # INPUT SCALE
        if self.is_static_input_scheme:
            scale = create_per_tensor_scale_param(output_partition_sizes,
                                                  **layer_kwargs)
            layer.register_parameter("input_scale", scale)

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:

        return apply_int8_linear(input=x,
                                 weight=layer.weight,
                                 weight_scale=layer.weight_scale,
                                 input_scale=layer.input_scale,
                                 bias=bias)
