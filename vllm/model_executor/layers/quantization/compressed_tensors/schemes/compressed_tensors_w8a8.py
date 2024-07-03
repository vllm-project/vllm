from typing import Callable, List, Tuple, Union

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    QuantizationStrategy)
from vllm.model_executor.utils import set_weight_attrs



class CompressedTensorsW8A8(CompressedTensorsScheme):

    def __init__(self,
                 strategy: QuantizationStrategy):
        self.strategy = strategy

    # If fused module, we have N scales for N fused weights, but Cutlass kernels 
    # support only per-tensor and per-channel cases. So we need to handle this
    # by converting the the N per-tensor scales to channelwise.
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        is_fused_module = len(self.logical_widths) > 1
        if (self.strategy == QuantizationStrategy.TENSOR
                and is_fused_module):

            # Load the N per-tensor scales into the channelwise buffer.
            weight_scale_channel = torch.empty(
                (sum(self.logical_widths), 1),
                dtype=torch.float32,
                device=layer.weight_scale.device)
            start = 0
            for idx, logical_width in enumerate(self.logical_widths):
                end = start + logical_width
                weight_scale_channel[start:end, :] = layer.weight_scale[idx]
                start = end

            layer.weight_scale = Parameter(weight_scale_channel,
                                           requires_grad=False)

    def create_weights(self, layer: torch.nn.Module,
                       output_partition_sizes: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):

        self.logical_widths = output_partition_sizes

        # WEIGHT SCALE
        shape: Union[Tuple[int], Tuple[int, int]]
        if self.strategy == QuantizationStrategy.CHANNEL:
            shape = (sum(self.logical_widths), 1)
        else:
            shape = (len(self.logical_widths), )

        weight_scale = Parameter(torch.empty(*shape, dtype=torch.float32),
                                 requires_grad=False)
        layer.register_parameter("weight_scale", weight_scale)
        if self.strategy == QuantizationStrategy.CHANNEL:
            set_weight_attrs(weight_scale, {
                "weight_loader": weight_loader,
                "output_dim": 0,
            })
        else:
            set_weight_attrs(weight_scale, {
                "weight_loader": weight_loader,
                "needs_scalar_to_array": True,
            })

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
