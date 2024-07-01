from typing import Callable, List, Tuple, Union

import torch

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    QuantizationStrategy)
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)


class CompressedTensorsW8A8(CompressedTensorsScheme):

    def __init__(self, strategy: str):
        self.strategy = strategy

    def create_weights(self, layer: torch.nn.Module,
                       output_partition_sizes: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):

        is_tensor_partitioned = len(output_partition_sizes) != 1
        weight_scale_dim = sum(output_partition_sizes) if (
            is_tensor_partitioned
            or self.strategy == QuantizationStrategy.CHANNEL) else 1

        shape: Union[Tuple[int], Tuple[int, int]] = (weight_scale_dim, )
        if self.strategy == QuantizationStrategy.CHANNEL:
            shape = (weight_scale_dim, 1)

        weight = ModelWeightParameter(data=torch.empty(
            sum(output_partition_sizes),
            input_size_per_partition,
            dtype=torch.int8),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)

        weight_scale_data = torch.empty(*shape, dtype=torch.float32)
        if self.strategy == QuantizationStrategy.CHANNEL:
            weight_scale = ChannelQuantScaleParameter(
                data=weight_scale_data,
                output_dim=0,
                weight_loader=weight_loader)
        else:
            weight_scale = PerTensorScaleParameter(
                data=weight_scale_data,
                weight_loader=weight_loader,
                logical_widths=output_partition_sizes)

        layer.register_parameter("weight", weight)
        layer.register_parameter("weight_scale", weight_scale)
