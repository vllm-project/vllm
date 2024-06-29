from typing import Callable, List

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.utils import set_weight_attrs

__all__ = ["CompressedTensorsUnquantized"]


class CompressedTensorsUnquantized(CompressedTensorsScheme):
    """
    Implements the scheme for all layers which are ignored 
    in the CompressedTensors config. The input and loaded weight are used 
    in a linear transformation.
    """

    def create_weights(self, layer: torch.nn.Module,
                       output_partition_sizes: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):

        weight = Parameter(torch.empty(sum(output_partition_sizes),
                                       input_size_per_partition,
                                       device="cuda",
                                       dtype=params_dtype),
                           requires_grad=False)

        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, {"weight_loader": weight_loader})

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor):
        weight = layer.weight
        return F.linear(x, weight)
