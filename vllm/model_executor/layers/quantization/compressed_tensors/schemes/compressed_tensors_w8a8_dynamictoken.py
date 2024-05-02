from typing import Callable, List, Tuple, Union

import torch
from torch.nn import Parameter

from vllm._C import ops
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.utils import set_weight_attrs

__all__ = ["CompressedTensorsW8A8DynamicToken"]


class CompressedTensorsW8A8DynamicToken(CompressedTensorsScheme):

    def __init__(self, fake_quant: bool):
        self.fake_quant = fake_quant

    def create_weights(self, layer: torch.nn.Module,
                    output_partition_sizes: List[int],
                    input_size_per_partition: int,
                    params_dtype: torch.dtype, weight_loader: Callable,
                    **kwargs):

        weight_zero_point = Parameter(torch.empty(1,
                                                  device="cuda",
                                                  dtype=torch.int8),
                                      requires_grad=False)

        weight_scale = Parameter(torch.empty(sum(output_partition_sizes),
                                             device="cuda",
                                             dtype=torch.float32),
                                 requires_grad=False)

        weight = Parameter(torch.empty(sum(output_partition_sizes),
                                       input_size_per_partition,
                                       device="cuda",
                                       dtype=params_dtype),
                           requires_grad=False)
        
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight, {"weight_loader": weight_loader})

        layer.register_parameter("weight_scale", weight_scale)
        set_weight_attrs(weight_scale, {"weight_loader": weight_loader})

        layer.register_parameter("weight_zero_point", weight_zero_point)
        set_weight_attrs(weight_zero_point, {"weight_loader": weight_loader})

    # Determine per token input scales on the fly
    def _quantize_activation(self, x: torch.Tensor):
        x_q = torch.empty_like(x, dtype=torch.int8)
        input_scales = torch.empty()
        ops.quant(x_q, x, input_scales)
        return x_q, input_scales

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor):
        weight = layer.weight
        weight_scale = layer.weight_scale

        x_q, input_scales = self._quantize_activation(x)
        if self.fake_quant:
            pass 
        
