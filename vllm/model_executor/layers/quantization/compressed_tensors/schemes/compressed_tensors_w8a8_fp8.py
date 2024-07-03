from typing import Callable, List, Optional

import torch

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    fp8_apply, requantize_with_max_scale, create_scale_param,
    cutlass_fp8_supported)
from vllm.model_executor.utils import set_weight_attrs

__all__ = ["CompressedTensorsW8A8FP8"]


class CompressedTensorsFp8(CompressedTensorsScheme):
    def __init__(self, input_dynamic: bool):
        self.input_dynamic = input_dynamic
        self.cutlass_fp8_supported = cutlass_fp8_supported()
        
    def create_weights(self,
                       layer: torch.nn.Module,
                       output_partition_sizes: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype,
                       weight_loader: Callable,
                       **kwargs):
        
        del params_dtype

        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.process_after_load = True

        # WEIGHT
        weight = torch.nn.Parameter(torch.empty(output_size_per_partition,
                                                input_size_per_partition,
                                                dtype=torch.float8_e4m3fn),
                                    requires_grad=False)
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, {
            "input_dim": 1, "output_dim": 0,
            "weight_loader": weight_loader,
        })

        # WEIGHT SCALE
        weight_scale = create_scale_param(output_partition_sizes,
                                          weight_loader=weight_loader)
        layer.register_parameter("weight_scale", weight_scale)
        
        # INPUT SCALE
        if not self.input_dynamic:
            input_scale = create_scale_param(output_partition_sizes,
                                             weight_loader=weight_loader)
            layer.register_parameter("input_scale", input_scale)    


    def process_weights_after_loading(self, layer) -> None:
        # Dequant -> Quant with max scale.            
        max_w_scale, weight = requantize_with_max_scale(
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            logical_widths=layer.logical_widths,
        )

        # Update layer with new values.
        layer.weight = torch.nn.Parameter(weight.t(), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(max_w_scale, requires_grad=False)
        if self.input_dynamic:
            layer.input_scale = None
        else:    
            layer.input_scale = torch.nn.Parameter(layer.input_scale.max(),
                                                   requires_grad=False)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        return fp8_apply(input=x,
                         weight=layer.weight,
                         weight_scale=layer.weight_scale,
                         input_scale=layer.input_scale,
                         bias=bias,
                         cutlass_fp8_supported=self.cutlass_fp8_supported)
