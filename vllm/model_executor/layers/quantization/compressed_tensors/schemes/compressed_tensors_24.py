from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.parameter import ModelWeightParameter
import torch
from typing import List, Callable, Optional

__all__ = ["CompressedTensors24"]

class CompressedTensors24(CompressedTensorsScheme):
    def __init__(self):
        pass
    
    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    def create_weights(self, layer: torch.nn.Module, input_size: int,
                    output_partition_sizes: List[int],
                    input_size_per_partition: int,
                    params_dtype: torch.dtype, weight_loader: Callable,
                    **kwargs):
        

        print("output_partition_sizes",output_partition_sizes)
        print("input_size_per_partition", input_size_per_partition)
        print("\n")

        # packed dim is dim 1/along input dim
        weight = ModelWeightParameter(data=torch.empty(
            sum(output_partition_sizes),
            input_size_per_partition // 2,
            dtype=torch.float8_e4m3fn),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader)

        # meta dim changes based on dtype
        meta = ModelWeightParameter(data=torch.empty(
            sum(output_partition_sizes), 
            input_size_per_partition // 16,
            dtype=torch.int32),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader)

        # per tensor quantization ---> map to channel?
        """
        weight_scale = torch.nn.Parameter()
        input_scale = torch.nn.Parameter()
        """
        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("meta", meta)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Any preprocessing for the kernel
        # e.g mapp per tensor scales to channel
        # apply marlin format to the weights before kernel call
        pass 

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        print("in forward")
        breakpoint()
                