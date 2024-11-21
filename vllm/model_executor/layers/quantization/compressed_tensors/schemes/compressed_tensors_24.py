from typing import List, Callable, Optional
import torch

from compressed_tensors.compressors import ModelCompressor
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.parameter import ModelWeightParameter, ChannelQuantScaleParameter
from vllm import _custom_ops as ops

__all__ = ["CompressedTensors24"]

class CompressedTensors24(CompressedTensorsScheme):
    def __init__(self, model_compressor: Optional[ModelCompressor] = None, layer_name = None):
        self.model_compressor = model_compressor
        self.layer_name = layer_name
        self.quantized = True  # toggle based on the case we're running
        self.compressed = False  # toggle based on the case we're running

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    def create_weights(self, layer: torch.nn.Module, input_size: int,
                    output_partition_sizes: List[int],
                    input_size_per_partition: int,
                    params_dtype: torch.dtype, weight_loader: Callable,
                    **kwargs):
        layer.logical_widths = output_partition_sizes
        self.params_dtype=params_dtype

        # parameter to store uncompressed weight or decompressed weight
        weight = ModelWeightParameter(
            data=torch.empty(sum(output_partition_sizes),
                             input_size_per_partition,
                             dtype=torch.float8_e4m3fn),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader)

        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty((sum(output_partition_sizes), 1),
                             dtype=torch.float32),
                             output_dim=0,
                             weight_loader=weight_loader)

        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Apply any transformations to the weights after loading
        them from disk

        :param layer: The layer with the weights to be processed
        """

        w_compressed, meta = ops.cutlass_compress_entry(layer.weight)
        layer.w_compressed = torch.nn.Parameter(w_compressed, requires_grad=False)
        layer.meta = torch.nn.Parameter(meta, requires_grad=False)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns the output tensor for the layer with 2:4 
        sparse compressed weights, given the input tensor
        and bias

        :param layer: The layer with 2:4 sparse compressed 
            weights to be used for the computation
        :param x: The input tensor to the layer
        :param bias: The bias to be added to the output tensor
        :return: The output tensor of the layer 
        """
        
        q_input, input_scale = ops.scaled_fp8_quant(
            x, use_per_token_if_dynamic=True)

        out = ops.cutlass_scaled_sparse_mm(
            a=layer.w_compressed,
            e=layer.meta,
            b=q_input.t(),
            scale_a=layer.weight_scale,
            scale_b=input_scale,
            out_dtype=self.params_dtype,
            bias=bias
        )

        assert out.is_contiguous()
        return out


def check_24(tensor):
    new_tensor = tensor.view(-1, 4)    
    zero_counts = (new_tensor == 0).sum(dim=1)
    return (zero_counts >= 2).all().item()

