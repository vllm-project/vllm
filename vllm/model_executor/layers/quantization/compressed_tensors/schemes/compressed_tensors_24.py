from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.parameter import ModelWeightParameter
import torch
from typing import List, Callable, Optional
from compressed_tensors.compressors import ModelCompressor
from torch.nn import Parameter
from vllm.model_executor.layers.sparsity.utils.cusparse_2_4_utils import (
    compress_to_torch_sparse_semi_structured_mat,
    semi_structured_dense_sparse_T_gemm
    )

__all__ = ["CompressedTensors24"]

class CompressedTensors24(CompressedTensorsScheme):
    def __init__(self, model_compressor: Optional[ModelCompressor] = None):
        self.model_compressor = model_compressor
        
    
    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    def create_weights(self, layer: torch.nn.Module, input_size: int,
                    output_partition_sizes: List[int],
                    input_size_per_partition: int,
                    params_dtype: torch.dtype, weight_loader: Callable,
                    **kwargs):
        
        weights_dtype = params_dtype
        weights = ModelWeightParameter(data=torch.empty(
            sum(output_partition_sizes),
            input_size_per_partition // 2,
            dtype=weights_dtype),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader)

        bits_per_weight_element = weights.itemsize * 8 
        
        meta_dtype = torch.int32 if bits_per_weight_element == 8 else torch.int16


        meta_input_size = (
            input_size_per_partition // 32
            if bits_per_weight_element == 8
            else input_size_per_partition // 16
        )
        meta = ModelWeightParameter(data=torch.empty(
            sum(output_partition_sizes), 
            meta_input_size,
            dtype=meta_dtype),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader)

        # TODO: replace weight_packed name, with something
        # more meaningful, like sparse24_packed, this will
        # require changes on compressed_tensors side

        layer.register_parameter("weight_packed", weights)
        layer.register_parameter("meta", meta)
        

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Apply any transformations to the weights after loading
        them from disk

        :param layer: The layer with the weights to be processed
        """
        
        # TODO: right now this is hard coded for 24 compressor
        # replace by a better way to identify targetted params
        # using COMPRESSION_PARAMS defined by sparse compressors
        # and decompress the weights accordingly

        if hasattr(layer, "weight_packed"):
            # TODO: this name will also be changed to sparse24_packed
            weight = layer.weight_packed.data
            meta = layer.meta.data

            weight_data = {
                "weight_packed": weight,
                "meta": meta
            }

            decompressed_weight = self.model_compressor.sparsity_compressor.decompress_weight(weight_data)
            decompressed_weight = decompressed_weight
            compressed = compress_to_torch_sparse_semi_structured_mat(decompressed_weight)
            layer.weight_packed = Parameter(compressed, requires_grad=False)

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
        result = semi_structured_dense_sparse_T_gemm(
            a_dense=x,
            b_T_packed=layer.weight_packed.data, 
            bias=bias,
            )
        
        has_nans = torch.any(torch.isnan(result))
        
        assert not has_nans

        print("Result: ", result)
        print("+" * 10)
        return result 
    


                