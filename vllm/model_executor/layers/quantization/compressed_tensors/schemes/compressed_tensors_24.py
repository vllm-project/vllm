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
from torch.sparse import to_sparse_semi_structured
from vllm.model_executor.layers.quantization.utils.marlin_utils_test_24 import sparse_semi_structured_to_dense_cutlass, sparse_semi_structured_from_dense_cutlass
__all__ = ["CompressedTensors24"]

class CompressedTensors24(CompressedTensorsScheme):
    def __init__(self, model_compressor: Optional[ModelCompressor] = None, layer_name = None):
        self.model_compressor = model_compressor
        self.layer_name = layer_name

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    def create_weights(self, layer: torch.nn.Module, input_size: int,
                    output_partition_sizes: List[int],
                    input_size_per_partition: int,
                    params_dtype: torch.dtype, weight_loader: Callable,
                    **kwargs):
        
        compressed = True  # toggle based on the case we're running
        weights_dtype = params_dtype
        weights = ModelWeightParameter(data=torch.empty(
            sum(output_partition_sizes),
            input_size_per_partition // 2,
            dtype=weights_dtype),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader)

        # parameter to store uncompressed weight or decompressed weight
        weight_unpacked = ModelWeightParameter(data=torch.empty(
            sum(output_partition_sizes),
            input_size_per_partition,
            dtype=weights_dtype),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader)
        
        # For the uncompressed case
        if compressed:
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

        layer.register_parameter("weight", weight_unpacked)
        

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
            weight_packed_data = layer.weight_packed.data
            meta = layer.meta.data

            qkv_sizes = [2048, 256, 256]
            gate_up_sizes = [5632, 5632]
            split_weights = None 
            split_meta = None

            def _process_split(input_weight, input_meta):
                weight_data = {
                    "weight_packed": input_weight,
                    "meta": input_meta
                }
                decompress = self.model_compressor.sparsity_compressor.decompress_weight(weight_data)
                return decompress

            print(self.layer_name)
            if "qkv" in self.layer_name:
                split_weights = torch.split(weight_packed_data, qkv_sizes)
                split_meta = torch.split(meta, qkv_sizes)
            elif "gate_up" in self.layer_name:
                split_weights = torch.split(weight_packed_data, gate_up_sizes)
                split_meta = torch.split(meta, gate_up_sizes)
            
            if split_weights:
                all_compress = []
                for i in range(len(split_weights)):
                    print(split_weights[i].shape, split_meta[i].shape)
                    compress_i = _process_split(split_weights[i], split_meta[i])
                    all_compress.append(compress_i)
                
                compressed = torch.cat(all_compress)
                compressed = compress_to_torch_sparse_semi_structured_mat(compressed)
            else:
                decompress = _process_split(weight_packed_data, meta)
                compressed = compress_to_torch_sparse_semi_structured_mat(decompress)
            
            layer.weight = Parameter(compressed, requires_grad=False)
            
        else:
            # assume uncompressd case
            # Proof that Alex's methods work: we can compress and decompress to get accurate generation using his methods below
            # Would be equivalent to uncommenting out the next two lines and passing decompress into compress_to_torch_sparse_semi_structured_mat which also works
            #comp, meta = sparse_semi_structured_from_dense_cutlass(layer.weight)
            #decompress = sparse_semi_structured_to_dense_cutlass(comp, meta)
            compressed = compress_to_torch_sparse_semi_structured_mat(layer.weight)
            layer.weight = Parameter(compressed, requires_grad=False)

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

        """ debugging code
        a_sparse = to_sparse_semi_structured(layer.weight)
        result = torch.mm(a_sparse, x.t().contiguous())
        return result.t().contiguous()
        """
        return semi_structured_dense_sparse_T_gemm(
            a_dense=x, 
            b_T_packed=layer.weight.data
        )


                