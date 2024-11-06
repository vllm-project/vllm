from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.parameter import ModelWeightParameter, PerTensorScaleParameter
import torch
from typing import List, Callable, Optional
from compressed_tensors.compressors import ModelCompressor
from torch.nn import Parameter
from vllm.model_executor.layers.sparsity.utils.cusparse_2_4_utils import (
    compress_to_torch_sparse_semi_structured_mat,
    semi_structured_dense_sparse_T_gemm,
    semi_structured_sparse_dense_gemm_scaled
    )
from torch.sparse import to_sparse_semi_structured
from vllm.model_executor.layers.quantization.utils.marlin_utils_test_24 import sparse_semi_structured_to_dense_cutlass, sparse_semi_structured_from_dense_cutlass
from vllm import _custom_ops as ops
from typing import Tuple

__all__ = ["CompressedTensors24"]

class CompressedTensors24(CompressedTensorsScheme):
    def __init__(self, model_compressor: Optional[ModelCompressor] = None, layer_name = None):
        self.model_compressor = model_compressor
        self.layer_name = layer_name
        self.quantized = True  # toggle based on the case we're running
        self.compressed = False  # toggle based on the case we're running



    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    def create_weights(self, layer: torch.nn.Module, input_size: int,
                    output_partition_sizes: List[int],
                    input_size_per_partition: int,
                    params_dtype: torch.dtype, weight_loader: Callable,
                    **kwargs):
        layer.logical_widths = output_partition_sizes
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
        
        if self.quantized:

            # assume per tensor static quantization
            weight_scale = PerTensorScaleParameter(data=torch.empty(
                    len(output_partition_sizes), dtype=torch.float),
                                                    weight_loader=weight_loader)

            weight_zero_point = PerTensorScaleParameter(data=torch.empty(
                len(output_partition_sizes), dtype=torch.float8_e4m3fn),
                                                weight_loader=weight_loader)
            
            
            input_scale = PerTensorScaleParameter(data=torch.empty(
                    len(output_partition_sizes), dtype=torch.float),
                                                    weight_loader=weight_loader)
            
            input_zero_point = PerTensorScaleParameter(data=torch.empty(
                len(output_partition_sizes), dtype=torch.float8_e4m3fn),
                                                weight_loader=weight_loader)


            layer.register_parameter("weight_scale", weight_scale)
            layer.register_parameter("input_scale", input_scale)
            layer.register_parameter("input_zero_point", input_zero_point)
            layer.register_parameter("weight_zero_point", weight_zero_point)
    
        if self.compressed:
            # store compression specific things to be used
            # later during decompression

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
        if self.compressed and hasattr(layer, "weight_packed"):
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
            # uncompressed case
            # quantize the weights to fp8 and store them

            dq_weight = layer.weight.data
            weight_scale = layer.weight_scale.data

            if len(weight_scale) != 1:
                # needed for cases where modules are merged
                # to reduce the number of scales to one
                scale, q_weight = quantize_with_max_scale(
                    dq_weight, weight_scale, layer.logical_widths
                )
            else:
                # if modules are not merged, we can directly
                # use the scale provided, and quantize the weights
                q_weight, scale = ops.scaled_fp8_quant(dq_weight, weight_scale)

            layer.weight_scale = Parameter(scale, requires_grad=False)

            # Temporary check to ensure that the weights are 2:4 sparse
            assert check_24(q_weight), "Not 2:4 sparse"
            
            compressed = compress_to_torch_sparse_semi_structured_mat(q_weight)
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

        if not self.quantized:
            return semi_structured_dense_sparse_T_gemm(
                a_dense=x, 
                b_T_packed=layer.weight.data
            )
        
        input_scale = layer.input_scale.data
        weight_scale = layer.weight_scale.data
        weight = layer.weight.data

        # Quantize the input tensor to fp8
        # can use the max scale for the input tensor
        # as the merged modules have a same scale
        # repeated for all the partitions
        input_scale = input_scale.max()
        q_input, input_scale = ops.scaled_fp8_quant(x, input_scale)
        
        if q_input.is_contiguous():
            # Make q_input non-contiguous
            # as expected by the kernel
            q_input = q_input.t().contiguous().t()


        assert not q_input.is_contiguous(), "Input is contiguous, the Kernel expects non-contiguous input"
        output =  semi_structured_sparse_dense_gemm_scaled(
            a_packed=weight,
            b_dense=q_input,
            scale_a=weight_scale,
            scale_b=input_scale,
            bias=bias
        )
        output = output.t().to(x.dtype)
        print()
        print(f"{self.layer_name} executed")
        print("\t", "Input shape:", x.shape, "weight shape:", weight.shape, "output shape:", output.shape)
        return output


def quantize_with_max_scale(
        weight: torch.Tensor, weight_scale: torch.Tensor,
        logical_widths: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    # Max scale to be used for quanitzation.
    max_w_scale = weight_scale.max()

    # QKV / MLP is fused in the on disk checkpoint if any of the
    # weight scales are still set to the default since we initialize
    # N weight scales for N shards but we only load 1 weight scale
    # from disk in this case. Skip requantization in this case (since)
    # we already are quantized with the single scale.
    # * Sample Model: nm-testing/Phi-3-mini-128k-instruct-FP8
    unfused_module_in_checkpoint = (weight_scale[-1] > torch.finfo(
        torch.float8_e4m3fn).min)
    q_weight = torch.empty_like(weight).to(torch.float8_e4m3fn)
    # If unfused checkpoint, need quantize with the single scale.
    if unfused_module_in_checkpoint:
        start = 0
        for idx, logical_width in enumerate(logical_widths):
            end = start + logical_width
            q_weight[start:end, :], _ = ops.scaled_fp8_quant(
                weight[start:end, :], max_w_scale)
            start = end    
    return max_w_scale, q_weight

def check_24(tensor):
    new_tensor = tensor.view(-1, 4)    
    zero_counts = (new_tensor == 0).sum(dim=1)
    return (zero_counts >= 2).all().item()

