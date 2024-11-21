from typing import Any, Dict, List, Callable, Optional
import torch

from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.quantization import QuantizationType, QuantizationStrategy
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.parameter import ModelWeightParameter, ChannelQuantScaleParameter, PerTensorScaleParameter
from vllm import _custom_ops as ops

__all__ = ["CompressedTensors24"]

class CompressedTensors24(CompressedTensorsScheme):
    def __init__(
            self, 
            layer_name: Optional[str] = None,
            quantized: bool = False,
            do_decompress: bool = False,
            weight_quant = None,
            input_quant = None,
            config: Optional[Dict[str, Any]] = None,
            ):
        self.layer_name = layer_name
        self.quantized = quantized
        self.do_decompress = do_decompress
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        self.model_compressor = (
            ModelCompressor.from_compression_config(compression_config=config)
            if self.do_decompress and config is not None
            else None
            )


    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    def create_weights(self, layer: torch.nn.Module, input_size: int,
                    output_partition_sizes: List[int],
                    input_size_per_partition: int,
                    params_dtype: torch.dtype, weight_loader: Callable,
                    **kwargs):
        layer.logical_widths = output_partition_sizes
        self.output_dtype=params_dtype

        weights_dtype: torch.dtype = self._get_params_dtype(params_dtype)

        # parameter to store uncompressed weight or decompressed weight
        weight = ModelWeightParameter(
            data=torch.empty(sum(output_partition_sizes),
                             input_size_per_partition,
                             dtype=weights_dtype),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader)
        
        if self.do_decompress:
            # store compression specific things to be used
            # later during decompression

            # compressed weight for 2:4 sparse (compressed-tensors)
            weight_packed = ModelWeightParameter(data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // 2,
                dtype=weights_dtype),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader
                )
            
            bits_per_weight_element = weight.itemsize * 8 
            meta_dtype = torch.int32 if bits_per_weight_element == 8 else torch.int16
            meta_input_size = (
                input_size_per_partition // 32
                if bits_per_weight_element == 8
                else input_size_per_partition // 16
            )

            # meta tensor for 2:4 decompression
            meta = ModelWeightParameter(data=torch.empty(
                sum(output_partition_sizes), 
                meta_input_size,
                dtype=meta_dtype),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader)

            layer.register_parameter("weight_packed", weight_packed)
            layer.register_parameter("meta", meta)
        
        if self.quantized:

            if self.weight_quant.strategy == QuantizationStrategy.CHANNEL.value:
                weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1),
                                dtype=torch.float32),
                                output_dim=0,
                                weight_loader=weight_loader)
            else:
                weight_scale = PerTensorScaleParameter(data=torch.empty(
                    len(output_partition_sizes), dtype=torch.float32),
                                                    weight_loader=weight_loader)
                # check if this is needed
                weight_zero_point = PerTensorScaleParameter(data=torch.empty(
                len(output_partition_sizes), dtype=weights_dtype),
                                                weight_loader=weight_loader)
                layer.register_parameter("weight_zero_point", weight_zero_point)
                
                
            layer.register_parameter("weight_scale", weight_scale)
            
            # input quant will be non-none
            if not self.input_quant.dynamic:
                # register input quant scale
                if self.input_quant.strategy == QuantizationStrategy.CHANNEL.value:
                    pass
                else:
                    input_scale = PerTensorScaleParameter(data=torch.empty(
                    len(output_partition_sizes), dtype=torch.float32),
                                                    weight_loader=weight_loader)
                    # Can we ignore this?
                    input_zero_point = PerTensorScaleParameter(data=torch.empty(
                    len(output_partition_sizes), dtype=weights_dtype),
                                                weight_loader=weight_loader)
                    layer.register_parameter("input_zero_point", input_zero_point)

                
                layer.register_parameter("input_scale", input_scale)

        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Apply any transformations to the weights after loading
        them from disk
        
        :post-condition: layer.w_compressed and layer.meta are
            set to the compressed weight and meta tensor in the
            format expected by the Cutlass kernels
        :param layer: The layer with the weights to be processed
        
        """
        weight_to_compress = (
            layer.weight if not self.do_decompress
            else self._decompress_24_weight(layer.weight_packed.data, layer.meta.data)
        )
        w_compressed, meta = ops.cutlass_compress_entry(weight_to_compress)
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
        if hasattr(layer, "input_scale"):
            q_input, input_scale = ops.scaled_fp8_quant(
                x, scale=layer.input_scale)
        else:
            q_input, input_scale = ops.scaled_fp8_quant(
            x, use_per_token_if_dynamic=True)

        out = ops.cutlass_scaled_sparse_mm(
            a=layer.w_compressed,
            e=layer.meta,
            b=q_input.t(),
            scale_a=layer.weight_scale,
            scale_b=input_scale,
            out_dtype=self.output_dtype,
            bias=bias
        )

        out = out.t()
        assert out.is_contiguous()
        return out
    
    def _decompress_24_weight(self, weight_packed: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
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
            split_weights = torch.split(weight_packed, qkv_sizes)
            split_meta = torch.split(meta, qkv_sizes)
        elif "gate_up" in self.layer_name:
            split_weights = torch.split(weight_packed, gate_up_sizes)
            split_meta = torch.split(meta, gate_up_sizes)

        if split_weights:
            all_compress = []
            for i in range(len(split_weights)):
                print(split_weights[i].shape, split_meta[i].shape)
                compress_i = _process_split(split_weights[i], split_meta[i])
                all_compress.append(compress_i)

            decompressed = torch.cat(all_compress)
        else:
            decompressed = _process_split(weight_packed, meta)

        return decompressed
    
    def _get_params_dtype(self, params_dtype: torch.dtype) -> torch.dtype:
        if not self.quantized:
            return params_dtype
        
        is_8_bits = self.weight_quant.num_bits == self.input_quant.num_bits == 8

        if not is_8_bits:
            raise ValueError("Cutlass only supports 8-bit quantization")
        
        if (self.weight_quant.type == QuantizationType.FLOAT
            and self.input_quant.type == QuantizationType.FLOAT):
            return torch.float8_e4m3fn
        
        if (self.weight_quant.type == QuantizationType.INT
            and self.input_quant.type == QuantizationType.INT):
            return torch.int8
        
        raise ValueError("Quantization type not supported by Cutlass")



def check_24(tensor):
    new_tensor = tensor.view(-1, 4)    
    zero_counts = (new_tensor == 0).sum(dim=1)
    return (zero_counts >= 2).all().item()

