from typing import Callable, List, Optional

import torch
from compressed_tensors.quantization import (QuantizationArgs,
                                             QuantizationStrategy,
                                             QuantizationType)

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise, sparse_cutlass_supported)
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           ChannelQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)

__all__ = ["CompressedTensors24"]


class CompressedTensors24(CompressedTensorsScheme):

    def __init__(self,
                 quantized: bool = False,
                 weight_quant: Optional[QuantizationArgs] = None,
                 input_quant: Optional[QuantizationArgs] = None):

        self.quantized = quantized
        self.weight_quant = weight_quant
        self.input_quant = input_quant

    @classmethod
    def get_min_capability(cls) -> int:
        # Only cutlass 3.x kernels are implemented so far
        return 90

    def create_weights(self, layer: torch.nn.Module, input_size: int,
                       output_partition_sizes: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):

        if not sparse_cutlass_supported():
            raise ValueError(
                "Sparse CUTLASS not supported. vLLM must be built with "
                "CUDA 12.2 or later to use this feature")

        self.output_dtype = params_dtype
        layer.logical_widths = output_partition_sizes
        self.weights_dtype: torch.dtype = self._get_params_dtype(params_dtype)

        # parameter to store uncompressed weight
        weight = ModelWeightParameter(data=torch.empty(
            sum(output_partition_sizes),
            input_size_per_partition,
            dtype=self.weights_dtype),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)

        # Check if quantized, not just 2:4 Sparse
        if self.quantized:
            if (self.weight_quant and self.weight_quant.strategy
                    == QuantizationStrategy.CHANNEL.value):
                weight_scale = ChannelQuantScaleParameter(
                    data=torch.empty((sum(output_partition_sizes), 1),
                                     dtype=torch.float32),
                    output_dim=0,
                    weight_loader=weight_loader)
            else:
                assert (self.weight_quant and self.weight_quant.strategy
                        == QuantizationStrategy.TENSOR.value)
                weight_scale = PerTensorScaleParameter(
                    data=torch.empty(len(output_partition_sizes),
                                     dtype=torch.float32),
                    weight_loader=weight_loader)

            layer.register_parameter("weight_scale", weight_scale)

            # input quant will be non-none
            if self.input_quant and not self.input_quant.dynamic:
                # register input quant scale
                assert (self.input_quant.strategy ==
                        QuantizationStrategy.TENSOR.value)
                input_scale = BasevLLMParameter(data=torch.empty(
                    1, dtype=torch.float32),
                                                weight_loader=weight_loader)

                layer.register_parameter("input_scale", input_scale)

        else:
            # for sparse-only, pass in 1 for weight/input scales
            weight_scale = torch.nn.Parameter(data=torch.ones(
                1, dtype=torch.float32),
                                              requires_grad=False)
            input_scale = torch.nn.Parameter(data=torch.ones(
                1, dtype=torch.float32),
                                             requires_grad=False)
            layer.register_parameter("input_scale", input_scale)
            layer.register_parameter("weight_scale", weight_scale)

        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Compress weights after loading. Store compressed weight and meta
            tensor
        
        :post-condition: layer.w_compressed and layer.meta are
            set to the compressed weight and meta tensor in the
            format expected by the Cutlass kernels
        :param layer: The layer with the weights to be processed
        
        """
        # torch.compile workaround
        if hasattr(layer, "input_scale"):
            layer.input_scale = torch.nn.Parameter(layer.input_scale.data,
                                                   requires_grad=False)

        if self.weight_quant:
            if self.weight_quant.strategy == QuantizationStrategy.TENSOR.value:
                layer.weight_scale = torch.nn.Parameter(convert_to_channelwise(
                    weight_scale=layer.weight_scale,
                    logical_widths=layer.logical_widths),
                                                        requires_grad=False)
            else:
                # torch.compile workaround
                layer.weight_scale = torch.nn.Parameter(
                    layer.weight_scale.data, requires_grad=False)

        w_compressed, meta = ops.cutlass_sparse_compress(layer.weight.data)
        layer.weight = torch.nn.Parameter(w_compressed, requires_grad=False)
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
        if self.quantized:
            scale = None
            if hasattr(layer, "input_scale"):
                scale = layer.input_scale

            if self.weights_dtype == torch.int8:
                ops_output = ops.scaled_int8_quant(x, scale=scale)
                q_input = ops_output[0]
                input_scale = ops_output[1]
            else:
                assert self.weights_dtype == torch.float8_e4m3fn
                if scale is not None:
                    q_input, input_scale = ops.scaled_fp8_quant(x, scale=scale)
                else:
                    q_input, input_scale = ops.scaled_fp8_quant(
                        x, use_per_token_if_dynamic=True)

        else:
            # Not quantized, nothing to do with the input_scales, use as is
            input_scale = layer.input_scale
            q_input = x

        out = ops.cutlass_scaled_sparse_mm(a=q_input,
                                           bt_nzs=layer.weight,
                                           bt_meta=layer.meta,
                                           scale_a=input_scale,
                                           scale_b=layer.weight_scale,
                                           out_dtype=self.output_dtype,
                                           bias=bias)
        assert out.is_contiguous()
        return out

    def _get_params_dtype(self, params_dtype: torch.dtype) -> torch.dtype:
        if not self.quantized:
            return params_dtype

        assert self.weight_quant is not None
        assert self.input_quant is not None

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
