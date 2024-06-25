from typing import Callable, List

import torch
from torch.nn import Parameter

from vllm import _custom_ops as custom_ops
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8 import (  # noqa: E501
    CompressedTensorsW8A8)
from vllm.model_executor.utils import set_weight_attrs

__all__ = ["CompressedTensorsW8A8StaticTensor"]


class CompressedTensorsW8A8StaticTensor(CompressedTensorsW8A8):

    def create_weights(self, layer: torch.nn.Module,
                       output_partition_sizes: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):

        super().create_weights(
            layer=layer,
            output_partition_sizes=output_partition_sizes,
            input_size_per_partition=input_size_per_partition,
            params_dtype=params_dtype,
            weight_loader=weight_loader)

        input_scale = Parameter(torch.empty(1, dtype=torch.float32),
                                requires_grad=False)

        layer.register_parameter("input_scale", input_scale)
        set_weight_attrs(input_scale, {
            "weight_loader": weight_loader,
            "ignore_warning": True,
        })

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor):
        weight = layer.weight
        weight_scale = layer.weight_scale
        act_scale = layer.input_scale

        # Input quantize
        x_q, _ = custom_ops.scaled_int8_quant(x, act_scale)

        return custom_ops.cutlass_scaled_mm(x_q, weight.t(), act_scale,
                                            weight_scale, x.dtype)
