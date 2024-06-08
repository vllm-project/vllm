from typing import Callable, List, Tuple, Union

import torch
from torch.nn import Parameter

from vllm import _custom_ops as custom_ops
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.utils import set_weight_attrs

__all__ = ["CompressedTensorsW8A8DynamicToken"]


class CompressedTensorsW8A8DynamicToken(CompressedTensorsScheme):

    def _shard_id_as_int(self, shard_id: Union[str, int]) -> int:
        if isinstance(shard_id, int):
            return shard_id

        assert isinstance(shard_id, str)
        qkv_idxs = {"q": 0, "k": 1, "v": 2}
        assert shard_id in qkv_idxs
        return qkv_idxs[shard_id]

    def scales_shard_splitter(
            self, param: torch.Tensor, loaded_weight: torch.Tensor,
            shard_id: Union[str, int],
            logical_widths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shard_id = self._shard_id_as_int(shard_id)
        offset = sum(logical_widths[:shard_id])
        size = logical_widths[shard_id]
        # update loaded weight with copies for broadcast.
        loaded_weight = loaded_weight.repeat(size)
        return param[offset:offset + size], loaded_weight

    def create_weights(self, layer: torch.nn.Module,
                       output_partition_sizes: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):

        # When the scales have a single value, it is required that they be
        # on the CPU for performance and CUDA Graphs compatibility. Please
        # refer to the comment in
        # CompressedTensorsW8A8StaticTensor::create_weights for further
        # information.
        is_tensor_partitioned = len(output_partition_sizes) != 1
        weight_scale_dim = sum(
            output_partition_sizes) if is_tensor_partitioned else 1

        weight_zero_point = Parameter(torch.empty(1, dtype=torch.int8),
                                      requires_grad=False)

        weight_scale = Parameter(torch.empty(weight_scale_dim,
                                             dtype=torch.float32),
                                 requires_grad=False)

        weight = Parameter(torch.empty(sum(output_partition_sizes),
                                       input_size_per_partition,
                                       dtype=torch.int8),
                           requires_grad=False)

        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight, {"weight_loader": weight_loader})
        set_weight_attrs(weight, {"logical_widths": output_partition_sizes})

        layer.register_parameter("weight_scale", weight_scale)
        set_weight_attrs(weight_scale, {"weight_loader": weight_loader})
        set_weight_attrs(
            weight_scale, {
                "shard_splitter": self.scales_shard_splitter,
                "logical_widths": output_partition_sizes
            })

        layer.register_parameter("weight_zero_point", weight_zero_point)
        set_weight_attrs(weight_zero_point, {"weight_loader": weight_loader})

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor):
        weight = layer.weight
        weight_scale = layer.weight_scale

        x_q, input_scales = custom_ops.scaled_int8_quant(x)
        return custom_ops.cutlass_scaled_mm_dq(x_q, weight.t(), input_scales,
                                               weight_scale, x.dtype)
