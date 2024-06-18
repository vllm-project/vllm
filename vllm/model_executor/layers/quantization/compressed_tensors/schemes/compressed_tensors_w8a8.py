from typing import Callable, List, Tuple, Union

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    QuantizationStrategy)
from vllm.model_executor.utils import set_weight_attrs


class CompressedTensorsW8A8(CompressedTensorsScheme):

    def __init__(self, strategy: str):
        self.strategy = strategy

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

        is_tensor_partitioned = len(output_partition_sizes) != 1
        weight_scale_dim = sum(output_partition_sizes) if (
            is_tensor_partitioned
            or self.strategy == QuantizationStrategy.CHANNEL) else 1

        shape: Union[Tuple[int], Tuple[int, int]] = (weight_scale_dim, )
        if self.strategy == QuantizationStrategy.CHANNEL:
            shape = (weight_scale_dim, 1)

        weight_scale = Parameter(torch.empty(*shape, dtype=torch.float32),
                                 requires_grad=False)

        layer.register_parameter("weight_scale", weight_scale)
        set_weight_attrs(weight_scale, {"weight_loader": weight_loader})

        weight = Parameter(torch.empty(sum(output_partition_sizes),
                                       input_size_per_partition,
                                       dtype=torch.int8),
                           requires_grad=False)

        layer.register_parameter("weight", weight)
        set_weight_attrs(
            weight, {
                "input_dim": 1,
                "output_dim": 0,
                "weight_loader": weight_loader,
                "logical_widths": output_partition_sizes
            })

        # Don't need a shard_splitter for channel-wise quantization
        # Use the default loading method
        if self.strategy == QuantizationStrategy.CHANNEL:
            set_weight_attrs(weight_scale, {
                "output_dim": 0,
            })
        else:
            set_weight_attrs(
                weight_scale, {
                    "logical_widths": output_partition_sizes,
                    "shard_splitter": self.scales_shard_splitter,
                })
