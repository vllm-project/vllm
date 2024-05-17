from typing import Callable, List, Tuple, Union

import torch
from torch.nn import Parameter

# TODO (varun) : Unify ops and custom ops
from vllm._C import ops
from vllm import _custom_ops as custom_ops

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.utils import set_weight_attrs

__all__ = ["CompressedTensorsW8A8StaticTensor"]


class CompressedTensorsW8A8StaticTensor(CompressedTensorsScheme):

    def __init__(self, fake_quant):
        self.fake_quant = fake_quant

    def _quantize(self,
                  x: torch.Tensor,
                  scales: torch.Tensor,
                  logical_widths: List[int],
                  split_dim: int = 0) -> torch.Tensor:

        x_q = torch.empty_like(x, dtype=torch.int8, device="cuda")
        x_q_split = x_q.split(logical_widths, dim=split_dim)
        x_split = x.split(logical_widths, dim=split_dim)

        for q, dq, scale in zip(x_q_split, x_split, scales):
            ops.static_scaled_int8_quant(q, dq, scale.item())

        return x_q

    def _quantize_single(self, x: torch.Tensor, scale: float):
        x_q = torch.empty_like(x, dtype=torch.int8, device="cuda")
        ops.static_scaled_int8_quant(x_q, x, scale)
        return x_q

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

        # TODO: remove zero_point parameters once the configs given remove them
        is_tensor_partitioned = len(output_partition_sizes) != 1
        dim = sum(output_partition_sizes) if is_tensor_partitioned else 1

        input_scale = Parameter(torch.empty(1,
                                            device="cuda",
                                            dtype=torch.float32),
                                requires_grad=False)
        input_zero_point = Parameter(torch.empty(1,
                                                 device="cuda",
                                                 dtype=torch.int8),
                                     requires_grad=False)

        weight_scale = Parameter(torch.empty(dim,
                                             device="cuda",
                                             dtype=torch.float32),
                                 requires_grad=False)
        weight_zero_point = Parameter(torch.empty(1,
                                                  device="cuda",
                                                  dtype=torch.int8),
                                      requires_grad=False)

        if not self.fake_quant:
            params_dtype = torch.int8
        weight = Parameter(torch.empty(sum(output_partition_sizes),
                                       input_size_per_partition,
                                       device="cuda",
                                       dtype=params_dtype),
                           requires_grad=False)

        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})

        set_weight_attrs(weight, {"weight_loader": weight_loader})
        set_weight_attrs(weight, {"logical_widths": output_partition_sizes})

        layer.register_parameter("input_scale", input_scale)
        set_weight_attrs(input_scale, {"weight_loader": weight_loader})
        layer.register_parameter("input_zero_point", input_zero_point)
        set_weight_attrs(input_zero_point, {"weight_loader": weight_loader})
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
        act_scale = layer.input_scale
        logical_widths = weight.logical_widths

        # Input quantize
        x_q = self._quantize_single(x, act_scale[0].item())

        # Weight quantize
        # TODO : try not to remove device-to-host copy.
        # i.e. keep the non-duplicated version of scales in the CPU
        if self.fake_quant:  # TODO: update
            w_scales = [
                weight_scale[sum(logical_widths[:i])].item()
                for i in range(len(logical_widths))
            ]
            w_scales = torch.FloatTensor(w_scales, device=torch.device("cpu"))
            w_q = self._quantize(weight, w_scales, logical_widths)
        else:
            w_q = weight  # already quantized

        return custom_ops.cutlass_scaled_mm_dq(x_q, w_q.t(), act_scale, weight_scale, x.dtype)
