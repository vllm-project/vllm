from typing import Callable, List, Tuple, Union

import torch
from torch.nn import Parameter

from vllm import _custom_ops as custom_ops
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.utils import set_weight_attrs

__all__ = ["CompressedTensorsW8A8StaticTensor"]


class CompressedTensorsW8A8StaticTensor(CompressedTensorsScheme):

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

        # Note on input/weight scales and zero_points
        #
        # When the scales have a single value, it is required that they be
        # on the CPU for 2 reasons,
        # 1. Performance:
        #   When the scales (input_scale/weight_scales) have only a single
        #   value, we perform a scalar broadcast of that value during the
        #   quant/dequant operations. The "quant" and the "gemm+dequant"
        #   kernels accept the Scalar by-value. These tensors are allocated
        #   on the CPU in order to avoid the GPU-to-CPU copy when passing
        #   by-value.
        #
        # 2. CUDA Graphs:
        #   CUDA Graphs don't support GPU-to-CPU copy operations during
        #   stream capture.
        #
        # TODO: zero-points are not supported yet. But we expect a similar
        # pattern.

        is_tensor_partitioned = len(output_partition_sizes) != 1
        weight_scale_dim = sum(
            output_partition_sizes) if is_tensor_partitioned else 1
        weight_scale_device = "cpu" if weight_scale_dim == 1 else "cuda"

        input_scale = Parameter(torch.empty(1,
                                            device="cpu",
                                            dtype=torch.float32),
                                requires_grad=False)
        input_zero_point = Parameter(torch.empty(1,
                                                 device="cpu",
                                                 dtype=torch.int8),
                                     requires_grad=False)

        weight_scale = Parameter(torch.empty(weight_scale_dim,
                                             device=weight_scale_device,
                                             dtype=torch.float32),
                                 requires_grad=False)
        weight_zero_point = Parameter(torch.empty(1,
                                                  device="cpu",
                                                  dtype=torch.int8),
                                      requires_grad=False)

        weight = Parameter(torch.empty(sum(output_partition_sizes),
                                       input_size_per_partition,
                                       dtype=torch.int8),
                           requires_grad=False)

        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})

        set_weight_attrs(weight, {"weight_loader": weight_loader})

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

        # Input quantize
        x_q = custom_ops.static_scaled_int8_quant(x, act_scale[0].item())

        return custom_ops.cutlass_scaled_mm_dq(x_q, weight.t(), act_scale,
                                               weight_scale, x.dtype)
