from typing import Callable, List, Optional

import torch
from torch.nn import Parameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.gptq_marlin_24 import (
    GPTQ_MARLIN_24_MAX_PARALLEL, GPTQ_MARLIN_24_MIN_THREAD_N)
from vllm.model_executor.utils import set_weight_attrs

__all__ = ["CompressedTensorsW4A16Sparse24"]
W4A16SPARSE24_SUPPORTED_BITS = [4]


class CompressedTensorsW4A16Sparse24(CompressedTensorsScheme):

    def __init__(self,
                 strategy: str,
                 num_bits: int,
                 group_size: Optional[int] = None):
        self.strategy = strategy
        self.group_size = group_size
        self.num_bits = num_bits
        self.tile_size = 16

        if self.strategy == "group" and self.group_size is None:
            raise ValueError(
                "group_size must be given when using strategy group")

    def create_weights(self, layer: torch.nn.Module, input_size: int,
                       output_partition_sizes: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):

        pack_factor = 32 // self.num_bits
        output_size_per_partition = sum(output_partition_sizes)

        qweight = Parameter(
            torch.empty(
                input_size_per_partition // self.tile_size // 2,
                output_size_per_partition * self.tile_size // pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight,
            {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": pack_factor,
                "marlin_tile_size": self.tile_size,
                "weight_loader": weight_loader
            },
        )

        layer.register_parameter("weight_packed", qweight)

        input_groups = (1 if self.group_size is None else
                        input_size_per_partition // self.group_size)

        scales = Parameter(
            torch.empty(
                input_groups,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            scales,
            {
                "output_dim": 1,
                "input_dim": None if input_groups == 1 else 0,
                "weight_loader": weight_loader
            },
        )
        layer.register_parameter("scale_packed", scales)

        weight_shape = Parameter(torch.empty(2, dtype=torch.int64),
                                 requires_grad=False)

        layer.register_parameter("weight_shape", weight_shape)
        set_weight_attrs(weight_shape, {"weight_loader": weight_loader})

        meta = Parameter(
            torch.empty(
                input_size_per_partition // 8 // 2 // 2,
                output_size_per_partition * 2,
                dtype=torch.int16,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            meta,
            {
                "input_dim": 0,
                "packed_dim": 1,
                "pack_factor": 1,
                "output_dim": 1,
                "marlin_tile_size": 2,
                "weight_loader": weight_loader
            },
        )
        layer.register_parameter("meta", meta)

        max_workspace_size = (
            output_size_per_partition //
            GPTQ_MARLIN_24_MIN_THREAD_N) * GPTQ_MARLIN_24_MAX_PARALLEL
        workspace = Parameter(torch.zeros(max_workspace_size, dtype=torch.int),
                              requires_grad=False)
        layer.workspace = workspace

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor):
        qweight = layer.weight_packed
        meta = layer.meta
        scales = layer.scale_packed
        workspace = layer.workspace

        x_2d = x.view(-1, x.shape[-1])

        size_m = x_2d.shape[0]
        size_k = x_2d.shape[1]
        size_n = scales.shape[1]

        output_2d = ops.gptq_marlin_24_gemm(x_2d, qweight, meta, scales,
                                            workspace, self.num_bits, size_m,
                                            size_n, size_k)

        output = output_2d.view(x.shape[:-1] + (output_2d.shape[1], ))
        return output
