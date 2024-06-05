from typing import Callable, List, Optional

import torch
from torch.nn import Parameter

from vllm._C import ops
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.utils import set_weight_attrs

__all__ = ["CompressedTensors24"]


class CompressedTensors24(CompressedTensorsScheme):
    def __init__(self, strategy: str, num_bits: int, group_size: Optional[int] = None):
        self.strategy = strategy
        self.group_size = group_size
        self.num_bits = num_bits

        if self.strategy == "group" and self.group_size is None:
            raise ValueError(
                "group_size must be given when using strategy group")


    def create_weights(self, layer: torch.nn.Module, input_size: int,
                       output_partition_sizes: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):

        pack_factor = 8
        output_size_per_partition = sum(output_partition_sizes)

        qweight = Parameter(
            torch.empty(
                output_size_per_partition * 16 // pack_factor,
                input_size_per_partition // 16 // 2,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "packed_dim": 0,
                "pack_factor": pack_factor,
                "marlin_tile_size": 16,
            },
        )

        layer.register_parameter("weight_packed", qweight)
        set_weight_attrs(qweight,  {"weight_loader": weight_loader})

        input_groups = (1 if self.group_size is None else
                        input_size_per_partition //
                        self.group_size)

        scales = Parameter(
            torch.empty(
                output_size_per_partition,
                input_groups,
                device="cuda",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            scales,
            {
                "output_dim": 0,
                "input_dim": None if input_groups == 1 else 1,
            },
        )
        layer.register_parameter("scale_packed", scales)
        set_weight_attrs(scales, {"weight_loader": weight_loader})
        

        weight_shape = Parameter(torch.empty(2,
                                             device="cuda",
                                             dtype=torch.int64),
                                 requires_grad=False)

        layer.register_parameter("weight_shape", weight_shape)
        set_weight_attrs(weight_shape, {"weight_loader": weight_loader})

        meta = Parameter(
            torch.empty(
                output_size_per_partition * 2,
                input_size_per_partition // 8 // 2 // 2,
                device="cuda",
                dtype=torch.int16,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            meta,
            {
                "input_dim": 1,
                "packed_dim": 1,
                "pack_factor": 1,
                "output_dim": 0,
                "marlin_tile_size": 2,
            },
        )
        layer.register_parameter("meta", meta)
        set_weight_attrs(meta, {"weight_loader": weight_loader})

        max_workspace_size = (output_size_per_partition // 128) * 64
        workspace = Parameter(torch.zeros(max_workspace_size,
                                          device="cuda",
                                          dtype=torch.int),
                              requires_grad=False)
        layer.workspace = workspace

    
    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor):
        qweight = layer.weight_packed.t().contiguous()
        meta = layer.meta.t().contiguous()
        scales = layer.scale_packed.t().contiguous()
        workspace = layer.workspace

        x_2d = x.view(-1, x.shape[-1])

        size_m = x_2d.shape[0]
        size_k = x_2d.shape[1]
        size_n = scales.shape[1]

        output_2d = ops.gptq_marlin_24_gemm(x_2d, qweight, meta, scales,
                                            workspace,
                                            self.num_bits,
                                            size_m, size_n, size_k)

        output = output_2d.view(x.shape[:-1] + (output_2d.shape[1], ))
        return output
