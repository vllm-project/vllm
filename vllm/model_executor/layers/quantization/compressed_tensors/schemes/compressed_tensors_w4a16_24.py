# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
from compressed_tensors.quantization import QuantizationArgs
from torch.nn import Parameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.gptq_marlin_24 import (
    GPTQ_MARLIN_24_MAX_PARALLEL,
    GPTQ_MARLIN_24_MIN_THREAD_N,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from vllm.scalar_type import scalar_types

__all__ = ["CompressedTensorsW4A16Sparse24"]
W4A16SPARSE24_SUPPORTED_TYPES_MAP = {
    4: scalar_types.uint4b8,
}
W4A16SPARSE24_SUPPORTED_BITS = list(W4A16SPARSE24_SUPPORTED_TYPES_MAP.keys())


class CompressedTensorsW4A16Sparse24(CompressedTensorsScheme):
    def __init__(self, strategy: str, num_bits: int, group_size: int | None = None):
        self.strategy = strategy
        self.group_size = group_size
        self.tile_size = 16

        if num_bits not in W4A16SPARSE24_SUPPORTED_TYPES_MAP:
            raise ValueError(
                f"Unsupported num_bits = {num_bits}. "
                f"Supported num_bits = {W4A16SPARSE24_SUPPORTED_BITS}"
            )

        self.quant_type = W4A16SPARSE24_SUPPORTED_TYPES_MAP[num_bits]

        if self.strategy == "group" and self.group_size is None:
            raise ValueError("group_size must be given when using strategy group")

    @classmethod
    def get_min_capability(cls, weight_quant: QuantizationArgs) -> int:
        # ampere + up
        return 80

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # required by torch.compile to be torch.nn.Parameter
        layer.weight_packed = Parameter(layer.weight_packed.data, requires_grad=False)
        layer.scale_packed = Parameter(layer.scale_packed.data, requires_grad=False)
        layer.meta = Parameter(layer.meta.data, requires_grad=False)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size: int,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        assert params_dtype == torch.float16, (
            "float16 is required for marlin24 compressed models. Set dtype=torch.float16"  # noqa: E501
        )

        pack_factor = 32 // self.quant_type.size_bits
        output_size_per_partition = sum(output_partition_sizes)

        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // self.tile_size // 2,
                output_size_per_partition * self.tile_size // pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=pack_factor,
            marlin_tile_size=self.tile_size,
            weight_loader=weight_loader,
        )

        input_groups = (
            1
            if self.group_size is None
            else input_size_per_partition // self.group_size
        )

        weight_scale_args = {
            "data": torch.empty(
                input_groups,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            "weight_loader": weight_loader,
        }

        if self.group_size is not None:
            scales = GroupQuantScaleParameter(
                output_dim=1, input_dim=0, **weight_scale_args
            )
        else:
            scales = ChannelQuantScaleParameter(output_dim=1, **weight_scale_args)

        weight_shape = BasevLLMParameter(
            data=torch.empty(2, dtype=torch.int64), weight_loader=weight_loader
        )

        meta = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // 8 // 2 // 2,
                output_size_per_partition * 2,
                dtype=torch.int16,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=1,
            marlin_tile_size=2,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight_packed", qweight)
        layer.register_parameter("weight_shape", weight_shape)
        layer.register_parameter("scale_packed", scales)
        layer.register_parameter("meta", meta)

        max_workspace_size = (
            output_size_per_partition // GPTQ_MARLIN_24_MIN_THREAD_N
        ) * GPTQ_MARLIN_24_MAX_PARALLEL

        workspace = Parameter(
            torch.zeros(max_workspace_size, dtype=torch.int), requires_grad=False
        )
        layer.workspace = workspace

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        qweight = layer.weight_packed
        meta = layer.meta
        scales = layer.scale_packed
        workspace = layer.workspace

        x_2d = x.view(-1, x.shape[-1])

        size_m = x_2d.shape[0]
        size_k = x_2d.shape[1]
        size_n = scales.shape[1]

        output_2d = ops.gptq_marlin_24_gemm(
            x_2d,
            qweight,
            meta,
            scales,
            workspace,
            self.quant_type,
            size_m,
            size_n,
            size_k,
        )

        output = output_2d.view(x.shape[:-1] + (output_2d.shape[1],))

        if bias is not None:
            output.add_(bias)  # In-place add

        return output
