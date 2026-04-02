# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    Mxfp8LinearOp,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
)

__all__ = ["CompressedTensorsW8A8Mxfp8"]


class CompressedTensorsW8A8Mxfp8(CompressedTensorsScheme):
    """
    Compressed tensors scheme for MXFP8 quantization (W8A8).

    Loads pre-quantized MXFP8 weights from compressed-tensors checkpoints.
    Activations are dynamically quantized to MXFP8 at runtime.

    MXFP8 format:
    - 8-bit float weights (E4M3) stored as float8_e4m3fn
    - Per-group E8M0 scales (uint8) with group_size=32
    - Activations dynamically quantized to MXFP8 during inference
    """

    def __init__(self):
        self.group_size = MXFP8_BLOCK_SIZE
        self.mxfp8_linear = Mxfp8LinearOp()

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.params_dtype = params_dtype

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.mxfp8_linear.process_weights(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.mxfp8_linear.apply(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            out_dtype=layer.params_dtype,
            bias=bias,
            workspace=getattr(layer, "workspace", None),
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
        )
