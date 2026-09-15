# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

from vllm.model_executor.kernels.linear import init_mxfp8_linear_kernel
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp8Dynamic,
    kMxfp8Static,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
)

from .quark_scheme import QuarkScheme

__all__ = ["QuarkW8A8Mxfp8"]


class QuarkW8A8Mxfp8(QuarkScheme):
    """MXFP8 (W8A8) for Quark checkpoints that spell the scale 2-D.

    Quark exports MXFP8 as ``fp8_e4m3`` / ``per_block`` with
    ``block_size=[R, 32]`` and an ``e8m0`` scale, i.e. one scale per RxR2
    block rather than the ``[out, in // 32]`` MXFP8 canonical layout. With
    ``R == 1`` the two coincide; DeepSeek-V4.1 uses ``R == 32``, so each
    checkpoint scale row covers 32 weight rows and is expanded on load.
    """

    supported_activation_quant_keys = [kMxfp8Dynamic]
    supported_weight_quant_keys = [kMxfp8Static]

    def __init__(
        self,
        weight_quant_key=kMxfp8Static,
        activation_quant_key=kMxfp8Dynamic,
        scale_block_rows: int = 1,
    ):
        super().__init__(weight_quant_key, activation_quant_key)
        self.scale_block_rows = scale_block_rows
        self.kernel = init_mxfp8_linear_kernel()

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    def _scale_weight_loader(self, weight_loader: Callable) -> Callable:
        if self.scale_block_rows == 1:
            return weight_loader

        rows = self.scale_block_rows

        def scaled_loader(param, loaded_weight, *args, **kwargs):
            loaded_weight = loaded_weight.view(torch.uint8).repeat_interleave(
                rows, dim=0
            )
            return weight_loader(param, loaded_weight, *args, **kwargs)

        return scaled_loader

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        if input_size_per_partition % MXFP8_BLOCK_SIZE != 0:
            raise ValueError(
                f"MXFP8 requires input size divisible by {MXFP8_BLOCK_SIZE}, "
                f"got {input_size_per_partition}"
            )
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.params_dtype = params_dtype

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=MXFP8_VALUE_DTYPE,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // MXFP8_BLOCK_SIZE,
                dtype=MXFP8_SCALE_DTYPE,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=self._scale_weight_loader(weight_loader),
        )
        layer.register_parameter("weight_scale", weight_scale)
        layer.weight_block_size = [1, MXFP8_BLOCK_SIZE]

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)
