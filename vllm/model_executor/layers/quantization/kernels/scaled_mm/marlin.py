# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import torch

from vllm.model_executor.layers.quantization.kernels.scaled_mm.ScaledMMLinearKernel import (  # noqa: E501
    FP8ScaledMMLinearLayerConfig,
    FP8W8A16LinearKernel,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    is_fp8_marlin_supported,
    prepare_fp8_layer_for_marlin,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform


class MarlinFP8ScaledMMLinearKernel(FP8W8A16LinearKernel):
    """
    FP8 Marlin kernel for GPUs that lack FP8 hardware support.
    Leverages the Marlin kernel for fast weight-only FP8 quantization.
    """

    @classmethod
    def get_min_capability(cls) -> int:
        return is_fp8_marlin_supported()

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        per_tensor_weight_scales = c.weight_quant_key.scale.group_shape.is_per_tensor()
        per_channel_weight_scales = (
            c.weight_quant_key.scale.group_shape.is_per_channel()
        )
        if not (per_tensor_weight_scales or per_channel_weight_scales):
            return False, "requires per tensor or per channel weight scales."
        return True, None

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "requires CUDA."
        # Check if platform supports FP8 Marlin
        if not is_fp8_marlin_supported():
            return False, "FP8 Marlin requires compute capability 8.0 or higher"

        return True, None

    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        super().__init__(c, layer_param_names)

        self.per_tensor_weight_scales = (
            c.weight_quant_key.scale.group_shape.is_per_tensor()
        )
        self.per_channel_weight_scales = (
            c.weight_quant_key.scale.group_shape.is_per_channel()
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w_q, _, _, _ = self._get_layer_params(layer)
        weight = w_q.t()
        w_q_name, _, _, _ = self.layer_param_names
        replace_parameter(layer, w_q_name, weight.data)
        prepare_fp8_layer_for_marlin(layer)
        del layer.input_scale

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return apply_fp8_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )
