# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.model_executor.layers.quantization.kernels.wFP8a16.WFP8A16_kernel import (
    FP8WoQLinearKernel,
    FP8WoQLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    is_fp8_marlin_supported,
    prepare_fp8_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise,
)
from vllm.platforms import current_platform


class FP8MarlinLinearKernel(FP8WoQLinearKernel):
    """
    FP8 Marlin kernel for GPUs that lack FP8 hardware support.
    Leverages the Marlin kernel for fast weight-only FP8 quantization.
    """

    @classmethod
    def get_min_capability(cls) -> int:
        return is_fp8_marlin_supported()

    @classmethod
    def can_implement(cls, c: FP8WoQLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "requires CUDA."
        # Check if platform supports FP8 Marlin
        if not is_fp8_marlin_supported():
            return False, "FP8 Marlin requires compute capability 8.0 or higher"
        per_tensor_weight_scales = c.weight_quant_key.scale.group_shape.is_per_tensor()
        per_channel_weight_scales = (
            c.weight_quant_key.scale.group_shape.is_per_channel()
        )
        if not (per_tensor_weight_scales or per_channel_weight_scales):
            return False, "requires per tensor or per channel weight scales."
        return True, None

    def __init__(
        self,
        c: FP8WoQLinearLayerConfig,
    ) -> None:
        super().__init__(c)

        self.per_tensor_weight_scales = (
            c.weight_quant_key.scale.group_shape.is_per_tensor()
        )
        self.per_channel_weight_scales = (
            c.weight_quant_key.scale.group_shape.is_per_channel()
        )
        self.input_dtype = c.input_dtype
        self.is_block_quant = c.is_block_quant
        self.size_k_first = not self.is_block_quant

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.per_tensor_weight_scales:
            ws_channelwise = convert_to_channelwise(
                layer.weight_scale, layer.logical_widths
            )
            layer.weight_scale = torch.nn.Parameter(ws_channelwise, requires_grad=False)
        elif self.per_channel_weight_scales:
            # required by torch.compile to be torch.nn.Parameter
            layer.weight_scale = torch.nn.Parameter(
                layer.weight_scale.data, requires_grad=False
            )
        # Weights must be transposed for marlin
        layer.weight = torch.nn.Parameter(layer.weight.t(), requires_grad=False)

        prepare_fp8_layer_for_marlin(
            layer, size_k_first=self.size_k_first, input_dtype=self.input_dtype
        )

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
            input_dtype=self.input_dtype,
            bias=bias,
        )
