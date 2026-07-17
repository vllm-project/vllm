# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Static

from .base import MxFp4LinearKernel, MxFp4LinearLayerConfig


class MarlinMxFp4LinearKernel(MxFp4LinearKernel):
    """Weight-only (A16) MXFP4 GEMM via Marlin. Does not quantize activations,
    so it can only be selected when the caller has no explicit activation
    quantization expectation."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            is_fp4_marlin_supported,
        )

        if is_fp4_marlin_supported():
            return True, None
        return False, "Marlin FP4 not available"

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        if config.weight_quant_key != kMxfp4Static:
            return False, "only supports MXFP4 weights"
        if config.activation_quant_key is not None:
            return False, "weight-only kernel, cannot quantize activations"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            prepare_fp4_layer_for_marlin,
        )

        prepare_fp4_layer_for_marlin(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            apply_fp4_marlin_linear,
        )

        return apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=None,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )
