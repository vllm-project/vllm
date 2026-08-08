# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    is_fp4_marlin_supported,
    prepare_fp4_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    cutlass_fp4_supported,
)

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

logger = init_logger(__name__)


class MarlinNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 weight-only GEMM via Marlin (W4A16)."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if is_fp4_marlin_supported():
            return True, None
        return False, "Marlin FP4 not available"

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Reachable on FP4-native GPUs too: weight-only (W4A16_NVFP4)
        # checkpoints have no activation scales, so Marlin is their only
        # valid kernel regardless of hardware.
        if cutlass_fp4_supported():
            logger.warning_once(
                "FP4 weights will be dequantized to the activation dtype "
                "inside the Marlin kernel instead of using this GPU's "
                "native FP4 tensor cores (weight-only FP4 checkpoints such "
                "as W4A16_NVFP4 only support this path). This may degrade "
                "performance for compute-heavy workloads."
            )
        else:
            logger.warning_once(
                "Your GPU does not have native support for FP4 computation "
                "but FP4 quantization is being used. Weight-only FP4 "
                "compression will be used leveraging the Marlin kernel. "
                "This may degrade performance for compute-heavy workloads."
            )
        prepare_fp4_layer_for_marlin(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )
