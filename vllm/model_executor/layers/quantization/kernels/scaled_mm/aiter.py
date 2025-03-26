# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

from .cutlass import CutlassScaledMMLinearKernel
from .ScaledMMLinearKernel import ScaledMMLinearLayerConfig


class AiterScaledMMLinearKernel(CutlassScaledMMLinearKernel):

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @classmethod
    def can_implement(
            cls, c: ScaledMMLinearLayerConfig) -> Tuple[bool, Optional[str]]:
        if current_platform.is_cpu():
            return (
                False,
                "AiterScaledMMLinearKernel requires `aiter` which is not " +
                "currently supported on CPU.")
        if not current_platform.is_rocm():
            return (
                False,
                "AiterScaledMMLinearKernel requires `aiter` which is only " +
                "currently supported on ROCm.")
        # try import aiter
        try:
            pass
        except Exception:
            return (
                False,
                "AiterScaledMMLinearKernel requires `aiter` which is not " +
                "installed supported on ROCm.")
        if not ops.is_rocm_aiter_gemm_w8a8_scaled_mm_enabled():
            return (False, "AiterScaledMMLinearKernel is disabled. " +
                    "Enable by setting `VLLM_ROCM_USE_AITER=1`.")

        if not c.input_symmetric:
            return (False,
                    "AiterScaledMMLinearKernel only supports symmetric " +
                    "quantization.")
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        return super().apply_weights(layer, x, bias)
