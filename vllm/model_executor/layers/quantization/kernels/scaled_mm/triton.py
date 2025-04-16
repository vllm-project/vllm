# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

from vllm.platforms import current_platform

from .cutlass import CutlassScaledMMLinearKernel
from .ScaledMMLinearKernel import ScaledMMLinearLayerConfig


class TritonScaledMMLinearKernel(CutlassScaledMMLinearKernel):

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def can_implement(
            cls, c: ScaledMMLinearLayerConfig) -> Tuple[bool, Optional[str]]:
        if current_platform.is_cpu():
            return (
                False,
                "TritonScaledMMLinearKernel requires Triton which is not " +
                "currently supported on CPU.")
        if not c.input_symmetric:
            return (False,
                    "TritonScaledMMLinearKernel only supports symmetric " +
                    "quantization.")
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        return super().apply_weights(layer, x, bias)
