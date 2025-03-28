# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
    triton_scaled_mm)
from vllm.platforms import current_platform

from .cutlass import CutlassScaledMMLinearKernel
from .ScaledMMLinearKernel import ScaledMMLinearLayerConfig


class TritonScaledMMLinearKernel(CutlassScaledMMLinearKernel):

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def is_supported(
        cls,
        compute_capability: Optional[int] = None
    ) -> Tuple[bool, Optional[str]]:
        if current_platform.is_rocm() or current_platform.is_cuda():
            return cls._current_capability_supported(compute_capability)

        return False, "Triton scaled_mm requires running on ROCm or CUDA."

    @classmethod
    def can_implement(
            cls, c: ScaledMMLinearLayerConfig) -> Tuple[bool, Optional[str]]:
        if not c.input_symmetric:
            return (False,
                    "TritonScaledMMLinearKernel only supports symmetric " +
                    "quantization.")
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # TODO maybe this doesn't need to transpose the weight?
        # Could also skip asymmetric-only paths
        super().process_weights_after_loading(layer)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        w_q, w_s, i_s, _, _ = self._get_weight_params(layer)

        # ops.scaled_int8_quant supports both dynamic and static quant:
        # * dynamic, i_s is None and x_s computed from x.
        # * static, i_s is scalar and x_s is i_s.

        # Only symmetric supported in triton_scaled_mm
        x_q, x_s, _ = ops.scaled_int8_quant(x, i_s, symmetric=True)

        return triton_scaled_mm(x_q,
                                w_q,
                                scale_a=x_s,
                                scale_b=w_s,
                                out_dtype=x.dtype,
                                bias=bias)
