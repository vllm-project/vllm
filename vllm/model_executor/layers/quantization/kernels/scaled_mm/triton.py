# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch

# import this to register custom op
import vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm  # noqa: F401
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.platforms import current_platform

from .cutlass import CutlassScaledMMLinearKernel
from .ScaledMMLinearKernel import ScaledMMLinearLayerConfig

logger = init_logger(__name__)


class TritonScaledMMLinearKernel(CutlassScaledMMLinearKernel):

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def can_implement(
            cls, c: ScaledMMLinearLayerConfig) -> tuple[bool, Optional[str]]:
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
        w_q, w_s, i_s, i_zp, _ = self._get_weight_params(layer)

        # ops.scaled_int8_quant supports both dynamic and static quant:
        # * dynamic, i_s is None and x_s computed from x.
        # * static, i_s is scalar and x_s is i_s.
        x_q, x_s, _ = ops.scaled_int8_quant(x.contiguous(),
                                            i_s,
                                            i_zp,
                                            symmetric=True)

        return torch.ops.vllm.triton_scaled_mm(x_q, w_q, x_s, w_s, x.dtype,
                                               bias)
