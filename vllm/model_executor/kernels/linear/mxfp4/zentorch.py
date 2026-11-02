# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 linear backed by zentorch's W4A8 custom kernel on AMD Zen CPUs.

The CPU had no MXFP4 linear kernel at all, so a dense MXFP4 checkpoint used to
fail kernel selection outright; only the MoE experts path was served.
"""

import torch

from vllm.model_executor.kernels.linear.zentorch_utils import (
    check_w4_group_shape,
    has_zentorch_op,
    w4_group_linear,
    w4_group_scale_from_e8m0,
    warn_if_zentorch_predates_e2m1,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic
from vllm.model_executor.utils import replace_parameter

from .base import MxFp4LinearKernel, MxFp4LinearLayerConfig


class ZentorchMxFp4LinearKernel(MxFp4LinearKernel):
    """MXFP4 linear served by zentorch's W4A8 kernel, weights left four-bit.

    Shares everything but the scale conversion with the NVFP4 kernel: E2M1
    codes go through packed, and only the E8M0 exponents are folded into the
    per-group floats the kernel takes.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not has_zentorch_op(["zentorch_dynamic_qlinear"]):
            return False, "requires a Zen CPU with zentorch loaded"
        return True, None

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        # The kernel quantizes bf16 activations to int8 itself, so a W4A4
        # checkpoint's activation scales go unused rather than unsupported.
        if config.activation_quant_key not in (None, kMxfp4Dynamic):
            return False, "only MXFP4 or unquantized activations"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        warn_if_zentorch_predates_e2m1()
        e8m0 = layer.weight_scale.data
        n, packed_k = layer.weight.shape
        check_w4_group_shape(n, packed_k * 2, e8m0.shape[-1])

        replace_parameter(layer, "weight_scale", w4_group_scale_from_e8m0(e8m0))

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return w4_group_linear(layer.weight, layer.weight_scale, x, bias)
