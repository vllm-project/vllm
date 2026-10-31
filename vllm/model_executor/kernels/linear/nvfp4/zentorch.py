# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 linear backed by zentorch's W4A8 custom kernel on AMD Zen CPUs."""

import torch

from vllm.model_executor.kernels.linear.zentorch_utils import (
    check_w4_group_shape,
    has_zentorch_op,
    w4_group_linear,
    w4_group_scale_from_e4m3,
    warn_if_zentorch_predates_e2m1,
)
from vllm.model_executor.utils import replace_parameter

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig


class ZentorchNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 linear served by zentorch's W4A8 kernel, weights left four-bit.

    Only the two scale levels are folded into the per-group floats the kernel
    takes; the E2M1 codes are passed through packed and are never requantized
    or expanded.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not has_zentorch_op(["zentorch_dynamic_qlinear"]):
            return False, "requires a Zen CPU with zentorch loaded"
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        # NvFp4LinearLayerConfig carries no shape, so the gates are checked
        # against the real weights in process_weights_after_loading.
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        warn_if_zentorch_predates_e2m1()
        block_scale = layer.weight_scale.data
        n, packed_k = layer.weight.shape
        check_w4_group_shape(n, packed_k * 2, block_scale.shape[-1])

        replace_parameter(
            layer,
            "weight_scale",
            w4_group_scale_from_e4m3(block_scale, layer.weight_global_scale.data),
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return w4_group_linear(layer.weight, layer.weight_scale, x, bias)
