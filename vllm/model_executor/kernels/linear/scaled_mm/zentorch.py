# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Zentorch dynamic-symmetric W8A8 int8 linear kernel for AMD Zen CPUs.

Selected by ``choose_scaled_mm_linear_kernel`` ahead of the generic
oneDNN-backed ``CPUInt8ScaledMMLinearKernel``. When ``is_supported`` or
``can_implement`` rejects a layer, the selector falls through to the next
kernel in ``_POSSIBLE_INT8_KERNELS[PlatformEnum.CPU]``.
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.zentorch import has_zentorch_op
from vllm.platforms import current_platform

from .ScaledMMLinearKernel import (
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)

logger = init_logger(__name__)


class ZentorchInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cpu():
            return False, "requires CPU."
        if not current_platform.is_zen_cpu():
            return False, "requires AMD Zen CPU."
        if not has_zentorch_op("zentorch_dynamic_qlinear"):
            return (
                False,
                "torch.ops.zentorch.zentorch_dynamic_qlinear is not registered.",
            )
        return True, None

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if c.is_static_input_scheme:
            return False, "requires dynamic activation quantization."
        if not c.input_symmetric:
            return False, "requires symmetric activation quantization."
        if not c.is_channelwise:
            return False, "requires per-channel weight quantization."
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Prepare weights for ``zentorch_dynamic_qlinear``.

        Keeps weight in [N, K] layout (int8, contiguous) and converts the
        per-channel weight scale to bf16 with shape ``(N,)``.
        """
        w_q_name, w_s_name, _, _, _ = self.layer_param_names
        weight = getattr(layer, w_q_name)
        n = weight.shape[0]
        replace_parameter(
            layer,
            w_q_name,
            torch.nn.Parameter(weight.data.contiguous(), requires_grad=False),
        )

        weight_scale = getattr(layer, w_s_name)
        ws = weight_scale.data
        if ws.dim() == 2 and ws.shape[-1] == 1:
            ws = ws.squeeze(-1)
        ws = ws.to(torch.bfloat16).contiguous()
        assert ws.shape == (n,), (
            f"[zen_cpu] expected weight scale shape ({n},), got {tuple(ws.shape)}"
        )

        replace_parameter(
            layer,
            w_s_name,
            torch.nn.Parameter(ws, requires_grad=False),
        )
        logger.info_once(
            "[zen_cpu] Using zentorch_dynamic_qlinear for W8A8 (dynamic-symmetric)"
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_q_name, w_s_name, _, _, _ = self.layer_param_names
        return torch.ops.zentorch.zentorch_dynamic_qlinear(
            x,
            getattr(layer, w_q_name),
            getattr(layer, w_s_name),
            bias,
            zentorch_op_name="zentorch::zentorch_dynamic_qlinear",
        )
