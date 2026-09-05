# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Zentorch dynamic-symmetric W8A8 int8 linear kernel for AMD Zen CPUs.

Selected by ``choose_scaled_mm_linear_kernel`` ahead of the generic
oneDNN-backed ``CPUInt8ScaledMMLinearKernel``. When ``is_supported`` or
``can_implement`` rejects a layer, the selector falls through to the next
kernel in ``_POSSIBLE_INT8_KERNELS[PlatformEnum.CPU]``.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.zentorch_utils import (
    has_zentorch_op,
    has_zentorch_op_arg,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.platforms import current_platform

from .ScaledMMLinearKernel import (
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)

logger = init_logger(__name__)


def _weight_prepack_enabled() -> bool:
    """Return ``True`` when zentorch can consume a prepacked W8A8 weight.

    The prepacked layout is only valid for the blocked matmul algo; any other
    algo silently misreads it instead of failing.
    """
    if not has_zentorch_op(["zentorch_weight_prepack_for_dynamic_qlinear"]):
        return False
    if not has_zentorch_op_arg("zentorch_dynamic_qlinear", "is_weight_prepacked"):
        return False
    return os.environ.get("ZENDNNL_MATMUL_ALGO", "1") == "1"


class ZentorchInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cpu():
            return False, "requires CPU."
        if not current_platform.is_zen_cpu():
            return False, "requires AMD Zen CPU."
        if not has_zentorch_op(["zentorch_dynamic_qlinear"]):
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
        weight_data = weight.data.contiguous()

        prepacked = _weight_prepack_enabled()
        if prepacked:
            weight_data = (
                torch.ops.zentorch.zentorch_weight_prepack_for_dynamic_qlinear(
                    weight_data
                )
            )
        layer._zentorch_weight_prepacked = prepacked

        replace_parameter(
            layer,
            w_q_name,
            torch.nn.Parameter(weight_data, requires_grad=False),
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
            "[zen_cpu] Using zentorch_dynamic_qlinear for W8A8 "
            "(dynamic-symmetric, weight_prepacked=%s)",
            prepacked,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_q_name, w_s_name, _, _, _ = self.layer_param_names
        weight = getattr(layer, w_q_name)
        weight_scale = getattr(layer, w_s_name)
        if getattr(layer, "_zentorch_weight_prepacked", False):
            # Only a newer zentorch declares this argument.
            return torch.ops.zentorch.zentorch_dynamic_qlinear(
                x,
                weight,
                weight_scale,
                bias,
                is_weight_prepacked=True,
                zentorch_op_name="zentorch::zentorch_dynamic_qlinear",
            )
        return torch.ops.zentorch.zentorch_dynamic_qlinear(
            x,
            weight,
            weight_scale,
            bias,
            zentorch_op_name="zentorch::zentorch_dynamic_qlinear",
        )
