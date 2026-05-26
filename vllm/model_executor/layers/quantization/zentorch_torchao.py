# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform-specific TorchAO linear method overrides."""

from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.torchao import TorchAOLinearMethod
from vllm.model_executor.layers.quantization.utils.zentorch import has_zentorch_op
from vllm.platforms import current_platform

logger = init_logger(__name__)


def get_optimized_method(_method: TorchAOLinearMethod, config):
    """Return a platform-optimized TorchAO method wrapper, or ``None``."""
    if not current_platform.is_zen_cpu():
        return None
    return ZentorchTorchAOLinearMethod(config)


class ZentorchTorchAOLinearMethod(TorchAOLinearMethod):
    """Zen CPU fast-path for TorchAO DA8W8 Int8 dynamic quantization."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        self._try_setup_zentorch_da8w8(layer)

    def _try_setup_zentorch_da8w8(self, layer: torch.nn.Module) -> None:
        if not has_zentorch_op(["zentorch_dynamic_qlinear"]):
            return
        try:
            from torchao.quantization.granularity import PerRow
            from torchao.quantization.quantize_.workflows import Int8Tensor
        except ImportError:
            return

        w = layer.weight
        if not isinstance(w, Int8Tensor) or w.act_quant_kwargs is None:
            return
        if w.act_quant_kwargs.granularity != PerRow():
            logger.warning_once(
                "zentorch will treat PerTensor granularity of activation "
                "quantization as PerRow for DA8W8 fast path."
            )

        scales = w.scale
        n = w.qdata.shape[0]
        if scales.dim() == 2 and scales.shape == (n, 1):
            scales = scales.squeeze(-1)
        else:
            return

        layer._zentorch_dynamic_qlinear_weight = w.qdata
        layer._zentorch_dynamic_qlinear_scales = scales
        layer.register_parameter(
            "weight", torch.nn.Parameter(torch.empty(0), requires_grad=False)
        )
        layer._zentorch_freed_weight = True
        layer._zentorch_kind = "torchao_da8w8"

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hasattr(layer, "_zentorch_dynamic_qlinear_weight"):
            return torch.ops.zentorch.zentorch_dynamic_qlinear(
                x,
                layer._zentorch_dynamic_qlinear_weight,
                layer._zentorch_dynamic_qlinear_scales,
                bias,
                zentorch_op_name="zentorch::zentorch_dynamic_qlinear",
            )
        return super().apply(layer, x, bias)
