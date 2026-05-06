# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import torch

from types import SimpleNamespace

from vllm import _custom_ops as ops
import vllm.model_executor.kernels.linear.base.w16a16 as w16a16
from vllm.platforms import CpuArchEnum, current_platform


class Kernel(w16a16.Kernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None,
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cpu():
            return False, "CPU platform not available"
        if not ops._supports_onednn:
            return False, "oneDNN is not available"
        if current_platform.get_cpu_architecture() == CpuArchEnum.POWERPC:
            return False, "oneDNN is not supported on PowerPC"
        return True, None

    @classmethod
    def can_implement(cls, config: w16a16.Config) -> tuple[bool, str | None]:
        if config.is_weight_meta:
            return False, "weight is meta"
        try:
            m, k = config.weight_shape
            probe = torch.empty(m, k, dtype=config.weight_dtype)
            ops.create_onednn_mm(probe.t(), 32)
        except RuntimeError as e:
            return False, f"oneDNN handler creation failed: {e}"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        params = self._get_layer_params(layer)
        handler = ops.create_onednn_mm(params.processed_weight.t(), 32)
        layer.extras = SimpleNamespace(handler=handler)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        params = self._get_layer_params(layer)
        return ops.onednn_mm(params.extra_kwargs.handler, x, bias)
