# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import torch

from types import SimpleNamespace

from vllm import envs
import vllm.model_executor.kernels.linear.base.w16a16 as w16a16
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform


class Kernel(w16a16.Kernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None,
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cpu():
            return False, "CPU platform not available"
        if not current_platform.is_zen_cpu():
            return False, "not a Zen CPU"
        if not hasattr(torch.ops.zentorch, "zentorch_linear_unary"):
            return False, "zentorch_linear_unary not available"
        return True, None

    @classmethod
    def can_implement(cls, config: w16a16.Config) -> tuple[bool, str | None]:
        if config.is_weight_meta:
            return False, "weight is meta"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        params = self._get_layer_params(layer)
        processed = params.processed_weight.detach()
        is_prepacked = False

        if envs.VLLM_ZENTORCH_WEIGHT_PREPACK and hasattr(
            torch.ops.zentorch, "zentorch_weight_prepack_for_linear"
        ):
            processed = torch.ops.zentorch.zentorch_weight_prepack_for_linear(
                processed
            )
            is_prepacked = True

        replace_parameter(layer, w16a16.Params.PROCESSED_WEIGHT, processed)
        layer.extras = SimpleNamespace(is_prepacked=is_prepacked)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        params = self._get_layer_params(layer)
        return torch.ops.zentorch.zentorch_linear_unary(
            x, params.processed_weight, bias,
            is_weight_prepacked=params.extra_kwargs.is_prepacked,
        )
