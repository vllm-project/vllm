# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import torch

from vllm import envs
import vllm.model_executor.kernels.linear.base.w16a16 as w16a16
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

_SUPPORTED_DTYPES = (torch.bfloat16, torch.int8)


class Kernel(w16a16.Kernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None,
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cpu():
            return False, "CPU platform not available"
        if not envs.VLLM_CPU_SGL_KERNEL:
            return False, "VLLM_CPU_SGL_KERNEL is not enabled"
        if not torch.cpu._is_amx_tile_supported():
            return False, "AMX tile instructions not supported"
        return True, None

    @classmethod
    def can_implement(cls, config: w16a16.Config) -> tuple[bool, str | None]:
        if config.is_weight_meta:
            return False, "weight is meta"
        n, k = config.weight_shape
        if config.weight_dtype not in _SUPPORTED_DTYPES:
            return False, f"dtype {config.weight_dtype} not supported, expected bf16 or int8"
        if k % 32 != 0:
            return False, f"K={k} must be divisible by 32"
        if n % 16 != 0:
            return False, f"N={n} must be divisible by 16"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        params = self._get_layer_params(layer)
        processed = torch.ops._C.convert_weight_packed(params.processed_weight)
        if layer.bias is not None:
            replace_parameter(layer, "bias", layer.bias.to(torch.float32))
        replace_parameter(layer, w16a16.Params.PROCESSED_WEIGHT, processed)

    @staticmethod
    def apply(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        return torch.ops._C.weight_packed_linear(x, weight, bias, True)
