# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import Optional

import torch

import vllm.model_executor.kernels.linear.base.w16a16 as w16a16
from vllm.platforms import current_platform

_AITER_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

_AITER_SUPPORTED_SHAPES = frozenset([
    (5120, 2880),
    (2880, 4096),
    (128, 2880),
    (640, 2880),
    (2880, 512),
])


def _fits_aiter_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> bool:
    n = x.numel() // x.size(-1)
    m = weight.shape[0]
    return not (n > 2048 and m > 512)


class Kernel(w16a16.Kernel):
    """Triton GEMM via aiter for specific weight shapes."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None,
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "ROCm platform not available"
        if current_platform.is_fp8_fnuz():
            return False, "fp8 fnuz not supported"
        try:
            from vllm._aiter_ops import rocm_aiter_ops
        except ImportError:
            return False, "aiter not available"
        if not rocm_aiter_ops.is_triton_gemm_enabled():
            return False, "aiter triton gemm not enabled"
        return True, None

    @classmethod
    def can_implement(cls, config: w16a16.Config) -> tuple[bool, str | None]:
        if config.is_weight_meta:
            return False, "weight is meta"
        if config.weight_dtype not in _AITER_SUPPORTED_DTYPES:
            return False, f"dtype {config.weight_dtype} not supported"
        if config.weight_shape not in _AITER_SUPPORTED_SHAPES:
            return False, f"shape {config.weight_shape} not in supported aiter shapes"
        return True, None

    @staticmethod
    def apply(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        from aiter.ops.triton.gemm_a16w16 import gemm_a16w16
        return gemm_a16w16(x, weight, bias)
