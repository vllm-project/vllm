# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import Optional

import torch

from vllm import _custom_ops as ops
from vllm import envs
import vllm.model_executor.kernels.linear.base.w16a16 as w16a16
from vllm.platforms import current_platform
from vllm.utils.platform_utils import num_compute_units

_SKINNY_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


def _fits_llmm1(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> bool:
    n = x.numel() // x.size(-1)
    return n == 1 and bias is None


class LLMM1Kernel(w16a16.Kernel):
    """Skinny GEMM for n=1, no bias. gfx9/gfx1x."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None,
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "ROCm platform not available"
        if not envs.VLLM_ROCM_USE_SKINNY_GEMM:
            return False, "VLLM_ROCM_USE_SKINNY_GEMM is not enabled"
        from vllm.platforms.rocm import on_gfx1x, on_gfx9
        if not (on_gfx9() or on_gfx1x()):
            return False, "gfx9 or gfx1x not available"
        return True, None

    @classmethod
    def can_implement(cls, config: w16a16.Config) -> tuple[bool, str | None]:
        if config.is_weight_meta:
            return False, "weight is meta"
        if config.weight_dtype not in _SKINNY_SUPPORTED_DTYPES:
            return False, f"dtype {config.weight_dtype} not supported"
        m, k = config.weight_shape
        if m % 4 != 0:
            return False, f"M={m} must be divisible by 4"
        if k % 8 != 0:
            return False, f"K={k} must be divisible by 8"
        if k > 8192:
            return False, f"K={k} must be <= 8192"
        return True, None

    @staticmethod
    def apply(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x_view = x.reshape(-1, x.size(-1))
        out = ops.LLMM1(weight, x_view, 4)
        return out.reshape(*x.shape[:-1], weight.shape[0])


def _fits_wvsplitk(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> bool:
    n = x.numel() // x.size(-1)
    return 0 < n <= 4


class WvSplitKKernel(w16a16.Kernel):
    """Skinny GEMM for very small batch sizes (n <= 4). gfx9/gfx1x."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None,
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "ROCm platform not available"
        if not envs.VLLM_ROCM_USE_SKINNY_GEMM:
            return False, "VLLM_ROCM_USE_SKINNY_GEMM is not enabled"
        from vllm.platforms.rocm import on_gfx1x, on_gfx9
        if not (on_gfx9() or on_gfx1x()):
            return False, "gfx9 or gfx1x not available"
        return True, None

    @classmethod
    def can_implement(cls, config: w16a16.Config) -> tuple[bool, str | None]:
        if config.is_weight_meta:
            return False, "weight is meta"
        if config.weight_dtype not in _SKINNY_SUPPORTED_DTYPES:
            return False, f"dtype {config.weight_dtype} not supported"
        m, k = config.weight_shape
        if k % 8 != 0:
            return False, f"K={k} must be divisible by 8"
        if m <= 8:
            return False, f"M={m} must be > 8"
        return True, None

    @staticmethod
    def apply(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x_view = x.reshape(-1, x.size(-1))
        out = ops.wvSplitK(weight, x_view, num_compute_units(), bias)
        return out.reshape(*x.shape[:-1], weight.shape[0])


def _fits_wvsplitkrc(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> bool:
    n = x.numel() // x.size(-1)
    m, k = weight.shape
    N_p2 = 1 << (n - 1).bit_length()
    rndup_cus = ((m + 64 - 1) // 64) * ((k + 512 - 1) // 512)
    grps_shr_b = min(N_p2 // 16, 4)
    cu_needed = rndup_cus * grps_shr_b
    fits = (N_p2 * m * ((k + 512 - 1) // 512)) <= 128 * 1024 * 12
    fits &= cu_needed <= num_compute_units()
    return 10 <= n <= 128 and fits


class WvSplitKrcKernel(w16a16.Kernel):
    """Skinny GEMM with atomic reduce-counting splitK. gfx950 only."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None,
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "ROCm platform not available"
        if not envs.VLLM_ROCM_USE_SKINNY_GEMM:
            return False, "VLLM_ROCM_USE_SKINNY_GEMM is not enabled"
        from vllm.platforms.rocm import on_gfx950
        if not on_gfx950():
            return False, "gfx950 not available"
        return True, None

    @classmethod
    def can_implement(cls, config: w16a16.Config) -> tuple[bool, str | None]:
        if config.is_weight_meta:
            return False, "weight is meta"
        if config.weight_dtype not in _SKINNY_SUPPORTED_DTYPES:
            return False, f"dtype {config.weight_dtype} not supported"
        m, k = config.weight_shape
        if m % 16 != 0:
            return False, f"M={m} must be divisible by 16"
        if k % 8 != 0:
            return False, f"K={k} must be divisible by 8"
        if k <= 512:
            return False, f"K={k} must be > 512"
        if not config.weight_contiguous:
            return False, "weight is not contiguous"
        return True, None

    @staticmethod
    def apply(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return ops.wvSplitKrc(x, weight, num_compute_units(), bias)
