# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from functools import lru_cache

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)


@lru_cache(maxsize=1)
def _load_rdna4_flydsl_mm():
    """Load the vLLM-owned FlyDSL kernel lazily."""
    from .flydsl_kernels.rdna4_fp8_blockscale import (
        rdna4_fp8_block_scaled_mm,
    )

    return rdna4_fp8_block_scaled_mm


def _rdna4_fp8_block_scaled_mm_impl(
    a: torch.Tensor,
    weight: torch.Tensor,
    a_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    return _load_rdna4_flydsl_mm()(a, weight, a_scale, weight_scale)


def _rdna4_fp8_block_scaled_mm_fake(
    a: torch.Tensor,
    weight: torch.Tensor,
    a_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    del a_scale, weight_scale
    return torch.empty(
        (a.size(0), weight.size(0)), dtype=torch.bfloat16, device=a.device
    )


direct_register_custom_op(
    op_name="rdna4_fp8_block_scaled_mm",
    op_func=_rdna4_fp8_block_scaled_mm_impl,
    mutates_args=[],
    fake_impl=_rdna4_fp8_block_scaled_mm_fake,
)


class RDNA4Fp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    """FlyDSL block-FP8 kernel family for RDNA4."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        del compute_capability
        if not current_platform.is_rocm():
            return False, "RDNA4 block-FP8 kernels require ROCm"
        from vllm.platforms.rocm import on_rdna4

        if not on_rdna4():
            return False, "RDNA4 block-FP8 kernels require gfx1200 or gfx1201"
        try:
            _load_rdna4_flydsl_mm()
        except (ImportError, OSError) as exc:
            return False, f"patched RDNA4 FlyDSL stack is unavailable: {exc}"
        return True, None

    @classmethod
    def can_implement(
        cls, config: FP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        can_implement, reason = super().can_implement(config)
        if not can_implement:
            return can_implement, reason
        if config.input_dtype != torch.bfloat16:
            return False, "RDNA4 block-FP8 supports only BF16 input"
        if config.out_dtype != torch.bfloat16:
            return False, "RDNA4 block-FP8 supports only BF16 output"
        if config.activation_quant_key.dtype != torch.float8_e4m3fn:
            return False, "RDNA4 block-FP8 requires OCP FP8 activations"
        if config.weight_quant_key.dtype != torch.float8_e4m3fn:
            return False, "RDNA4 block-FP8 requires OCP FP8 weights"
        if config.activation_quant_key.scale.dtype != torch.float32:
            return False, "RDNA4 block-FP8 requires FP32 activation scales"
        if config.weight_quant_key.scale.dtype != torch.float32:
            return False, "RDNA4 block-FP8 requires FP32 weight scales"
        if config.activation_quant_key.scale.group_shape != GroupShape(1, 128):
            return False, "RDNA4 block-FP8 requires 1x128 activation scales"
        if config.weight_quant_key.scale.group_shape != GroupShape(128, 128):
            return False, "RDNA4 block-FP8 requires 128x128 weight scales"
        n, k = config.weight_shape
        if n <= 0 or n % 128 != 0 or k <= 0 or k % 128 != 0:
            return False, "RDNA4 block-FP8 requires N and K divisible by 128"
        return True, None

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.vllm.rdna4_fp8_block_scaled_mm(A, B, As, Bs)
