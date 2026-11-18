# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform

from .aiter import AiterFp8BlockScaledMMKernel
from .ScaledMMLinearKernel import FP8ScaledMMLinearLayerConfig


class RocmGfx1151SkinnyFp8BlockScaledMMKernel(AiterFp8BlockScaledMMKernel):
    """gfx1151 native skinny FP8 blockscale GEMM with AITER fallback."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        supported, reason = super().is_supported(compute_capability)
        if not supported:
            return False, reason
        if not current_platform.is_rocm():
            return False, "requires ROCm."

        from vllm.platforms.rocm import on_gfx1151

        if not on_gfx1151() or not rocm_aiter_ops.is_rdna_linear_enabled():
            return False, "requires gfx1151 RDNA AITER blockscale."
        return True, None

    @classmethod
    def can_implement(
        cls, config: FP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        supported, reason = super().can_implement(config)
        if not supported:
            return False, reason
        if config.weight_shape not in ((1536, 4096), (4096, 4096)):
            return False, "requires a validated skinny projection shape."
        if config.out_dtype != torch.bfloat16:
            return False, "requires bfloat16 output."
        if config.weight_quant_key.scale.group_shape != GroupShape(128, 128):
            return False, "requires 128x128 weight blocks."
        return True, None

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        n, k = self.config.weight_shape
        if (
            self.use_triton
            and not self.use_bpreshuffle
            and tuple(self.weight_group_shape) == (128, 128)
            and self.config.out_dtype == torch.bfloat16
            and (n, k) in ((1536, 4096), (4096, 4096))
            and A.ndim == B.ndim == As.ndim == Bs.ndim == 2
            and 1 <= A.shape[0] <= 6
            and A.shape[1] == k
            and B.shape == (n, k)
            and As.shape == (A.shape[0], k // 128)
            and Bs.shape == (n // 128, k // 128)
            and A.dtype == B.dtype == torch.float8_e4m3fn
            and As.dtype == Bs.dtype == torch.float32
            and A.is_contiguous()
            and B.stride(1) == 1
            and B.stride(0) >= k
            and B.stride(0) % 16 == 0
            and As.is_contiguous()
            and Bs.is_contiguous()
            and A.device == B.device == As.device == Bs.device
            and A.data_ptr() % 16 == 0
            and B.data_ptr() % 16 == 0
        ):
            cu_count = torch.cuda.get_device_properties(A.device).multi_processor_count
            return ops.wvSplitKQBlockScale(B, A, As, Bs, cu_count)
        return super().apply_block_scaled_mm(A, B, As, Bs)
