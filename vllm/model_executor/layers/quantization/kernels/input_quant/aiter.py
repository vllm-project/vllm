# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform

from .cuda import CudaInputQuantKernel
from .InputQuantKernel import _FP8_DTYPE, InputQuantConfig, InputQuantKernel
from .pytorch import PytorchInputQuantKernel
from .triton import TritonInputQuantKernel


class AiterInputQuantKernel(InputQuantKernel[InputQuantConfig]):
    @classmethod
    def is_supported(cls):
        if not current_platform.is_fp8_fnuz():
            return False, (
                f"aiter operates quantization with \
                {torch.float8_e4m3fnuz} data type and  \
                    this device does not support it."
            )

        return (
            rocm_aiter_ops.is_linear_enabled(),
            "Only supported on ROCm platform \
                with aiter package installed.",
        )

    @classmethod
    def can_implement(cls, config: InputQuantConfig):
        if config.group_shape.is_per_group() and config.group_shape != GroupShape(
            1, 128
        ):
            return (
                False,
                f"aiter group quantization only supports \
                    group shape of (1, 128). given {config.group_shape}",
            )
        return True, ""

    @classmethod
    def ordered_fallback_kernels(cls) -> list[type[InputQuantKernel[InputQuantConfig]]]:
        return [CudaInputQuantKernel, TritonInputQuantKernel, PytorchInputQuantKernel]

    def apply_group_quant(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Fall back to Triton kernel when weight shape is incompatible with aiter.
        use_triton = kwargs.get("use_triton", False)
        if use_triton:
            return TritonInputQuantKernel(self.config).apply_group_quant(
                x, scale, scale_ub
            )

        return rocm_aiter_ops.group_fp8_quant(x, self.group_shape.col)

    def apply_per_token_per_tensor_quant(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not x.is_contiguous() or scale_ub is not None:
            fall_backs = self.ordered_fallback_kernels()
            for kernel in fall_backs:
                if kernel.is_supported()[0] and kernel.can_implement(self.config)[0]:
                    return kernel(self.config).apply(x, scale, scale_ub)

            raise ValueError(
                f"No suitable fallback kernel found for quantization. "
                f"Input contiguous: {x.is_contiguous()},"
                f"scale_ub provided: {scale_ub is not None}, "
                f"config: {self.config}"
            )

        if self.group_shape.is_per_tensor():
            return rocm_aiter_ops.per_tensor_quant(x, _FP8_DTYPE, scale)

        # Per-tensor already handled, so this must be per-token
        assert self.group_shape.is_per_token()
        return rocm_aiter_ops.per_token_quant(x, _FP8_DTYPE, scale)
