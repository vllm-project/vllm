# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn.parameter import Parameter

from vllm.platforms import current_platform

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


class TritonMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A8 GEMM with dynamic K32 activation quantization on SM90."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "Triton MXFP8 requires CUDA"
        if compute_capability is None:
            capability = current_platform.get_device_capability()
            compute_capability = capability.to_int() if capability else None
        if compute_capability != 90:
            return False, "Triton MXFP8 currently supports SM90 only"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        if c.bmm_batch_size is not None:
            return False, "Triton MXFP8 does not support batched weights"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        if weight.ndim != 2 or weight.dtype != torch.float8_e4m3fn:
            raise ValueError("Triton MXFP8 requires a 2D E4M3 weight")
        n, k = weight.shape
        if not n or not k or k % 32:
            raise ValueError("Triton MXFP8 requires positive N and K divisible by 32")
        scales = layer.weight_scale.data
        if (
            scales.dtype != torch.uint8
            or scales.ndim != 2
            or scales.shape[0] < n
            or scales.shape[1] < k // 32
        ):
            raise ValueError(
                "Triton MXFP8 requires uint8 E8M0 scales covering [N, K/32]"
            )
        layer.weight = Parameter(weight.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(
            scales[:n, : k // 32].contiguous().view(torch.float8_e8m0fnu).float(),
            requires_grad=False,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.triton_mxfp8 import (
            triton_mxfp8_linear,
        )

        output = triton_mxfp8_linear(x, layer.weight, layer.weight_scale)
        if bias is not None:
            output.add_(bias)
        return output
