# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    mxfp8_e4m3_quantize,
    swizzle_mxfp8_scale,
    xpu_mxfp8_quantize,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig

_E8M0 = torch.float8_e8m0fnu

class TorchMxFp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A8 GEMM using the native ``torch._scaled_mm`` dispatch.

    Supported on XPU (oneDNN) and CUDA (cuBLASLt, SM90+).
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if current_platform.is_xpu():
            return True, None
        if current_platform.is_cuda() and current_platform.has_device_capability(90):
            return True, None
        return False, "requires XPU or CUDA (>=sm_90) for native torch._scaled_mm"

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # weight: [N, K] fp8_e4m3; weight_scale: [N, K//32] (E8M0-as-uint8).
        weight = layer.weight.data  # [N, K]
        N, K = weight.shape
        scale_k = K // MXFP8_BLOCK_SIZE

        weight_scale = layer.weight_scale.data[:N, :scale_k].contiguous()

        if current_platform.is_xpu():
            # oneDNN: un-swizzled scales; store column-major weight (B operand).
            replace_parameter(layer, "weight", weight.t())
            weight_scale = weight_scale.t().contiguous().view(_E8M0)
        else:
            # CUDA: L4-padded SWIZZLE_32_4_4 scales; keep [N, K] weight and
            # transpose to column-major at apply time.
            replace_parameter(layer, "weight", weight.contiguous())
            weight_scale = swizzle_mxfp8_scale(weight_scale, M=N, K=K).view(_E8M0)

        replace_parameter(layer, "weight_scale", weight_scale)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_dtype = x.dtype
        weight = layer.weight
        weight_scale = layer.weight_scale
        input_shape = x.shape

        if current_platform.is_xpu():
            # weight stored column-major [K, N].
            K, N = weight.shape
            mat_b = weight
            input_2d = x.reshape(-1, K)
            # xpu_mxfp8_quantize already returns float8_e8m0fnu, [M, K//32].
            x_fp8, x_scale = xpu_mxfp8_quantize(input_2d)
        else:
            # weight stored [N, K]; transpose to column-major for _scaled_mm.
            N, K = weight.shape
            mat_b = weight.t()
            input_2d = x.reshape(-1, K)
            x_fp8, x_scale = mxfp8_e4m3_quantize(
                input_2d, is_sf_swizzled_layout=True
            )
            x_scale = x_scale.view(_E8M0)

        out = torch._scaled_mm(
            x_fp8,
            mat_b,
            scale_a=x_scale,
            scale_b=weight_scale,
            bias=bias,
            out_dtype=out_dtype,
        )

        return out.reshape(*input_shape[:-1], N)