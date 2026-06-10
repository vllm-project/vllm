# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    xpu_mxfp4_quantize as quant_mxfp4,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .base import MxFp4LinearKernel, MxFp4LinearLayerConfig


class XPUMxFp4LinearKernel(MxFp4LinearKernel):
    """MXFP4 W4A4 GEMM on XPU."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUMxFp4 only support on XPU"
        return True, None

    @classmethod
    def can_implement(cls, c: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.view(torch.float4_e2m1fn_x2)
        replace_parameter(layer, "weight", weight.data.t())

        weight_scale = layer.weight_scale.view(torch.float8_e8m0fnu)
        weight_scale = weight_scale.t().contiguous()
        replace_parameter(layer, "weight_scale", weight_scale.data)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_dtype = x.dtype
        x_fp4, x_blockscale = quant_mxfp4(x)
        return torch.ops._xpu_C.fp4_gemm(
            x_fp4,
            layer.weight,
            x_blockscale,
            layer.weight_scale,
            out_dtype,
            bias,
        )
