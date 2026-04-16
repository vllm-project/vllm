# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    xpu_mxfp4_quantize as quant_mxfp4,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .Mxfp4LinearKernel import Mxfp4LinearKernel, Mxfp4LinearLayerConfig


class XPUMxfp4LinearKernel(Mxfp4LinearKernel):
    """MXFP4 W4A4 GEMM on XPU."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUMxFp4 only support on XPU"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight_packed.view(torch.float4_e2m1fn_x2)
        weight = weight.t()
        # Rename CT checkpoint names to standardized names
        layer.weight = Parameter(weight.data, requires_grad=False)
        del layer.weight_packed

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
