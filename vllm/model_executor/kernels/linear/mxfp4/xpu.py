# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    MXFP4_BLOCK_SIZE,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    xpu_mxfp4_quant as quant_mxfp4,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .MXFP4LinearKernel import MXFP4LinearKernel, MXFP4LinearLayerConfig


class XPUMXFP4LinearKernel(MXFP4LinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @classmethod
    def can_implement(cls, c: MXFP4LinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPU MXFP4 Linear only supported on XPU"

        in_features, out_features = c.partition_weight_shape
        if in_features % MXFP4_BLOCK_SIZE or out_features % MXFP4_BLOCK_SIZE:
            return (
                False,
                f"XPU MXFP4 Linear requires in/out features to be multiples of "
                f"{MXFP4_BLOCK_SIZE}, got in_features={in_features}, "
                f"out_features={out_features}",
            )

        return True, None

    def __init__(
        self,
        c: MXFP4LinearLayerConfig,
        w_q_param_name: str,
        w_s_param_name: str,
    ) -> None:
        super().__init__(c, w_q_param_name, w_s_param_name)

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
        orig_dtype = x.dtype
        x_fp4, x_blockscale = quant_mxfp4(x)
        return torch.ops._xpu_C.fp4_gemm(
            x_fp4,
            layer.weight,
            x_blockscale,
            layer.weight_scale,
            orig_dtype,
            bias,
        )
