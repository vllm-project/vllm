# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    xpu_mxfp8_quantize as quant_mxfp8,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .MXFP8LinearKernel import MXFP8LinearKernel, MXFP8LinearLayerConfig


class XPUMXFP8LinearKernel(MXFP8LinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @classmethod
    def can_implement(cls, c: MXFP8LinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPU MXFP8 Linear only supported on XPU"

        in_features, out_features = c.partition_weight_shape
        if in_features % MXFP8_BLOCK_SIZE or out_features % MXFP8_BLOCK_SIZE:
            return (
                False,
                f"XPU MXFP8 Linear requires in/out features to be multiples of "
                f"{MXFP8_BLOCK_SIZE}, got in_features={in_features}, "
                f"out_features={out_features}",
            )

        return True, None

    def __init__(
        self,
        c: MXFP8LinearLayerConfig,
        w_q_param_name: str,
        w_s_param_name: str,
    ) -> None:
        super().__init__(c, w_q_param_name, w_s_param_name)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_scale = layer.weight_scale.view(torch.float8_e8m0fnu)
        weight_scale = weight_scale.t().contiguous()
        replace_parameter(layer, "weight", layer.weight.t())
        replace_parameter(layer, "weight_scale", weight_scale.data)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_fp8, x_scale = quant_mxfp8(x)
        return torch.ops._xpu_C.fp8_gemm(
            x_fp8,
            layer.weight,
            self.act_type,
            x_scale,
            layer.weight_scale,
            bias,
        )
