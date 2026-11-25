# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    mxfp8_e4m3_quantize,
    swizzle_mxfp8_scale,
    xpu_mxfp8_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp8Dynamic,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig

_E8M0 = torch.float8_e8m0fnu


class TorchMxFp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A8 GEMM using the native `torch._scaled_mm` dispatch.

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

    def input_quant_key(self) -> QuantKey | None:
        # CUDA activations use FlashInfer's F8_128x4 swizzled scale layout.
        return kMxfp8Dynamic if current_platform.is_cuda() else None

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
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        weight_scale = layer.weight_scale
        qa = as_quantized_activation(x, self.input_quant_key())
        if qa is not None:
            out_dtype, input_shape = qa.orig_dtype, qa.orig_shape
        else:
            assert isinstance(x, torch.Tensor)
            out_dtype, input_shape = layer.params_dtype, x.shape

        if current_platform.is_xpu():
            assert isinstance(x, torch.Tensor)
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
            if qa is not None:
                x_fp8, x_scale = qa.data, qa.scale
            else:
                assert isinstance(x, torch.Tensor)
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
