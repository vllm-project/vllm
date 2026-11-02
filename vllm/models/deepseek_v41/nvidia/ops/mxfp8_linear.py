# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 MXFP8 linear layers that run large GEMMs on cuBLASLt."""

import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
    FlashInferCutedslMxfp8LinearKernel,
)
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
)
from vllm.model_executor.layers.quantization.modelopt import ModelOptLinearMethod
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.utils.flashinfer import is_flashinfer_autotuning


def cublaslt_mxfp8_apply(
    kernel: FlashInferCutedslMxfp8LinearKernel,
    layer: torch.nn.Module,
    x: torch.Tensor | QuantizedActivation,
    bias: torch.Tensor | None,
    min_m: int,
) -> torch.Tensor:
    """``kernel.apply_weights``, with GEMMs of at least ``min_m`` rows on
    cuBLASLt.

    ``torch._scaled_mm`` reads the CuTe-DSL operands as they are: E4M3 data,
    column-major weights and 128x4-swizzled E8M0 scales. Smaller GEMMs and
    FlashInfer autotuning stay on CuTe-DSL, so its small-M tactics stay tuned.
    """
    K, N = layer.weight.shape
    qa = as_quantized_activation(x, kernel.input_quant_key())
    if qa is not None:
        m = qa.data.shape[0]
    else:
        assert isinstance(x, torch.Tensor)
        m = x.numel() // K
    if m < min_m or N % 16 or is_flashinfer_autotuning():
        return kernel.apply_weights(layer, x, bias)

    if qa is not None:
        input_mxfp8, input_scale = qa.data, qa.scale
        out_dtype, input_shape = qa.orig_dtype, qa.orig_shape
    else:
        assert isinstance(x, torch.Tensor)
        out_dtype, input_shape = layer.params_dtype, x.shape
        input_mxfp8, input_scale = mxfp8_e4m3_quantize(
            x.view(-1, K), is_sf_swizzled_layout=True
        )
    output = torch._scaled_mm(
        input_mxfp8,
        layer.weight,
        scale_a=input_scale.view(torch.float8_e8m0fnu),
        scale_b=layer.weight_scale.view(torch.float8_e8m0fnu),
        out_dtype=out_dtype,
    )
    if bias is not None:
        output = output + bias
    return output.view(*input_shape[:-1], N)


class DeepseekV41Mxfp8LinearMethod(ModelOptLinearMethod):
    """Applies an automatically selected CuTe-DSL kernel through
    ``cublaslt_mxfp8_apply``.

    The kernel object is kept as is, so DeepSeek-V4.1 fusions that match its
    exact type still apply. Explicit linear backends are left alone.
    """

    cublaslt_min_m = 0

    def create_weights(self, layer: torch.nn.Module, *args, **kwargs) -> None:
        super().create_weights(layer, *args, **kwargs)
        config = get_current_vllm_config_or_none()
        if config is not None:
            kc = config.kernel_config
            per_quant = kc.linear_backend_per_quant or {}
            if per_quant.get("mxfp8", kc.linear_backend) == "auto":
                self.cublaslt_min_m = envs.VLLM_DSV41_MXFP8_CUBLASLT_MIN_M

    def apply(self, layer, x, bias=None):
        if (
            not self.cublaslt_min_m
            or type(self.kernel) is not FlashInferCutedslMxfp8LinearKernel
        ):
            return super().apply(layer, x, bias)
        return self.fmt.apply(
            layer,
            x,
            bias,
            lambda lyr, inp, b: cublaslt_mxfp8_apply(
                self.kernel, lyr, inp, b, self.cublaslt_min_m
            ),
        )
