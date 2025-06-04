# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform

# Using the default value (240.0) from pytorch will cause accuracy
# issue on dynamic quantization models. Here use 224.0 for rocm.
_FP8_DTYPE = current_platform.fp8_dtype()
_FP8_MAX = 224.0 if current_platform.is_rocm() else torch.finfo(_FP8_DTYPE).max
_FP8_MIN = -224.0 if current_platform.is_rocm() else torch.finfo(
    _FP8_DTYPE).min
_FP8_MIN_SCALING_FACTOR = 1.0 / (_FP8_MAX * 512.0)


@CustomOp.register("quant_fp8_per_token")
class QuantFP8PerToken(CustomOp):
    """
    Quantize input tensor to dynamic per-token FP8 and return quantized
    tensor and scale.

    Args:
        x: The input tensor to be quantized to FP8
        scale_ub: Optional upper bound for scaling factor

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
            scaling factor.
    """

    def forward_native(
        self,
        x: torch.Tensor,
        scale_ub: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_token_max, _ = x.abs().max(dim=-1)
        x_token_max = x_token_max.to(torch.float32)
        if scale_ub is not None:
            x_token_max = x_token_max.clamp(max=scale_ub)
        scales = (x_token_max / _FP8_MAX).unsqueeze(-1)
        scales = scales.clamp(min=_FP8_MIN_SCALING_FACTOR)

        out = x.to(torch.float32) / scales
        out = out.clamp(_FP8_MIN, _FP8_MAX).to(_FP8_DTYPE)
        return out, scales

    def forward_cuda(
        self,
        x: torch.Tensor,
        scale_ub: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return ops.scaled_fp8_quant(x,
                                    scale_ub=scale_ub,
                                    use_per_token_if_dynamic=True)


@CustomOp.register("quant_fp8_per_tensor")
class QuantFP8PerTensor(CustomOp):
    """
    Quantize input tensor to per-tensor FP8 and return quantized tensor and
    scale.

    This function supports both static and dynamic quantization: If you
    provide the scale, it will use static scaling and if you omit it,
    the scale will be determined dynamically.

    Args:
        input: The input tensor to be quantized to FP8
        scale: Optional scaling factor for the FP8 quantization

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
            scaling factor.
    """

    def forward_native(self,
                       x: torch.Tensor,
                       scale: Optional[torch.Tensor] = None
                       ) -> tuple[torch.Tensor, torch.Tensor]:
        if scale is None:
            scale = torch.zeros(1, device=x.device, dtype=torch.float32)
            x_max = x.abs().max().to(torch.float32)
            scale = x_max / _FP8_MAX

        out = (x.to(torch.float32) * scale.reciprocal()).clamp(
            _FP8_MIN, _FP8_MAX).to(_FP8_DTYPE)
        return out, scale.view((1, ))

    def forward_cuda(self,
                     x: torch.Tensor,
                     scale: Optional[torch.Tensor] = None
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        return ops.scaled_fp8_quant(x, scale=scale)
