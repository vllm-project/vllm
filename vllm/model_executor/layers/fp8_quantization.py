# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape)
from vllm.platforms import current_platform

# Using the default value (240.0) from pytorch will cause accuracy
# issue on dynamic quantization models. Here use 224.0 for fnuz on ROCm.
_FP8_DTYPE = current_platform.fp8_dtype()
_FP8_FINFO = torch.finfo(_FP8_DTYPE)
_FP8_MAX = 224.0 if current_platform.is_fp8_fnuz() else _FP8_FINFO.max
_FP8_MIN = -224.0 if current_platform.is_fp8_fnuz() else _FP8_FINFO.min
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


@CustomOp.register("quant_fp8")
class QuantFP8(CustomOp):
    """
    Quantize input tensor to per-tensor or per-token FP8.
    This CustomOp supports both static and dynamic quantization.
    """

    def __init__(self,
                 static: bool,
                 group_shape: GroupShape,
                 num_token_padding: Optional[int] = None):
        """

        :param static: static or dynamic quantization
        :param group_shape: quantization group shape (PER_TOKEN or PER_TENSOR)
        :param num_token_padding: Pad the token dimension of output to this size
        """
        super().__init__()
        self.num_token_padding = num_token_padding
        assert group_shape in {GroupShape.PER_TOKEN, GroupShape.PER_TENSOR}
        assert not static or group_shape == GroupShape.PER_TENSOR, \
            "Only per-tensor scales supported for static quantization."
        self.static = static
        self.group_shape = group_shape
        self.use_per_token_if_dynamic = group_shape == GroupShape.PER_TOKEN

    def forward_cuda(
        self,
        x: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        scale_ub: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert (scale is not None) == self.static
        assert scale_ub is None or (not self.static and self.group_shape
                                    == GroupShape.PER_TOKEN
                                    and scale_ub.size() == (1, ))

        return ops.scaled_fp8_quant(
            x,
            scale,
            num_token_padding=self.num_token_padding,
            scale_ub=scale_ub,
            use_per_token_if_dynamic=self.use_per_token_if_dynamic)

    def forward_native(
        self,
        x: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        scale_ub: Optional[torch.Tensor] = None,
    ):
        assert (scale is not None) == self.static
        assert scale_ub is None or (not self.static and self.group_shape
                                    == GroupShape.PER_TOKEN
                                    and scale_ub.size() == (1, ))

        if scale is None:
            if self.group_shape == GroupShape.PER_TOKEN:
                x_max, _ = x.abs().max(dim=-1)
                x_max = x_max.unsqueeze(-1).to(torch.float32)
                if scale_ub is not None:
                    x_max = x_max.clamp(max=scale_ub)
            else:
                x_max = x.abs().max().to(torch.float32)

            scale = x_max / _FP8_MAX
            scale = scale.clamp(min=_FP8_MIN_SCALING_FACTOR)

        # Even for dynamic per-token scales,
        # reciprocal performs slightly better than division
        out = x.to(torch.float32) * scale.reciprocal()
        out = out.clamp(_FP8_MIN, _FP8_MAX).to(_FP8_DTYPE)

        # This currently generates an extra Triton kernel in compilation.
        # Fortunately, we don't use padding if compiling.
        # TODO(luka): benchmark torch._scaled_mm to hopefully remove padding
        #  in general.
        if self.num_token_padding is not None:
            padding = max(self.num_token_padding - out.size(0), 0)
            out = F.pad(out, (0, 0, 0, padding), "constant", 0.0)

        return out, scale
