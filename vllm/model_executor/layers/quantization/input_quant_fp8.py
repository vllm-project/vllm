# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm import envs
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape)
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op

# Using the default value (240.0) from pytorch will cause accuracy
# issue on dynamic quantization models. Here use 224.0 for fnuz on ROCm.
_FP8_DTYPE = current_platform.fp8_dtype()
_FP8_FINFO = torch.finfo(_FP8_DTYPE)
_FP8_MAX = 224.0 if current_platform.is_fp8_fnuz() else _FP8_FINFO.max
_FP8_MIN = -224.0 if current_platform.is_fp8_fnuz() else _FP8_FINFO.min
_FP8_MIN_SCALING_FACTOR = 1.0 / (_FP8_MAX * 512.0)


def rocm_aiter_per_tensor_quant_impl(
        x: torch.Tensor, scale: torch.Tensor,
        dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.ops.quant import per_tensor_quant_hip
    return per_tensor_quant_hip(x, scale, dtype)


def rocm_aiter_per_tensor_quant_fake(
        x: torch.Tensor, scale: torch.Tensor,
        dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x, dtype=dtype), torch.empty(1,
                                                         dtype=torch.float32,
                                                         device=x.device)


def rocm_aiter_per_token_quant_impl(
        x: torch.Tensor, scale: Optional[torch.Tensor],
        dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.ops.quant import per_token_quant_hip
    return per_token_quant_hip(x, scale, dtype)


def rocm_aiter_per_token_quant_fake(
        x: torch.Tensor, scale: Optional[torch.Tensor],
        dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    scale_shape = (*x.shape[:-1], 1)
    return torch.empty_like(x, dtype=dtype), torch.empty(scale_shape,
                                                         dtype=torch.float32,
                                                         device=x.device)


direct_register_custom_op(
    op_name="rocm_aiter_per_tensor_quant",
    op_func=rocm_aiter_per_tensor_quant_impl,
    mutates_args=[],
    fake_impl=rocm_aiter_per_tensor_quant_fake,
    dispatch_key=current_platform.dispatch_key,
)

direct_register_custom_op(
    op_name="rocm_aiter_per_token_quant",
    op_func=rocm_aiter_per_token_quant_impl,
    mutates_args=[],
    fake_impl=rocm_aiter_per_token_quant_fake,
    dispatch_key=current_platform.dispatch_key,
)


def use_aiter():
    return envs.VLLM_ROCM_USE_AITER and envs.VLLM_ROCM_USE_AITER_LINEAR


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
        self.use_aiter = use_aiter()

    def forward_cuda(
        self,
        x: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        scale_ub: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert (scale is not None) == self.static
        assert scale_ub is None or (not self.static and self.group_shape
                                    == GroupShape.PER_TOKEN
                                    and scale_ub.numel() == 1)

        return ops.scaled_fp8_quant(
            x,
            scale,
            num_token_padding=self.num_token_padding,
            scale_ub=scale_ub,
            use_per_token_if_dynamic=self.use_per_token_if_dynamic)

    def forward_hip(
        self,
        x: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        scale_ub: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_aiter_per_tensor_quant = (
            self.group_shape == GroupShape.PER_TENSOR and \
            self.use_aiter and \
            scale_ub is not None
        )
        use_aiter_per_token_quant = (
            self.group_shape == GroupShape.PER_TOKEN and \
            self.use_aiter and \
            scale_ub is not None and \
            not self.static
        )

        if use_aiter_per_tensor_quant:
            return torch.ops.vllm.rocm_aiter_per_tensor_quant(
                x, scale, _FP8_DTYPE)
        if use_aiter_per_token_quant:
            return torch.ops.vllm.rocm_aiter_per_token_quant(
                x, scale, _FP8_DTYPE)

        # Fallback to CUDA implementation
        return self.forward_cuda(x, scale, scale_ub)

    def forward_native(
        self,
        x: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        scale_ub: Optional[torch.Tensor] = None,
    ):
        assert (scale is not None) == self.static
        assert scale_ub is None or (not self.static and self.group_shape
                                    == GroupShape.PER_TOKEN
                                    and scale_ub.numel() == 1)

        if scale is None:
            if self.group_shape == GroupShape.PER_TOKEN:
                x_max, _ = x.abs().max(dim=-1)
                x_max = x_max.unsqueeze(-1).to(torch.float32)
                if scale_ub is not None:
                    x_max = x_max.clamp(max=scale_ub)
            else:
                x_max = x.abs().max().unsqueeze(-1).to(torch.float32)

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
