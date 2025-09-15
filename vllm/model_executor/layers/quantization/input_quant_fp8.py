# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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


@CustomOp.register("quant_fp8")
class QuantFP8(CustomOp):
    """
    Quantize input tensor to FP8 (per-tensor, per-token, or per-group).
    This CustomOp supports both static and dynamic quantization.
    """

    def __init__(self,
                 static: bool,
                 group_shape: GroupShape,
                 num_token_padding: Optional[int] = None,
                 column_major_scales: bool = False):
        """
        :param static: static or dynamic quantization
        :param group_shape: quantization group shape (PER_TOKEN, PER_TENSOR,
            or arbitrary block size)
        :param num_token_padding: Pad the token dimension of output to this
            size
        :param column_major_scales: For group quantization, output scales in
            column major format
        """
        super().__init__()
        self.static = static
        self.group_shape = group_shape
        self.num_token_padding = num_token_padding
        self.column_major_scales = column_major_scales

        self.is_group_quant = group_shape.is_per_group()
        if self.is_group_quant:
            assert not static, "Group quantization only supports dynamic mode"
            self.group_size = group_shape.col
        else:
            assert group_shape in {GroupShape.PER_TOKEN, GroupShape.PER_TENSOR}
            assert not static or group_shape == GroupShape.PER_TENSOR, \
                "Only per-tensor scales supported for static quantization."
            self.use_per_token_if_dynamic = group_shape == GroupShape.PER_TOKEN

    def forward_cuda(
        self,
        x: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        scale_ub: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_group_quant:
            assert scale is None, "Group quantization is always dynamic"
            from vllm.model_executor.layers.quantization.utils import fp8_utils
            return fp8_utils.per_token_group_quant_fp8(
                x,
                group_size=self.group_size,
                column_major_scales=self.column_major_scales,
                dtype=_FP8_DTYPE)

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

    def forward_native(
        self,
        x: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        scale_ub: Optional[torch.Tensor] = None,
    ):
        if self.is_group_quant:
            assert scale is None, "Group quantization is always dynamic"
            return self._quantize_group_native(x)

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

            scale = (x_max / _FP8_MAX).clamp(min=_FP8_MIN_SCALING_FACTOR)

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

    def _quantize_group_native(
            self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        hidden_dim = x.shape[-1]
        num_groups = (hidden_dim + self.group_size - 1) // self.group_size
        padded_dim = num_groups * self.group_size

        if padded_dim != hidden_dim:
            padding = padded_dim - hidden_dim
            x = F.pad(x, (0, padding), mode='constant', value=0.0)

        x_grouped = x.view(-1, num_groups, self.group_size)
        absmax = x_grouped.abs().max(dim=-1, keepdim=True)[0].float()
        scales = (absmax / _FP8_MAX).clamp(min=_FP8_MIN_SCALING_FACTOR)

        x_scaled = x_grouped / scales
        x_quant = x_scaled.clamp(_FP8_MIN, _FP8_MAX).to(_FP8_DTYPE)

        x_quant = x_quant.view(-1, padded_dim)
        if padded_dim != hidden_dim:
            x_quant = x_quant[..., :hidden_dim]
        x_quant = x_quant.view(orig_shape)

        scales = scales.squeeze(-1)
        scales = scales.reshape(orig_shape[:-1] + (num_groups, ))

        if self.column_major_scales:
            scales = scales.transpose(-2, -1).contiguous()

        return x_quant, scales
