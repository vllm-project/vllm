# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization.utils.fp8_quant_ops import (
    quantize_fp8_per_group, quantize_fp8_per_tensor, quantize_fp8_per_token)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape)
from vllm.platforms import current_platform

# Using the default value (240.0) from pytorch will cause accuracy
# issue on dynamic quantization models. Here use 224.0 for fnuz on ROCm.
_FP8_DTYPE = current_platform.fp8_dtype()


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
            return self._quantize_group_cuda(x)

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

        if self.use_per_token_if_dynamic and scale is None:
            out, scale = quantize_fp8_per_token(x, scale, scale_ub)
        else:
            out, scale = quantize_fp8_per_tensor(x, scale)

        # This currently generates an extra Triton kernel in compilation.
        # Fortunately, we don't use padding if compiling.
        # TODO(luka): benchmark torch._scaled_mm to hopefully remove padding
        #  in general.
        if self.num_token_padding is not None:
            padding = max(self.num_token_padding - out.size(0), 0)
            out = F.pad(out, (0, 0, 0, padding), "constant", 0.0)

        return out, scale

    def _quantize_group_cuda(
            self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8)
        return per_token_group_quant_fp8(
            x,
            group_size=self.group_size,
            column_major_scales=self.column_major_scales,
            dtype=_FP8_DTYPE)

    def _quantize_group_native(
            self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return quantize_fp8_per_group(x, self.group_size,
                                      self.column_major_scales)
