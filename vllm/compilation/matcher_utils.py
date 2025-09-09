# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional, Union

import torch
from torch._higher_order_ops import auto_functionalized
from torch._ops import OpOverload

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    _normalize_quant_group_shape,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
)

RMS_OP = torch.ops._C.rms_norm.default
RMS_ADD_OP = torch.ops._C.fused_add_rms_norm.default

QUANT_OPS: dict[QuantKey, OpOverload] = {
    kFp8StaticTensorSym: torch.ops._C.static_scaled_fp8_quant.default,  # noqa: E501
    kFp8DynamicTensorSym: torch.ops._C.dynamic_scaled_fp8_quant.default,  # noqa: E501
    kFp8DynamicTokenSym: torch.ops._C.dynamic_per_token_scaled_fp8_quant.default,  # noqa: E501
}

# TODO
# if current_platform.is_cuda() and hasattr(torch.ops._C, "scaled_fp4_quant"):
#     QUANT_OPS[
#         kNvfp4Quant] = torch.ops._C.scaled_fp4_quant.default  # noqa: E501


class MatcherRMSNorm:
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if residual is None:
            result = torch.empty_like(input)
            _, result = auto_functionalized(
                RMS_OP,
                result=result,
                input=input,
                weight=weight,
                epsilon=self.epsilon,
            )

            return result
        else:
            _, result, residual = auto_functionalized(
                RMS_ADD_OP,
                input=input,
                residual=residual,
                weight=weight,
                epsilon=self.epsilon,
            )

            return result, residual

    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.forward(input, weight, residual)


class MatcherQuant:
    def __init__(self, quant_key: QuantKey):
        self.quant_key = quant_key
        assert quant_key in QUANT_OPS, f"unsupported quantization scheme {quant_key}"
        self.QUANT_OP = QUANT_OPS[quant_key]

    def forward(
        self, input: torch.Tensor, scale: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # TODO: why does empty_like produce a permute but
        #  empty via shape doesn't?
        result = torch.empty(
            input.shape, device=input.device, dtype=self.quant_key.dtype
        )

        if self.quant_key.scale.static:
            assert scale is not None
            _, result = auto_functionalized(
                self.QUANT_OP, result=result, input=input, scale=scale
            )
            return result
        else:
            assert scale is None
            scale = self.make_scale(input)
            _, result, scale = auto_functionalized(
                self.QUANT_OP, result=result, input=input, scale=scale, scale_ub=None
            )
            return result, scale

    def make_scale(self, input: torch.Tensor):
        normalized_group_shape = _normalize_quant_group_shape(
            input, self.quant_key.scale.group_shape
        )
        scale_shape = (
            input.shape[0] // normalized_group_shape[0],
            input.shape[1] // normalized_group_shape[1],
        )

        return torch.empty(scale_shape, device=input.device, dtype=torch.float32)

    def __call__(
        self, input: torch.Tensor, scale: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.forward(input, scale)
