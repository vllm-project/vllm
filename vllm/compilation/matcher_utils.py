# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod

import torch
from torch._higher_order_ops import auto_functionalized
from torch._ops import OpOverload

from vllm import envs
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    _normalize_quant_group_shape,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
    kNvfp4Quant,
)
from vllm.platforms import current_platform

RMS_OP = torch.ops._C.rms_norm.default
RMS_ADD_OP = torch.ops._C.fused_add_rms_norm.default

QUANT_OPS: dict[QuantKey, OpOverload] = {
    kFp8StaticTensorSym: torch.ops._C.static_scaled_fp8_quant.default,  # noqa: E501
    kFp8DynamicTensorSym: torch.ops._C.dynamic_scaled_fp8_quant.default,  # noqa: E501
    kFp8DynamicTokenSym: torch.ops._C.dynamic_per_token_scaled_fp8_quant.default,  # noqa: E501
}

if current_platform.is_cuda() and hasattr(torch.ops._C, "scaled_fp4_quant"):
    QUANT_OPS[kNvfp4Quant] = torch.ops._C.scaled_fp4_quant.default  # noqa: E501


class MatcherCustomOp(ABC):
    def __init__(self, enabled: bool):
        config = get_current_vllm_config()
        self.model_dtype = config.model_config.dtype if config.model_config else None
        self.device = config.device_config.device if config.device_config else None

        self.enabled = enabled
        self.forward = self.forward_custom if enabled else self.forward_native

    @abstractmethod
    def forward_custom(self, *args, **kws):
        pass

    @abstractmethod
    def forward_native(self, *args, **kws):
        pass

    def __call__(self, *args, **kws):
        return self.forward(*args, **kws)

    def empty(self, *args, **kws):
        return torch.empty(*args, dtype=self.model_dtype, device=self.device, **kws)

    def empty_bf16(self, *args, **kws):
        return torch.empty(*args, dtype=torch.bfloat16, device=self.device, **kws)

    def empty_f32(self, *args, **kws):
        return torch.empty(*args, dtype=torch.float32, device=self.device, **kws)

    def inputs(self) -> list[torch.Tensor]:
        """Utility for inputs to the pattern"""
        raise NotImplementedError


class MatcherRMSNorm(MatcherCustomOp):
    def __init__(self, epsilon: float, enabled: bool | None = None):
        if enabled is None:
            enabled = RMSNorm.enabled()

        super().__init__(enabled)
        self.epsilon = epsilon

    def inputs(self):
        input = self.empty(5, 16) if self.enabled else self.empty_f32(5, 16)
        weight = self.empty(16)
        return [input, weight]

    def forward_custom(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        result = torch.empty_like(input)
        _, result = auto_functionalized(
            RMS_OP,
            result=result,
            input=input,
            weight=weight,
            epsilon=self.epsilon,
        )

        return result

    def forward_native(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        return RMSNorm.forward_static(
            input, self.epsilon, input.size(-1), self.model_dtype, weight
        )


class MatcherFusedAddRMSNorm(MatcherCustomOp):
    def __init__(self, epsilon: float, enabled: bool | None = None):
        if enabled is None:
            enabled = RMSNorm.enabled()

        super().__init__(enabled)
        self.epsilon = epsilon

    def inputs(self):
        input = self.empty(5, 16) if self.enabled else self.empty_f32(5, 16)
        weight = self.empty(16)
        residual = self.empty(5, 16)
        return [input, weight, residual]

    def forward_custom(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, result, residual = auto_functionalized(
            RMS_ADD_OP,
            input=input,
            residual=residual,
            weight=weight,
            epsilon=self.epsilon,
        )

        return result, residual

    def forward_native(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return RMSNorm.forward_static(
            input, self.epsilon, input.size(-1), self.model_dtype, weight, residual
        )


class MatcherQuantFP8(MatcherCustomOp):
    def __init__(self, quant_key: QuantKey, enabled: bool | None = None):
        if enabled is None:
            enabled = QuantFP8.enabled()

        super().__init__(enabled)
        self.quant_key = quant_key
        assert quant_key in QUANT_OPS, f"unsupported quantization scheme {quant_key}"
        self.QUANT_OP = QUANT_OPS[quant_key]

        assert quant_key.dtype == current_platform.fp8_dtype(), (
            "Only QuantFP8 supported by"
        )
        assert quant_key.scale2 is None
        self.quant_fp8 = QuantFP8(quant_key.scale.static, quant_key.scale.group_shape)

    def forward_custom(
        self,
        input: torch.Tensor,
        scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = torch.empty(
            input.shape, device=input.device, dtype=self.quant_key.dtype
        )

        if self.quant_key.scale.static:
            assert scale is not None
            _, result = auto_functionalized(
                self.QUANT_OP, result=result, input=input, scale=scale
            )
            return result, scale
        else:
            assert scale is None
            scale = self.make_scale(input)
            _, result, scale = auto_functionalized(
                self.QUANT_OP, result=result, input=input, scale=scale, scale_ub=None
            )
            return result, scale

    def forward_native(
        self,
        input: torch.Tensor,
        scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.quant_fp8(input, scale)

    def make_scale(self, input: torch.Tensor):
        normalized_group_shape = _normalize_quant_group_shape(
            input, self.quant_key.scale.group_shape
        )
        scale_shape = (
            input.shape[0] // normalized_group_shape[0],
            input.shape[1] // normalized_group_shape[1],
        )

        return torch.empty(scale_shape, device=input.device, dtype=torch.float32)

    def inputs(self) -> list[torch.Tensor]:
        input = self.empty(5, 16)
        if self.quant_key.scale.static:
            return [input, self.empty_f32(1, 1)]

        return [input]


if current_platform.is_rocm() and envs.VLLM_ROCM_USE_AITER:
    from aiter.ops.triton.fused_mul_add import fused_mul_add

    from vllm.utils.torch_utils import direct_register_custom_op

    def rocm_aiter_fused_mul_add_impl(
        x: torch.Tensor, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        # fused_mul_add(x, a, b) computes x * a + b
        x = x.contiguous()
        out = fused_mul_add(x, a, b)
        return out

    def rocm_aiter_fused_mul_add_fake(
        x: torch.Tensor, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        return torch.empty_like(x)

    direct_register_custom_op(
        op_name="rocm_aiter_fused_mul_add",
        op_func=rocm_aiter_fused_mul_add_impl,
        fake_impl=rocm_aiter_fused_mul_add_fake,
    )

    class MatcherAiterFusedMulAdd(MatcherCustomOp):
        def __init__(self, enabled: bool | None = None):
            if enabled is None:
                enabled = current_platform.is_rocm() and envs.VLLM_ROCM_USE_AITER

            super().__init__(enabled)

        def inputs(self):
            x = self.empty_bf16(5, 16)
            a = self.empty_f32(1, 1)
            b = self.empty_bf16(5, 16)
            return [x, a, b]

        def forward_custom(
            self,
            x: torch.Tensor,
            a: torch.Tensor,
            b: torch.Tensor,
        ) -> torch.Tensor:
            # fused_mul_add(x, a, b) computes x * a + b
            out = torch.ops.vllm.rocm_aiter_fused_mul_add(x, a, b)
            return out

        def forward_native(
            self,
            x: torch.Tensor,
            a: torch.Tensor,
            b: torch.Tensor,
        ) -> torch.Tensor:
            mul_result = x * a
            add_result = mul_result + b
            return add_result
