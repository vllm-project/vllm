# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod

import torch
from torch._higher_order_ops import auto_functionalized
from torch._ops import OpOverload

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    _normalize_quant_group_shape,
    kFp8Dynamic64Sym,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
    kNvfp4Quant,
)
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform

RMS_OP = torch.ops._C.rms_norm.default
RMS_ADD_OP = torch.ops._C.fused_add_rms_norm.default
ROTARY_OP = torch.ops._C.rotary_embedding.default
FLASHINFER_ROTARY_OP = torch.ops.vllm.flashinfer_rotary_embedding.default

QUANT_OPS: dict[QuantKey, OpOverload] = {
    kFp8StaticTensorSym: torch.ops._C.static_scaled_fp8_quant.default,  # noqa: E501
    kFp8DynamicTensorSym: torch.ops._C.dynamic_scaled_fp8_quant.default,  # noqa: E501
    kFp8DynamicTokenSym: torch.ops._C.dynamic_per_token_scaled_fp8_quant.default,  # noqa: E501
}

if current_platform.is_cuda() and hasattr(torch.ops._C, "scaled_fp4_quant"):
    QUANT_OPS[kNvfp4Quant] = torch.ops._C.scaled_fp4_quant.default  # noqa: E501

if current_platform.is_cuda():
    QUANT_OPS[kFp8Dynamic128Sym] = torch.ops._C.per_token_group_fp8_quant.default  # noqa: E501
    QUANT_OPS[kFp8Dynamic64Sym] = torch.ops._C.per_token_group_fp8_quant.default  # noqa: E501

SILU_MUL_OP = torch.ops._C.silu_and_mul.default


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

    def empty_int64(self, *args, **kws):
        return torch.empty(*args, dtype=torch.int64, device=self.device, **kws)

    def empty_f32(self, *args, **kws):
        return torch.empty(*args, dtype=torch.float32, device=self.device, **kws)

    def inputs(self) -> list[torch.Tensor]:
        """Utility for inputs to the pattern"""
        raise NotImplementedError


class MatcherRotaryEmbedding(MatcherCustomOp):
    def __init__(
        self,
        is_neox: bool,
        head_size: int,
        num_heads: int,
        num_kv_heads: int,
        use_flashinfer: bool = False,
        enabled: bool | None = None,
    ) -> None:
        if enabled is None:
            enabled = RotaryEmbedding.enabled()

        super().__init__(enabled)
        self.is_neox = is_neox
        self.head_size = head_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_size = self.num_heads * self.head_size
        self.kv_size = self.num_kv_heads * self.head_size
        self.rotary_dim = head_size
        if use_flashinfer:
            self.rotary_op = FLASHINFER_ROTARY_OP
        else:
            self.rotary_op = ROTARY_OP

    def inputs(self) -> list[torch.Tensor]:
        positions = self.empty_int64(5)
        query = self.empty(5, self.q_size)
        key = self.empty(5, self.kv_size)
        cos_sin_cache = self.empty(4096, self.rotary_dim)
        return [positions, query, key, cos_sin_cache]

    def forward_custom(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None,
        cos_sin_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        result = auto_functionalized(
            self.rotary_op,
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=cos_sin_cache,
            is_neox=self.is_neox,
        )
        query_out = result[1]
        key_out = result[2] if len(result) > 2 else None
        return query_out, key_out

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None,
        cos_sin_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return RotaryEmbedding.forward_static(
            positions,
            query,
            key,
            self.head_size,
            self.rotary_dim,
            cos_sin_cache,
            self.is_neox,
        )


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
    def __init__(
        self,
        quant_key: QuantKey,
        enabled: bool | None = None,
        has_col_major_scales: bool = False,
        is_e8m0: bool = False,
    ):
        if enabled is None:
            enabled = QuantFP8.enabled()

        super().__init__(enabled)
        self.quant_key = quant_key
        assert quant_key in QUANT_OPS, f"unsupported quantization scheme {quant_key}"
        self.QUANT_OP = QUANT_OPS[quant_key]

        self.has_col_major_scales = has_col_major_scales
        self.is_e8m0 = is_e8m0

        assert quant_key.dtype == current_platform.fp8_dtype(), (
            "Only QuantFP8 supported by"
        )
        assert quant_key.scale2 is None
        self.quant_fp8 = QuantFP8(
            quant_key.scale.static,
            quant_key.scale.group_shape,
            column_major_scales=has_col_major_scales,
            use_ue8m0=is_e8m0,
        )

    def forward_custom(
        self,
        input: torch.Tensor,
        scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = torch.empty(
            input.shape, device=input.device, dtype=self.quant_key.dtype
        )

        if self.quant_key.scale.group_shape.is_per_group():
            assert scale is None
            scale = self.make_scale(input, transposed=self.has_col_major_scales)

            finfo = torch.finfo(self.quant_key.dtype)
            fp8_min = finfo.min
            fp8_max = finfo.max

            _, result, scale = auto_functionalized(
                self.QUANT_OP,
                input=input,
                output_q=result,
                output_s=scale,
                group_size=self.quant_key.scale.group_shape[1],
                eps=1e-10,
                fp8_min=fp8_min,
                fp8_max=fp8_max,
                scale_ue8m0=self.is_e8m0,
            )
            return result, scale

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

    def make_scale(self, input: torch.Tensor, transposed: bool = False):
        normalized_group_shape = _normalize_quant_group_shape(
            input, self.quant_key.scale.group_shape
        )
        scale_shape = (
            input.shape[0] // normalized_group_shape[0],
            input.shape[1] // normalized_group_shape[1],
        )
        if transposed:
            scale_shape = tuple(reversed(scale_shape))
            return torch.empty(
                scale_shape, device=input.device, dtype=torch.float32
            ).permute(-1, -2)

        return torch.empty(scale_shape, device=input.device, dtype=torch.float32)

    def inputs(self) -> list[torch.Tensor]:
        input = self.empty(5, 16)
        if self.quant_key.scale.static:
            return [input, self.empty_f32(1, 1)]

        return [input]


class MatcherSiluAndMul(MatcherCustomOp):
    def __init__(self, enabled: bool | None = None):
        if enabled is None:
            enabled = SiluAndMul.enabled()
        super().__init__(enabled)

    def inputs(self) -> list[torch.Tensor]:
        input = self.empty(5, 4)
        return [input]

    def forward_custom(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        result = auto_functionalized(SILU_MUL_OP, result=out, input=x)
        return result[1]

    def forward_native(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return SiluAndMul.forward_native(x)
