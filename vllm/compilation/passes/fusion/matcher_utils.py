# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch._higher_order_ops import auto_functionalized
from torch._ops import OpOverload

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    _normalize_quant_group_shape,
    kFp8Dynamic64Sym,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform

RMS_ADD_OP = torch.ops._C.fused_add_rms_norm.default

QUANT_OPS: dict[QuantKey, OpOverload] = {
    kFp8StaticTensorSym: torch.ops._C.static_scaled_fp8_quant.default,  # noqa: E501
    kFp8DynamicTensorSym: torch.ops._C.dynamic_scaled_fp8_quant.default,  # noqa: E501
    kFp8DynamicTokenSym: torch.ops._C.dynamic_per_token_scaled_fp8_quant.default,  # noqa: E501
}

if current_platform.is_cuda() and hasattr(torch.ops._C, "scaled_fp4_quant"):
    QUANT_OPS[kNvfp4Dynamic] = torch.ops._C.scaled_fp4_quant.out  # noqa: E501

if current_platform.is_cuda():
    QUANT_OPS[kFp8Dynamic128Sym] = torch.ops._C.per_token_group_fp8_quant.default  # noqa: E501
    QUANT_OPS[kFp8Dynamic64Sym] = torch.ops._C.per_token_group_fp8_quant.default  # noqa: E501

SILU_MUL_OP = torch.ops._C.silu_and_mul.default


class MatcherCustomOp(ABC):
    def __init__(self, enabled: bool) -> None:
        config = get_current_vllm_config()
        self.model_dtype = config.model_config.dtype if config.model_config else None
        self.device = config.device_config.device if config.device_config else None

        self.enabled = enabled
        self.forward = self.forward_custom if enabled else self.forward_native

    @abstractmethod
    def forward_custom(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def forward_native(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def empty(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.empty(*args, dtype=self.model_dtype, device=self.device, **kwargs)

    def empty_int64(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.empty(*args, dtype=torch.int64, device=self.device, **kwargs)

    def empty_f32(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.empty(*args, dtype=torch.float32, device=self.device, **kwargs)

    def inputs(self) -> list[torch.Tensor]:
        """Utility for inputs to the pattern"""
        raise NotImplementedError


class MatcherQuantFP8(MatcherCustomOp):
    def __init__(
        self,
        quant_key: QuantKey,
        enabled: bool | None = None,
        has_col_major_scales: bool = False,
        is_e8m0: bool = False,
        match_rocm_aiter: bool = False,
        is_tma_aligned: bool = False,
    ) -> None:
        if enabled is None:
            enabled = QuantFP8.enabled()

        super().__init__(enabled)
        self.quant_key = quant_key
        self.has_col_major_scales = has_col_major_scales
        self.is_e8m0 = is_e8m0
        self.match_rocm_aiter = match_rocm_aiter
        self.is_tma_aligned = is_tma_aligned

        if match_rocm_aiter:
            assert not quant_key.scale.group_shape.is_per_tensor(), (
                "ROCm aiter fusion pass does not support per tensor quantization"
            )
            if quant_key.scale.group_shape.is_per_token():
                self.QUANT_OP = rocm_aiter_ops.get_per_token_quant_op()
            else:
                assert quant_key.scale.group_shape.col == 128, (
                    "ROCm aiter fusion pass currently supports "
                    "quantization operation with group_size 128"
                )
                if current_platform.is_fp8_fnuz():
                    self.QUANT_OP = rocm_aiter_ops.get_group_quant_op()
                else:
                    self.QUANT_OP = (
                        torch.ops.vllm.triton_per_token_group_quant_fp8.default
                    )

        else:
            assert quant_key in QUANT_OPS, (
                f"unsupported quantization scheme {quant_key}"
            )
            self.QUANT_OP = QUANT_OPS[quant_key]

            assert quant_key.dtype == current_platform.fp8_dtype(), (
                "Only QuantFP8 supported by"
            )
            assert quant_key.scale2 is None

        self.quant_fp8 = QuantFP8(
            quant_key.scale.static,
            quant_key.scale.group_shape,
            column_major_scales=has_col_major_scales,
            use_ue8m0=is_e8m0,
            tma_aligned_scales=self.is_tma_aligned,
            compile_native=False,
        )

    def forward_rocm_aiter(
        self,
        input: torch.Tensor,
        scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        quant_key_group_shape = self.quant_key.scale.group_shape
        if quant_key_group_shape == GroupShape.PER_TOKEN:
            return self.QUANT_OP(  # type: ignore[no-any-return]
                x=input,
                quant_dtype=self.quant_key.dtype,
                scale=scale,
            )
        else:
            return self.QUANT_OP(input, quant_key_group_shape.col)  # type: ignore[no-any-return]

    def forward_custom(
        self,
        input: torch.Tensor,
        scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.match_rocm_aiter:
            return self.forward_rocm_aiter(input, scale)

        result = torch.empty(
            input.shape, device=input.device, dtype=self.quant_key.dtype
        )

        if self.quant_key.scale.group_shape.is_per_group():
            # for tma_aligned, the scale must be passed to forward_custom
            # tma_aligned fusion then matches by custom op arguments
            if not self.is_tma_aligned:
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
                dummy_is_scale_transposed=self.has_col_major_scales,
                dummy_is_tma_aligned=self.is_tma_aligned,
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
        return self.quant_fp8(input, scale)  # type: ignore[no-any-return]

    def make_scale(self, input: torch.Tensor, transposed: bool = False) -> torch.Tensor:
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
    def __init__(self, enabled: bool | None = None) -> None:
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
