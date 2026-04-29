# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch._higher_order_ops import auto_functionalized

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic64Sym,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform

RMS_ADD_OP = torch.ops._C.fused_add_rms_norm.default
ROTARY_OP = torch.ops._C.rotary_embedding.default
FLASHINFER_ROTARY_OP = torch.ops.vllm.flashinfer_rotary_embedding.default

QUANT_OPS: dict[QuantKey, torch._ops.OpOverloadPacket | torch._ops.OpOverload] = {
    kFp8StaticTensorSym: torch.ops.vllm_ir.static_quant_fp8,
    kFp8DynamicTensorSym: torch.ops.vllm_ir.dynamic_quant_fp8,
    kFp8DynamicTokenSym: torch.ops.vllm_ir.dynamic_quant_fp8,
    kFp8Dynamic128Sym: torch.ops.vllm_ir.dynamic_group_quant_fp8,
    kFp8Dynamic64Sym: torch.ops.vllm_ir.dynamic_group_quant_fp8,
    **(
        {kNvfp4Dynamic: torch.ops._C.scaled_fp4_quant.out}
        if current_platform.is_cuda() and hasattr(torch.ops._C, "scaled_fp4_quant")
        else {}
    ),
}

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


class MatcherRotaryEmbedding(MatcherCustomOp):
    def __init__(
        self,
        is_neox: bool,
        head_size: int,
        num_heads: int,
        num_kv_heads: int,
        use_flashinfer: bool = False,
        match_rocm_aiter: bool | None = None,
        enabled: bool | None = None,
    ) -> None:
        if enabled is None:
            enabled = RotaryEmbedding.enabled()
        if match_rocm_aiter is None:
            match_rocm_aiter = rocm_aiter_ops.is_triton_rotary_embed_enabled()

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
        elif match_rocm_aiter:
            self.rotary_op = rocm_aiter_ops.get_triton_rotary_embedding_op()
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
        result: tuple[torch.Tensor, torch.Tensor | None] = (
            RotaryEmbedding.forward_static(
                positions,
                query,
                key,
                self.head_size,
                self.rotary_dim,
                cos_sin_cache,
                self.is_neox,
            )
        )
        return result


class MatcherFusedAddRMSNorm(MatcherCustomOp):
    def __init__(
        self,
        epsilon: float,
        enabled: bool | None = None,
        match_rocm_aiter: bool = False,
    ) -> None:
        if enabled is None:
            enabled = RMSNorm.enabled()

        super().__init__(enabled)
        self.epsilon = epsilon
        self.match_rocm_aiter = match_rocm_aiter

        self._rmsnorm_op = RMS_ADD_OP

        if match_rocm_aiter:
            self._rmsnorm_op = rocm_aiter_ops.get_rmsnorm_fused_add_op()

    def inputs(self) -> list[torch.Tensor]:
        input = self.empty(5, 16) if self.enabled else self.empty_f32(5, 16)
        weight = self.empty(16)
        residual = self.empty(5, 16)
        return [input, weight, residual]

    def forward_rocm_aiter(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._rmsnorm_op(  # type: ignore[no-any-return]
            x=input, residual=residual, weight=weight, variance_epsilon=self.epsilon
        )

    def forward_custom(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.match_rocm_aiter:
            return self.forward_rocm_aiter(input, weight, residual)

        _, result, residual = auto_functionalized(
            self._rmsnorm_op,
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
        result: tuple[torch.Tensor, torch.Tensor] = RMSNorm.forward_static(
            input, self.epsilon, input.size(-1), self.model_dtype, weight, residual
        )
        return result


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
