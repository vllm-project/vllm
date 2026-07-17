# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

from ._ll_router_common import (
    cute_context,
    current_cuda_stream,
    is_cutedsl_available,
    make_fake_gemm_tensors,
    use_pdl,
    validate_common_gemm_inputs,
)

logger = logging.getLogger(__name__)


def is_available() -> bool:
    return is_cutedsl_available("ll_fp32w_gemm")


_DEFAULT_DOTPROD_CONFIG = (192, 1)
_TUNED_DOTPROD_CONFIGS: dict[tuple[int, int], dict[int, tuple[int, int]]] = {
    # MiniMax-M3, measured on B300 with bf16 activations.
    (6144, 128): {
        **{M: (384, 2) for M in (4, 8)},
        9: (384, 1),
        **{M: (256, 1) for M in (14, 16)},
    },
}
_SUPPORTED_ACTIVATION_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


class LLFp32WGemm:
    @dataclass(frozen=True, slots=True)
    class CompileKey:
        M: int
        K: int
        bs: int
        a_dtype: torch.dtype
        token_groups: int = 1

    def __init__(self) -> None:
        # Dot-prod: keyed on static M/K/tile shape and activation dtype.
        self._compiled_cache: dict[tuple[int, int, int, torch.dtype, int], Any] = {}

    def dispatch(
        self,
        *,
        M: int,
        K: int,
        N: int,
        a_dtype: torch.dtype,
    ) -> CompileKey:
        bs, token_groups = _TUNED_DOTPROD_CONFIGS.get((K, N), {}).get(
            M, _DEFAULT_DOTPROD_CONFIG
        )
        return self.CompileKey(
            M=M,
            K=K,
            bs=bs,
            a_dtype=a_dtype,
            token_groups=token_groups,
        )

    def get_warmup_keys(
        self,
        *,
        shapes: Iterable[tuple[int, int]],
        m_values: Iterable[int],
        a_dtypes: Iterable[torch.dtype] = (torch.bfloat16,),
    ) -> list[CompileKey]:
        return list(
            dict.fromkeys(
                self.dispatch(M=M, K=K, N=N, a_dtype=a_dtype)
                for K, N in shapes
                for M in m_values
                for a_dtype in a_dtypes
            )
        )

    @staticmethod
    def _fake_gemm_tensors(
        *,
        M,
        K,
        N,
        a_dtype: torch.dtype,
        divisibility: int,
    ):
        return make_fake_gemm_tensors(
            M=M,
            K=K,
            N=N,
            a_dtype=a_dtype,
            b_dtype=torch.float32,
            divisibility=divisibility,
        )

    def _compile_dotprod(self, compile_key: CompileKey) -> None:
        cute, _ = cute_context()
        from ._ll_fp32w_dotprod import LLFp32WDotprod

        N = cute.sym_int()
        stride_divisibility = math.gcd(8, compile_key.K)
        hidden_states, router_weight, output = self._fake_gemm_tensors(
            M=compile_key.M,
            K=compile_key.K,
            N=N,
            a_dtype=compile_key.a_dtype,
            divisibility=stride_divisibility,
        )
        gemm = LLFp32WDotprod(
            m=compile_key.M,
            k=compile_key.K,
            bs=compile_key.bs,
            token_groups=compile_key.token_groups,
            use_pdl=use_pdl(),
        )
        compiled = cute.compile(
            gemm,
            hidden_states,
            router_weight,
            output,
            1,  # runtime N placeholder for fake-tensor compile
            current_cuda_stream(),
            options="--enable-tvm-ffi --ptxas-options -maxrregcount=64",
        )
        cache_key = (
            compile_key.M,
            compile_key.K,
            compile_key.bs,
            compile_key.a_dtype,
            compile_key.token_groups,
        )
        self._compiled_cache[cache_key] = compiled
        logger.debug(
            "Compiled ll_fp32w_dotprod: M=%d, K=%d, bs=%d, token_groups=%d, a_dtype=%s",
            compile_key.M,
            compile_key.K,
            compile_key.bs,
            compile_key.token_groups,
            compile_key.a_dtype,
        )

    def compile(self, compile_key: CompileKey) -> None:
        if compile_key.token_groups not in (1, 2):
            raise ValueError("ll_fp32w_gemm supports token_groups 1 or 2")
        if compile_key.M % compile_key.token_groups != 0:
            raise ValueError("M must be divisible by token_groups")
        cache_key = (
            compile_key.M,
            compile_key.K,
            compile_key.bs,
            compile_key.a_dtype,
            compile_key.token_groups,
        )
        if cache_key not in self._compiled_cache:
            self._compile_dotprod(compile_key)

    def warmup(
        self,
        *,
        shapes: Iterable[tuple[int, int]],
        m_values: Iterable[int],
        a_dtypes: Iterable[torch.dtype] = (torch.bfloat16,),
    ) -> None:
        for compile_key in self.get_warmup_keys(
            shapes=shapes, m_values=m_values, a_dtypes=a_dtypes
        ):
            self.compile(compile_key)

    @staticmethod
    def _validate_inputs(
        hidden_states: torch.Tensor,
        router_weight: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> None:
        if hidden_states.dtype not in _SUPPORTED_ACTIVATION_DTYPES:
            raise ValueError("hidden_states must have dtype bf16, fp16, or fp32")
        if router_weight.dtype != torch.float32:
            raise ValueError("router_weight must have dtype=float32")
        validate_common_gemm_inputs(
            hidden_states,
            router_weight,
            output_dtype,
            op_name="ll_fp32w_gemm",
        )

    def __call__(
        self,
        hidden_states: torch.Tensor,  # [M, K] bf16/fp16/fp32
        router_weight: torch.Tensor,  # [N, K] fp32
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:  # [M, N] fp32
        self._validate_inputs(hidden_states, router_weight, output_dtype)

        M, K = hidden_states.shape
        N = router_weight.shape[0]
        compile_key = self.dispatch(M=M, K=K, N=N, a_dtype=hidden_states.dtype)
        cache_key = (
            compile_key.M,
            compile_key.K,
            compile_key.bs,
            compile_key.a_dtype,
            compile_key.token_groups,
        )
        if cache_key not in self._compiled_cache:
            self.compile(compile_key)
        kernel = self._compiled_cache[cache_key]

        stream = current_cuda_stream()
        output = torch.empty(M, N, dtype=output_dtype, device=hidden_states.device)
        kernel(hidden_states, router_weight, output, N, stream)
        return output


ll_fp32w_gemm_kernel = LLFp32WGemm()


def ll_fp32w_gemm(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return ll_fp32w_gemm_kernel(hidden_states, router_weight, output_dtype)
