# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup import VllmJitKernel, zip_inputs

from ._ll_router_common import (
    current_cuda_stream,
    cute_context,
    is_cutedsl_available,
    make_fake_gemm_tensors,
    use_pdl,
    validate_common_gemm_inputs,
)

logger = logging.getLogger(__name__)


def is_available() -> bool:
    return is_cutedsl_available("ll_fp32w_gemm")


_DEFAULT_DOTPROD_CONFIG = (192, 1, 1)
_DEFAULT_DOTPROD_GROUPED_CONFIG = (192, 2, 1)


_TUNED_DOTPROD_CONFIGS: dict[tuple[int, int], dict[int, tuple[int, int, int]]] = {
    (6144, 128): {
        **{
            M: (384, 1, 1)
            for M in (
                *range(1, 6),
                *range(9, 14, 2),
                17,
                *range(23, 32, 2),
            )
        },
        **{M: (256, 1, 1) for M in (7, 15, *range(19, 22, 2))},
        **{M: (384, 2, 1) for M in range(6, 12, 2)},
        **{M: (192, 2, 1) for M in (*range(12, 26, 2), 32)},
        **{M: (128, 2, 1) for M in range(26, 32, 2)},
    },
    (6144, 256): {
        **{M: (384, 1, 1) for M in range(1, 4)},
        **{M: (128, 1, 1) for M in (*range(4, 7, 2), 11)},
        **{M: (128, 1, 2) for M in range(5, 11, 2)},
        **{M: (128, 2, 2) for M in range(8, 33, 2)},
        **{M: (256, 1, 2) for M in range(13, 33, 2)},
    },
}
_SUPPORTED_ACTIVATION_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


class LLFp32WGemm(VllmJitKernel["LLFp32WGemm.CompileKey"]):
    # Dot-prod is specialized on M/K, tile shape, and activation dtype.
    @dataclass(frozen=True, slots=True)
    class CompileKey:
        m: int
        k: int
        bs: int
        a_dtype: torch.dtype
        token_groups: int = 1
        epb: int = 1

    @staticmethod
    def kernel(compile_key: CompileKey) -> Any:
        from ._ll_fp32w_dotprod import LLFp32WDotprod

        return LLFp32WDotprod(
            m=compile_key.m,
            k=compile_key.k,
            bs=compile_key.bs,
            token_groups=compile_key.token_groups,
            epb=compile_key.epb,
            use_pdl=use_pdl(),
        )

    def dispatch(  # type: ignore[override]
        self,
        *,
        M: int,
        K: int,
        N: int,
        a_dtype: torch.dtype,
    ) -> CompileKey:
        default_config = (
            _DEFAULT_DOTPROD_GROUPED_CONFIG
            if N <= 128 and M >= 12 and M % 2 == 0
            else _DEFAULT_DOTPROD_CONFIG
        )
        tuned_configs = _TUNED_DOTPROD_CONFIGS.get((K, N))
        config = (
            tuned_configs.get(M, default_config) if tuned_configs else default_config
        )
        return self.CompileKey(
            m=M,
            k=K,
            bs=config[0],
            a_dtype=a_dtype,
            token_groups=config[1],
            epb=config[2],
        )

    def get_warmup_keys(
        self,
        *,
        shapes: Iterable[tuple[int, int]],
        m_values: Iterable[int],
        a_dtypes: Iterable[torch.dtype] = (torch.bfloat16,),
    ) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(
            zip_inputs(*(dict(K=K, N=N) for K, N in shapes)),
            M=tuple(m_values),
            a_dtype=tuple(a_dtypes),
        )

    def compile(self, compile_key: CompileKey) -> None:
        if self._compiled_cache_contains(compile_key):
            return
        if compile_key.token_groups not in (1, 2):
            raise ValueError("ll_fp32w_gemm supports token_groups 1 or 2")
        if compile_key.epb not in (1, 2):
            raise ValueError("ll_fp32w_gemm supports epb 1 or 2")
        if compile_key.m % compile_key.token_groups != 0:
            raise ValueError("M must be divisible by token_groups")

        cute, _ = cute_context()
        N = cute.sym_int()
        stride_divisibility = math.gcd(8, compile_key.k)
        hidden_states, router_weight, output = make_fake_gemm_tensors(
            M=compile_key.m,
            K=compile_key.k,
            N=N,
            a_dtype=compile_key.a_dtype,
            b_dtype=torch.float32,
            divisibility=stride_divisibility,
        )
        compiled = cute.compile(
            self.kernel(compile_key),
            hidden_states,
            router_weight,
            output,
            1,  # runtime N placeholder for fake-tensor compile
            current_cuda_stream(),
            options="--enable-tvm-ffi",
        )
        self._compiled_cache[compile_key] = compiled

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
        compiled = self._get_or_compile(
            compile_key,
            runtime_context={"M": M, "K": K, "N": N},
        )

        stream = current_cuda_stream()
        output = torch.empty(M, N, dtype=output_dtype, device=hidden_states.device)
        compiled(hidden_states, router_weight, output, N, stream)
        return output


LL_FP32W_GEMM_KERNEL = LLFp32WGemm()
