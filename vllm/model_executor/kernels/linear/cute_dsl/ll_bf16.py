# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import torch

logger = logging.getLogger(__name__)

_cutedsl_available: bool | None = None


def is_available() -> bool:
    global _cutedsl_available
    if _cutedsl_available is not None:
        return _cutedsl_available
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401

        _cutedsl_available = True
    except ImportError:
        _cutedsl_available = False
        logger.info("cuteDSL (CUTLASS Python) not available, ll_bf16_gemm disabled")
    return _cutedsl_available


_DEFAULT_DOTPROD_BS = 128
_DEFAULT_DOTPROD_MAX_M = 4
_DEFAULT_SPLITK_CONFIG = (6, 4)
_TUNED_DOTPROD_MAX_M: dict[tuple[int, int], int] = {
    (7168, 256): 6,
}
_TUNED_CONFIGS: dict[tuple[int, int], dict[int, tuple[int, int]]] = {
    (7168, 384): {
        5: (4, 4),
        **{M: (5, 4) for M in range(6, 17)},
    },
}


_cute_ctx = None


def _cute():
    global _cute_ctx
    if _cute_ctx is not None:
        return _cute_ctx
    import cutlass.cute as cute
    from cuda.bindings.driver import CUstream

    _cute_ctx = (cute, CUstream)
    return _cute_ctx


def _stream():
    _, CUstream = _cute()
    from vllm.utils.torch_utils import current_stream

    return CUstream(current_stream().cuda_stream)


def _use_pdl() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_arch_support_pdl()


class LLBf16Gemm:
    @dataclass(frozen=True, slots=True)
    class CompileKey:
        backend: Literal["dotprod", "splitk"]
        M: int = 0
        K: int = 0
        bs: int = 0
        split_k: int = 0
        num_stages: int = 0

    def __init__(self) -> None:
        # Dot-prod: keyed on (M, K, bs), because M and K are Constexpr.
        self._compiled_cache: dict[tuple[int, int, int], Any] = {}
        # Split-K: keyed on (split_k, num_stages), fully shape-dynamic.
        self._splitk_cache: dict[tuple[int, int], Any] = {}

    def dispatch(self, *, M: int, K: int, N: int) -> CompileKey:
        dotprod_max_m = _TUNED_DOTPROD_MAX_M.get((K, N), _DEFAULT_DOTPROD_MAX_M)
        if dotprod_max_m >= M or K < 2048:
            return self.CompileKey(backend="dotprod", M=M, K=K, bs=_DEFAULT_DOTPROD_BS)

        split_k, num_stages = _TUNED_CONFIGS.get((K, N), {}).get(
            M, _DEFAULT_SPLITK_CONFIG
        )
        return self.CompileKey(backend="splitk", split_k=split_k, num_stages=num_stages)

    def get_warmup_keys(
        self,
        *,
        shapes: Iterable[tuple[int, int]],
        m_values: Iterable[int],
    ) -> list[CompileKey]:
        return list(
            dict.fromkeys(
                self.dispatch(M=M, K=K, N=N) for K, N in shapes for M in m_values
            )
        )

    @staticmethod
    def _fake_gemm_tensors(*, M, K, N, divisibility: int):
        from cutlass import BFloat16, Float32
        from quack.compile_utils import make_fake_tensor

        hidden_states = make_fake_tensor(BFloat16, (M, K), divisibility=divisibility)
        router_weight = make_fake_tensor(BFloat16, (N, K), divisibility=divisibility)
        output = make_fake_tensor(Float32, (M, N), divisibility=1)
        return hidden_states, router_weight, output

    def _compile_splitk(self, compile_key: CompileKey) -> None:
        cute, _ = _cute()
        from ._ll_bf16_splitk import LLBf16SplitK

        hidden_states, router_weight, output = self._fake_gemm_tensors(
            M=cute.sym_int(),
            K=cute.sym_int(),
            N=cute.sym_int(),
            divisibility=8,
        )
        gemm = LLBf16SplitK(
            tile_n=16,
            tile_k=256,
            num_stages=compile_key.num_stages,
            num_dma_warps=4,
            split_k=compile_key.split_k,
            use_pdl=_use_pdl(),
        )
        compiled = cute.compile(
            gemm,
            hidden_states,
            router_weight,
            output,
            _stream(),
            options="--enable-tvm-ffi",
        )
        self._splitk_cache[(compile_key.split_k, compile_key.num_stages)] = compiled
        logger.debug(
            "Compiled ll_bf16_splitk: sk=%d ns=%d",
            compile_key.split_k,
            compile_key.num_stages,
        )

    def _compile_dotprod(self, compile_key: CompileKey) -> None:
        cute, _ = _cute()
        from ._ll_bf16_dotprod import LLBf16Dotprod

        N = cute.sym_int()
        stride_divisibility = math.gcd(8, compile_key.K)
        hidden_states, router_weight, output = self._fake_gemm_tensors(
            M=compile_key.M,
            K=compile_key.K,
            N=N,
            divisibility=stride_divisibility,
        )
        gemm = LLBf16Dotprod(k=compile_key.K, bs=compile_key.bs, use_pdl=_use_pdl())
        compiled = cute.compile(
            gemm,
            hidden_states,
            router_weight,
            output,
            compile_key.M,
            compile_key.K,
            1,  # runtime N placeholder for fake-tensor compile
            _stream(),
            options="--enable-tvm-ffi --ptxas-options -maxrregcount=64",
        )
        self._compiled_cache[(compile_key.M, compile_key.K, compile_key.bs)] = compiled
        logger.debug(
            "Compiled ll_bf16_dotprod: M=%d, K=%d, bs=%d",
            compile_key.M,
            compile_key.K,
            compile_key.bs,
        )

    def compile(self, compile_key: CompileKey) -> None:
        if compile_key.backend == "splitk":
            splitk_cache_key = (compile_key.split_k, compile_key.num_stages)
            if splitk_cache_key not in self._splitk_cache:
                self._compile_splitk(compile_key)
            return

        dotprod_cache_key = (compile_key.M, compile_key.K, compile_key.bs)
        if dotprod_cache_key not in self._compiled_cache:
            self._compile_dotprod(compile_key)

    def warmup(
        self,
        *,
        shapes: Iterable[tuple[int, int]],
        m_values: Iterable[int],
    ) -> None:
        for compile_key in self.get_warmup_keys(shapes=shapes, m_values=m_values):
            self.compile(compile_key)

    @staticmethod
    def _validate_inputs(
        hidden_states: torch.Tensor,
        router_weight: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> None:
        if hidden_states.dim() != 2 or router_weight.dim() != 2:
            raise ValueError("hidden_states and router_weight must be 2D tensors")
        if (
            hidden_states.dtype != torch.bfloat16
            or router_weight.dtype != torch.bfloat16
        ):
            raise ValueError("hidden_states and router_weight must have dtype=bfloat16")
        if hidden_states.device.type != "cuda" or router_weight.device.type != "cuda":
            raise ValueError(
                "hidden_states and router_weight must have device_type=cuda"
            )
        if hidden_states.device != router_weight.device:
            raise ValueError(
                "hidden_states and router_weight must be on the same CUDA device"
            )
        if output_dtype != torch.float32:
            raise ValueError("ll_bf16_gemm only supports output_dtype=torch.float32")
        if hidden_states.shape[1] != router_weight.shape[1]:
            raise ValueError(
                "hidden_states and router_weight must have matching K dimensions"
            )
        # Kernels use vectorized bf16 loads and require 16-byte row alignment.
        if hidden_states.shape[1] % 8 != 0:
            raise ValueError("ll_bf16_gemm requires K to be divisible by 8")
        if not hidden_states.is_contiguous() or not router_weight.is_contiguous():
            raise ValueError(
                "hidden_states and router_weight must be contiguous row-major inputs"
            )

    def __call__(
        self,
        hidden_states: torch.Tensor,  # [M, K] bf16
        router_weight: torch.Tensor,  # [N, K] bf16
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:  # [M, N] fp32
        self._validate_inputs(hidden_states, router_weight, output_dtype)

        M, K = hidden_states.shape
        N = router_weight.shape[0]
        compile_key = self.dispatch(M=M, K=K, N=N)
        if compile_key.backend == "splitk":
            splitk_cache_key = (compile_key.split_k, compile_key.num_stages)
            if splitk_cache_key not in self._splitk_cache:
                self.compile(compile_key)
            kernel = self._splitk_cache[splitk_cache_key]
        else:
            dotprod_cache_key = (compile_key.M, compile_key.K, compile_key.bs)
            if dotprod_cache_key not in self._compiled_cache:
                self.compile(compile_key)
            kernel = self._compiled_cache[dotprod_cache_key]

        stream = _stream()
        output = torch.empty(M, N, dtype=output_dtype, device=hidden_states.device)
        if compile_key.backend == "splitk":
            kernel(hidden_states, router_weight, output, stream, 1.0)
        else:
            kernel(hidden_states, router_weight, output, N, stream)
        return output


ll_bf16_gemm_kernel = LLBf16Gemm()


def ll_bf16_gemm(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return ll_bf16_gemm_kernel(hidden_states, router_weight, output_dtype)
