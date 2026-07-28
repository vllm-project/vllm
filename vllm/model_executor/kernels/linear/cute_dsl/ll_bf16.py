# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import torch

from vllm.model_executor.warmup.jit_warmup import VllmJitKernel, zip_inputs
from vllm.model_executor.warmup.jit_warmup_cutedsl_helper import compile_cutedsl

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

    _cute_ctx = cute
    return _cute_ctx


def _use_pdl() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_arch_support_pdl()


class LLBf16Gemm(VllmJitKernel["LLBf16Gemm.CompileKey"]):
    # Dot-prod: keyed on (M, K, bs), because M and K are Constexpr.
    # Split-K: keyed on (split_k, num_stages), fully shape-dynamic.
    @dataclass(frozen=True, slots=True)
    class CompileKey:
        backend: Literal["dotprod", "splitk"]
        m: int = 0
        k: int = 0
        bs: int = 0
        split_k: int = 0
        num_stages: int = 0

    @staticmethod
    def kernel(compile_key: CompileKey) -> Any:
        if compile_key.backend == "splitk":
            from ._ll_bf16_splitk import LLBf16SplitK

            return LLBf16SplitK(
                tile_n=16,
                tile_k=256,
                num_stages=compile_key.num_stages,
                num_dma_warps=4,
                split_k=compile_key.split_k,
                use_pdl=_use_pdl(),
            )

        from ._ll_bf16_dotprod import LLBf16Dotprod

        return LLBf16Dotprod(
            k=compile_key.k,
            bs=compile_key.bs,
            use_pdl=_use_pdl(),
        )

    def dispatch(  # type: ignore[override]
        self, *, M: int, K: int, N: int
    ) -> CompileKey:
        dotprod_max_m = _TUNED_DOTPROD_MAX_M.get((K, N), _DEFAULT_DOTPROD_MAX_M)
        is_dotprod = dotprod_max_m >= M or K < 2048
        tuned_config = _TUNED_CONFIGS.get((K, N))
        splitk_config = (
            tuned_config.get(M, _DEFAULT_SPLITK_CONFIG)
            if tuned_config is not None
            else _DEFAULT_SPLITK_CONFIG
        )
        return self.CompileKey(
            backend="dotprod" if is_dotprod else "splitk",
            m=M if is_dotprod else 0,
            k=K if is_dotprod else 0,
            bs=_DEFAULT_DOTPROD_BS if is_dotprod else 0,
            split_k=0 if is_dotprod else splitk_config[0],
            num_stages=0 if is_dotprod else splitk_config[1],
        )

    def get_warmup_keys(
        self,
        *,
        shapes: Iterable[tuple[int, int]],
        m_values: Iterable[int],
    ) -> list[CompileKey]:
        shape_rows = tuple(dict(K=K, N=N) for K, N in shapes)
        m_options = tuple(m_values)
        if not shape_rows or not m_options:
            return []

        return self._trace_dispatch(self.dispatch)(
            zip_inputs(*shape_rows),
            M=m_options,
        )

    def compile(self, compile_key: CompileKey) -> None:
        if self._compiled_cache_contains(compile_key):
            return

        cute = _cute()
        from cutlass import BFloat16, Float32
        from quack.compile_utils import make_fake_tensor

        if compile_key.backend == "splitk":
            hidden_states = make_fake_tensor(
                BFloat16, (cute.sym_int(), cute.sym_int()), divisibility=8
            )
            router_weight = make_fake_tensor(
                BFloat16, (cute.sym_int(), cute.sym_int()), divisibility=8
            )
            output = make_fake_tensor(
                Float32, (cute.sym_int(), cute.sym_int()), divisibility=1
            )
            compiled = compile_cutedsl(
                self.kernel(compile_key),
                hidden_states,
                router_weight,
                output,
            )
            self._compiled_cache[compile_key] = compiled
            logger.debug(
                "Compiled ll_bf16_splitk: sk=%d ns=%d",
                compile_key.split_k,
                compile_key.num_stages,
            )
            return

        N = cute.sym_int()
        stride_divisibility = math.gcd(8, compile_key.k)
        hidden_states = make_fake_tensor(
            BFloat16,
            (compile_key.m, compile_key.k),
            divisibility=stride_divisibility,
        )
        router_weight = make_fake_tensor(
            BFloat16,
            (N, compile_key.k),
            divisibility=stride_divisibility,
        )
        output = make_fake_tensor(Float32, (compile_key.m, N), divisibility=1)
        compiled = compile_cutedsl(
            self.kernel(compile_key),
            hidden_states,
            router_weight,
            output,
            compile_key.m,
            compile_key.k,
            1,  # runtime N placeholder for fake-tensor compile
            options="--enable-tvm-ffi --ptxas-options -maxrregcount=64",
        )
        self._compiled_cache[compile_key] = compiled
        logger.debug(
            "Compiled ll_bf16_dotprod: M=%d, K=%d, bs=%d",
            compile_key.m,
            compile_key.k,
            compile_key.bs,
        )

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
        runtime_context = {"M": M, "K": K, "N": N}
        compiled = self._get_or_compile(
            compile_key,
            runtime_context={**runtime_context, "backend": compile_key.backend},
        )

        output = torch.empty(M, N, dtype=output_dtype, device=hidden_states.device)
        if compile_key.backend == "splitk":
            compiled(hidden_states, router_weight, output)
        else:
            compiled(hidden_states, router_weight, output, N)
        return output


LL_BF16_GEMM_KERNEL = LLBf16Gemm()
