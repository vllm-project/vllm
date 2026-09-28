# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapted from vllm/model_executor/kernels/linear/cute_dsl/ll_bf16.py.

Fused Qwen4Exp mHC down projection + SiLU epilogue, where
columns < lora_rank get ``silu(bf16(acc) / hc_count)``, the hc_count
injection-logit columns pass through as ``bf16(acc)``, and the pad columns
are never computed. Output is bf16 (ll_bf16's fp32-output bonus is given up
to preserve the production rounding boundary).

The GEMM uses an FMA backend for M <= 4 and a split-K MMA backend beyond,
both with PDL. The model uses its Linear module above M=48.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Any, Literal

import torch

from vllm.platforms import current_platform

logger = logging.getLogger(__name__)

# The fused kernel stops winning past M ~ 64 on Qwen3.8-Next-Flash.
MAX_FUSED_M = 48
# Split-K settings were tuned only for this (rank, hc, K) shape.
_TUNED_SHAPE = (320, 4, 10240)

_DEFAULT_FMA_MAX_M = 4
_DEFAULT_FMA_BS = 128
# (split_k, num_stages, tile_n)
_DEFAULT_SPLITK_CONFIG = (6, 4, 16)

# SM100f-specific tuned split-K configs for the mHC down shape (K=10240,
# N_compute=324), swept on GB300 under the bit-identity constraint.
_SM100F_TUNED_SPLITK: dict[int, tuple[int, int, int]] = {
    8: (6, 5, 8),
    16: (6, 5, 8),
    32: (6, 5, 16),
}


class HcDownSiluGemm:
    """Dispatch/compile cache for the fused mHC down+SiLU GEMM."""

    @dataclass(frozen=True, slots=True)
    class CompileKey:
        backend: Literal["fma", "mma"]
        M: int = 0
        K: int = 0
        bs: int = 0
        split_k: int = 0
        num_stages: int = 0
        tile_n: int = 0

    def __init__(
        self, rank: int, hc: int, k: int, *, prefetch_pdl_weights: bool = False
    ) -> None:
        self.rank = rank
        self.hc = hc
        self.k = k
        self._prefetch_pdl_weights = prefetch_pdl_weights
        # FMA: keyed on (M, K, bs), because M and K are Constexpr.
        self._fma_cache: dict[tuple[int, int, int], Any] = {}
        # MMA: keyed on (split_k, num_stages, tile_n), fully shape-dynamic.
        self._mma_cache: dict[tuple[int, int, int], Any] = {}
        self._warmup_m: set[int] = set()
        self._warmup_registered = False

    def dispatch(self, m: int) -> CompileKey:
        if m <= _DEFAULT_FMA_MAX_M or self.k < 2048:
            return self.CompileKey(backend="fma", M=m, K=self.k, bs=_DEFAULT_FMA_BS)
        tuned = (
            _SM100F_TUNED_SPLITK
            if (self.rank, self.hc, self.k) == _TUNED_SHAPE
            and current_platform.is_device_capability_family(100)
            else {}
        )
        split_k, num_stages, tile_n = tuned.get(m, _DEFAULT_SPLITK_CONFIG)
        return self.CompileKey(
            backend="mma",
            split_k=split_k,
            num_stages=num_stages,
            tile_n=tile_n,
        )

    @staticmethod
    def _fake_gemm_tensors(*, M, K, N, divisibility: int):
        from cutlass import BFloat16
        from quack.compile_utils import make_fake_tensor

        hidden_states = make_fake_tensor(BFloat16, (M, K), divisibility=divisibility)
        router_weight = make_fake_tensor(BFloat16, (N, K), divisibility=divisibility)
        output = make_fake_tensor(BFloat16, (M, N), divisibility=1)
        return hidden_states, router_weight, output

    def _compile_mma(self, compile_key: CompileKey) -> None:
        import cutlass.cute as cute

        from ._hc_down_silu_mma import HcDownSiluMma

        hidden_states, router_weight, output = self._fake_gemm_tensors(
            M=cute.sym_int(),
            K=cute.sym_int(),
            N=cute.sym_int(),
            divisibility=8,
        )
        gemm = HcDownSiluMma(
            tile_n=compile_key.tile_n,
            num_stages=compile_key.num_stages,
            split_k=compile_key.split_k,
            use_pdl=current_platform.is_arch_support_pdl(),
            rank=self.rank,
            hc=self.hc,
        )
        compiled = cute.compile(
            gemm,
            hidden_states,
            router_weight,
            output,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        self._mma_cache[
            (compile_key.split_k, compile_key.num_stages, compile_key.tile_n)
        ] = compiled
        logger.debug(
            "Compiled hc_down_silu_mma: sk=%d ns=%d tile_n=%d",
            compile_key.split_k,
            compile_key.num_stages,
            compile_key.tile_n,
        )

    def _compile_fma(self, compile_key: CompileKey) -> None:
        import cutlass.cute as cute

        from ._hc_down_silu_fma import HcDownSiluFma

        N = cute.sym_int()
        hidden_states, router_weight, output = self._fake_gemm_tensors(
            M=compile_key.M,
            K=compile_key.K,
            N=N,
            divisibility=8,
        )
        gemm = HcDownSiluFma(
            k=compile_key.K,
            bs=compile_key.bs,
            use_pdl=current_platform.is_arch_support_pdl(),
            prefetch_pdl_weights=self._prefetch_pdl_weights,
            rank=self.rank,
            hc=self.hc,
        )
        compiled = cute.compile(
            gemm,
            hidden_states,
            router_weight,
            output,
            compile_key.M,
            compile_key.K,
            1,  # runtime N placeholder for fake-tensor compile
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi --ptxas-options -maxrregcount=64",
        )
        self._fma_cache[(compile_key.M, compile_key.K, compile_key.bs)] = compiled
        logger.debug(
            "Compiled hc_down_silu_fma: M=%d, K=%d, bs=%d",
            compile_key.M,
            compile_key.K,
            compile_key.bs,
        )

    def compile(self, compile_key: CompileKey) -> None:
        if compile_key.backend == "mma":
            mma_cache_key = (
                compile_key.split_k,
                compile_key.num_stages,
                compile_key.tile_n,
            )
            if mma_cache_key not in self._mma_cache:
                self._compile_mma(compile_key)
            return

        fma_cache_key = (compile_key.M, compile_key.K, compile_key.bs)
        if fma_cache_key not in self._fma_cache:
            self._compile_fma(compile_key)

    def request_warmup(self, m_values: Iterable[int]) -> None:
        m_values = set(m_values)
        if not m_values:
            return
        self._warmup_m.update(m_values)
        if self._warmup_registered:
            return
        from vllm.model_executor.warmup.cutedsl_warmup import (
            register_cutedsl_warmup_provider,
        )

        register_cutedsl_warmup_provider(self)
        self._warmup_registered = True

    def get_cutedsl_warmup_compile_units(self):
        from vllm.model_executor.warmup.cutedsl_warmup import CuTeDSLCompileUnit

        return tuple(
            CuTeDSLCompileUnit(
                name="Qwen4Exp HC down+SiLU GEMM",
                key=(
                    "qwen4-exp-hc-down-silu",
                    self.rank,
                    self.hc,
                    self.k,
                    self._prefetch_pdl_weights,
                    compile_key,
                ),
                compile=partial(self.compile, compile_key),
            )
            for compile_key in dict.fromkeys(
                self.dispatch(m) for m in sorted(self._warmup_m)
            )
        )

    def __call__(
        self,
        hidden_states: torch.Tensor,  # [M, K] bf16
        router_weight: torch.Tensor,  # [N, K] bf16
    ) -> torch.Tensor:  # [M, N] bf16
        M = hidden_states.shape[0]
        N = router_weight.shape[0]
        n_compute = self.rank + self.hc
        w_gemm = router_weight[:n_compute]
        compile_key = self.dispatch(M)
        self.compile(compile_key)
        if compile_key.backend == "mma":
            kernel = self._mma_cache[
                (compile_key.split_k, compile_key.num_stages, compile_key.tile_n)
            ]
        else:
            kernel = self._fma_cache[(compile_key.M, compile_key.K, compile_key.bs)]

        output = torch.empty(M, N, dtype=torch.bfloat16, device=hidden_states.device)
        out_gemm = output[:, :n_compute]
        if compile_key.backend == "mma":
            kernel(hidden_states, w_gemm, out_gemm, 1.0)
        else:
            kernel(hidden_states, w_gemm, out_gemm, n_compute)
        return output


@lru_cache
def _get_kernel(
    rank: int, hc: int, k: int, prefetch_pdl_weights: bool = False
) -> HcDownSiluGemm:
    return HcDownSiluGemm(rank, hc, k, prefetch_pdl_weights=prefetch_pdl_weights)


def hc_down_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    rank: int,
    hc: int,
) -> torch.Tensor:
    """Fused Qwen4Exp HC down projection + SiLU.

    Args:
        x: Normalized hyper-hidden input, [M, K].
        weight: Merged down+inject weight, [N, K].
        rank: Number of low-rank output columns.
        hc: Number of injection-logit output columns.

    Returns:
        [M, weight.shape[0]] bf16 tensor; pad columns are uninitialized when
        the fused kernel runs (production discards them).

    """
    kernel = _get_kernel(rank, hc, weight.shape[1], x.shape[0] == 1)
    return kernel(x, weight)


def request_hc_down_silu_warmup(
    m_values: Iterable[int], rank: int, hc: int, k: int
) -> None:
    """Precompile the fused kernels for CUDA-graph capture token counts.

    Args:
        m_values: Token counts the model may capture CUDA graphs for; values
            outside the fused dispatch range are ignored.
        rank: Number of low-rank output columns.
        hc: Number of injection-logit output columns.
        k: Input feature size.

    """
    if not current_platform.has_device_capability(90) or k % 8 != 0:
        return
    m_set = {int(m) for m in m_values if 1 <= m <= MAX_FUSED_M}
    # M=1 dispatches to the weight-prefetching PDL variant.
    _get_kernel(rank, hc, k, True).request_warmup(m_set & {1})
    _get_kernel(rank, hc, k).request_warmup(m_set - {1})


__all__ = [
    "MAX_FUSED_M",
    "hc_down_silu",
    "request_hc_down_silu_warmup",
]
