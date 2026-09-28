# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapted from vllm/model_executor/kernels/linear/cute_dsl/ll_bf16.py.

Fused Qwen4Exp mHC down projection + SiLU epilogue:
``out[M, 336] = silu_epilogue(x[M, 10240] @ weight[336, 10240]^T)`` where
columns < lora_rank get ``silu(bf16(acc) / hc_count)``, the hc_count
injection-logit columns pass through as ``bf16(acc)``, and the pad columns
are never computed. Output is bf16 (ll_bf16's fp32-output bonus is given up
to preserve the production rounding boundary).

The GEMM uses an FMA backend for M <= 4 and a split-K MMA backend beyond,
both with PDL. It is bit-identical to the unfused
``hc_silu(bf16(ll_bf16_gemm))`` reference on the computed columns. Dispatch
is gated to M <= 48 with an F.linear + hc_silu fallback.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import torch

import vllm.envs as envs
from vllm.platforms import current_platform

from ..hc import hc_silu

logger = logging.getLogger(__name__)

# Qwen4Exp mHC down+inject projection: lora_rank + hc_count + 12 pad rows.
_RANK = 320
_HC = 4
_WEIGHT_N = 336
_WEIGHT_K = 10240
# Pad columns [_N_COMPUTE, _WEIGHT_N) are never computed or read.
_N_COMPUTE = _RANK + _HC
# The fused kernel stops winning past M ~ 64; gate the decode regime it was
# tuned for.
_MAX_FUSED_M = 48

_DEFAULT_FMA_MAX_M = 4
# bs=256 is ~0.1-0.35 us faster on this shape but changes the shuffle
# reduction tree, which breaks bit-identity with the ll_bf16 reference
# (1-ulp flips observed at M=4 on GB300). bs=128 stays.
_DEFAULT_FMA_BS = 128
# (split_k, num_stages, tile_n); tile_n=16 and (6, 4) are the ll_bf16
# defaults. split_k=6 keeps the ll_bf16 K-reduction order and is the only
# split that stays bit-identical; num_stages/tile_n only affect pipelining
# and N-tiling.
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

    def __init__(self, *, prefetch_pdl_weights: bool = False) -> None:
        self._prefetch_pdl_weights = prefetch_pdl_weights
        # FMA: keyed on (M, K, bs), because M and K are Constexpr.
        self._fma_cache: dict[tuple[int, int, int], Any] = {}
        # MMA: keyed on (split_k, num_stages, tile_n), fully shape-dynamic.
        self._mma_cache: dict[tuple[int, int, int], Any] = {}
        self._warmup_m: set[int] = set()
        self._warmup_registered = False

    def dispatch(self, m: int) -> CompileKey:
        if m <= _DEFAULT_FMA_MAX_M:
            return self.CompileKey(backend="fma", M=m, K=_WEIGHT_K, bs=_DEFAULT_FMA_BS)
        tuned = (
            _SM100F_TUNED_SPLITK
            if current_platform.is_device_capability_family(100)
            else {}
        )
        split_k, num_stages, tile_n = tuned.get(m, _DEFAULT_SPLITK_CONFIG)
        return self.CompileKey(
            backend="mma",
            split_k=split_k,
            num_stages=num_stages,
            tile_n=tile_n,
        )

    def get_warmup_keys(self, m_values: Iterable[int]) -> list[CompileKey]:
        return list(dict.fromkeys(self.dispatch(m) for m in m_values))

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
            rank=_RANK,
            hc=_HC,
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
            rank=_RANK,
            hc=_HC,
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
                    self._prefetch_pdl_weights,
                    compile_key,
                ),
                compile=partial(self.compile, compile_key),
            )
            for compile_key in self.get_warmup_keys(sorted(self._warmup_m))
        )

    @staticmethod
    def _validate_inputs(
        hidden_states: torch.Tensor,
        router_weight: torch.Tensor,
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
        if hidden_states.shape[1] != router_weight.shape[1]:
            raise ValueError(
                "hidden_states and router_weight must have matching K dimensions"
            )
        # Kernels use vectorized bf16 loads and require 16-byte row alignment.
        if hidden_states.shape[1] % 8 != 0:
            raise ValueError("hc_down_silu_gemm requires K to be divisible by 8")
        if not hidden_states.is_contiguous() or not router_weight.is_contiguous():
            raise ValueError("hc_down_silu_gemm requires contiguous row-major inputs")

    def __call__(
        self,
        hidden_states: torch.Tensor,  # [M, K] bf16
        router_weight: torch.Tensor,  # [N, K] bf16
    ) -> torch.Tensor:  # [M, N] bf16
        self._validate_inputs(hidden_states, router_weight)

        M, K = hidden_states.shape
        N = router_weight.shape[0]
        w_gemm = router_weight[:_N_COMPUTE]
        compile_key = self.dispatch(M)
        self.compile(compile_key)
        if compile_key.backend == "mma":
            kernel = self._mma_cache[
                (compile_key.split_k, compile_key.num_stages, compile_key.tile_n)
            ]
        else:
            kernel = self._fma_cache[(compile_key.M, compile_key.K, compile_key.bs)]

        output = torch.empty(M, N, dtype=torch.bfloat16, device=hidden_states.device)
        out_gemm = output[:, :_N_COMPUTE]
        if compile_key.backend == "mma":
            kernel(hidden_states, w_gemm, out_gemm, 1.0)
        else:
            kernel(hidden_states, w_gemm, out_gemm, _N_COMPUTE)
        return output


_hc_down_silu_gemm_kernel = HcDownSiluGemm()
_hc_down_silu_gemm_m1_pdl_kernel = HcDownSiluGemm(prefetch_pdl_weights=True)


def is_fused_eligible(weight: torch.Tensor, lora_rank: int, hc_count: int) -> bool:
    """Check the supported down+inject configuration and platform."""
    return (
        lora_rank == _RANK
        and hc_count == _HC
        and weight.shape == (_WEIGHT_N, _WEIGHT_K)
        and weight.dtype == torch.bfloat16
        and current_platform.is_cuda()
        and current_platform.has_device_capability(90)
    )


def hc_down_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Qwen4Exp HC down projection + SiLU, with an unfused fallback.

    Args:
        x: Normalized hyper-hidden input, [M, 10240].
        weight: Merged down+inject weight, [336, 10240].

    Returns:
        [M, weight.shape[0]] bf16 tensor; pad columns are uninitialized when
        the fused kernel runs (production discards them).

    """
    if (
        not envs.VLLM_BATCH_INVARIANT
        and x.shape[0] <= _MAX_FUSED_M
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.is_contiguous()
        and is_fused_eligible(weight, _RANK, _HC)
    ):
        kernel = (
            _hc_down_silu_gemm_m1_pdl_kernel
            if x.shape[0] == 1
            else _hc_down_silu_gemm_kernel
        )
        return kernel(x, weight)
    down = torch.nn.functional.linear(x, weight)
    lora = hc_silu(down[:, :_RANK], _HC)
    return torch.cat([lora, down[:, _RANK:]], dim=1)


def request_hc_down_silu_warmup(m_values: Iterable[int]) -> None:
    """Precompile the fused kernels for CUDA-graph capture token counts.

    Args:
        m_values: Token counts the model may capture CUDA graphs for; values
            outside the fused dispatch range are ignored.

    """
    if not current_platform.has_device_capability(90):
        return
    m_set = {int(m) for m in m_values if 1 <= m <= _MAX_FUSED_M}
    # M=1 dispatches to the weight-prefetching PDL variant.
    _hc_down_silu_gemm_m1_pdl_kernel.request_warmup(m_set & {1})
    _hc_down_silu_gemm_kernel.request_warmup(m_set - {1})


__all__ = [
    "hc_down_silu",
    "is_fused_eligible",
    "request_hc_down_silu_warmup",
]
