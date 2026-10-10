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

import logging
from collections.abc import Iterable
from functools import lru_cache, partial
from typing import Any, Literal

import torch

from vllm.platforms import current_platform

logger = logging.getLogger(__name__)

# The fused kernel stops winning past M ~ 64 on Qwen3.8-Next-Flash.
MAX_FUSED_M = 48

# Split-K configs tuned on SM100f for Qwen3.8-Flash-Next (K=10240, N=324).
_SM100F_TUNED_SPLITK: dict[int, tuple[int, int, int]] = {
    **{m: (6, 5, 8) for m in range(5, 17)},
    **{m: (6, 5, 16) for m in range(17, 33)},
    **{m: (6, 4, 16) for m in range(33, MAX_FUSED_M + 1)},
}

# FMA: (backend, M, K, threadblock_size); MMA: (backend, split_k, stages, tile_n).
CompileKey = tuple[Literal["fma", "mma"], int, int, int]


class HcDownSiluGemm:
    """Dispatch/compile cache for the fused mHC down+SiLU GEMM."""

    def __init__(
        self, rank: int, hc: int, k: int, *, prefetch_pdl_weights: bool = False
    ) -> None:
        self.rank = rank
        self.hc = hc
        self.k = k
        self._prefetch_pdl_weights = prefetch_pdl_weights
        self._compiled: dict[CompileKey, Any] = {}
        self._warmup_m: set[int] = set()
        self._warmup_registered = False

    def dispatch(self, m: int) -> CompileKey:
        if m <= 4 or self.k < 2048:
            return ("fma", m, self.k, 128)
        return ("mma", *_SM100F_TUNED_SPLITK[m])

    def compile(self, compile_key: CompileKey) -> None:
        if compile_key in self._compiled:
            return

        import cutlass.cute as cute
        from cutlass import BFloat16
        from quack.compile_utils import make_fake_tensor

        N = cute.sym_int()
        gemm: Any
        extra_args: tuple[int, ...]
        if compile_key[0] == "mma":
            from ._hc_down_silu_mma import HcDownSiluMma

            _, split_k, num_stages, tile_n = compile_key
            M, K = cute.sym_int(), cute.sym_int()
            gemm = HcDownSiluMma(
                tile_n=tile_n,
                num_stages=num_stages,
                split_k=split_k,
                rank=self.rank,
                hc=self.hc,
            )
            extra_args = ()
            options = "--enable-tvm-ffi"
        else:
            from ._hc_down_silu_fma import HcDownSiluFma

            _, M, K, threadblock_size = compile_key
            gemm = HcDownSiluFma(
                k=K,
                threadblock_size=threadblock_size,
                prefetch_pdl_weights=self._prefetch_pdl_weights,
                rank=self.rank,
                hc=self.hc,
            )
            extra_args = (M, K, 1)  # runtime N placeholder
            options = "--enable-tvm-ffi --ptxas-options -maxrregcount=64"

        hidden_states = make_fake_tensor(BFloat16, (M, K), divisibility=8)
        router_weight = make_fake_tensor(BFloat16, (N, K), divisibility=8)
        output = make_fake_tensor(BFloat16, (M, N), divisibility=1)
        self._compiled[compile_key] = cute.compile(
            gemm,
            hidden_states,
            router_weight,
            output,
            *extra_args,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options=options,
        )
        logger.debug("Compiled hc_down_silu: %s", compile_key)

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
        kernel = self._compiled[compile_key]

        output = torch.empty(M, N, dtype=torch.bfloat16, device=hidden_states.device)
        out_gemm = output[:, :n_compute]
        if compile_key[0] == "mma":
            kernel(hidden_states, w_gemm, out_gemm)
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Qwen4Exp HC down projection + SiLU.

    Args:
        x: Normalized hyper-hidden input, [M, K].
        weight: Merged down+inject weight, [N, K].
        rank: Number of low-rank output columns.
        hc: Number of injection-logit output columns.

    Returns:
        LoRA activations [M, rank] and injection logits [M, hc].

    """
    kernel = _get_kernel(rank, hc, weight.shape[1], x.shape[0] == 1)
    output = kernel(x, weight)
    return output[:, :rank], output[:, rank : rank + hc]


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
