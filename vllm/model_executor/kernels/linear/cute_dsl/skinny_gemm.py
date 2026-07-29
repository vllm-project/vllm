# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch

logger = logging.getLogger(__name__)

_cutedsl_available: bool | None = None


@dataclass(frozen=True, slots=True)
class SkinnyGemmConfig:
    num_rows: int
    block_size: int
    outputs_per_block: int
    k_unroll: int = 1
    vector_width: int = 8


class ShapeDynamicSkinnyGemm:
    def __init__(self) -> None:
        self._compiled: dict[tuple[torch.dtype, SkinnyGemmConfig, bool], Any] = {}
        self._warmup_configs: set[tuple[torch.dtype, SkinnyGemmConfig, bool]] = set()
        self._warmup_registered = False

    @staticmethod
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
            logger.info("cuteDSL is not available; skinny GEMM is disabled")
        return _cutedsl_available

    @staticmethod
    def _config(m: int, n: int, k: int) -> SkinnyGemmConfig:
        num_rows = m
        wide_block = 224
        if m == 1 and k >= 7168 and k % (wide_block * 8) == 0:
            if n % 3 == 0:
                k_unroll = 2 if n <= 2304 else 4
                return SkinnyGemmConfig(num_rows, wide_block, 3, k_unroll)
            if 2304 < n < 4096 and n % 2 == 0:
                return SkinnyGemmConfig(num_rows, wide_block, 2, k_unroll=4)

        if k <= 2048 or k % (128 * 8) != 0:
            outputs_per_block = 2 if k <= 2048 else 4
            if n % outputs_per_block:
                outputs_per_block = 1
            if k % (64 * 8) == 0:
                return SkinnyGemmConfig(num_rows, 64, outputs_per_block, 2)
            if k % (32 * 8) == 0:
                return SkinnyGemmConfig(num_rows, 32, outputs_per_block, 2)
            return SkinnyGemmConfig(num_rows, 32, outputs_per_block, 2, vector_width=4)

        block_size = 64 if 4096 <= n < 8192 else 128
        outputs_per_block = 1 if m == 1 and n <= 2304 else 2
        if n % outputs_per_block:
            outputs_per_block = 1
        k_unroll = 2 if n <= 2304 or n >= 16384 else 1
        return SkinnyGemmConfig(
            num_rows,
            block_size,
            outputs_per_block,
            k_unroll=k_unroll,
        )

    @staticmethod
    def _cutlass_dtype(dtype: torch.dtype):
        from cutlass import BFloat16, Float16

        return BFloat16 if dtype == torch.bfloat16 else Float16

    @staticmethod
    def _stream():
        from cuda.bindings.driver import CUstream

        from vllm.utils.torch_utils import current_stream

        return CUstream(current_stream().cuda_stream)

    @staticmethod
    def _use_pdl() -> bool:
        from vllm.platforms import current_platform

        return current_platform.is_arch_support_pdl()

    def _compile(
        self,
        dtype: torch.dtype,
        config: SkinnyGemmConfig,
        has_residual: bool,
    ) -> None:
        import cutlass.cute as cute
        from quack.compile_utils import make_fake_tensor

        from ._skinny_gemm import CuteSkinnyGemm

        element_type = self._cutlass_dtype(dtype)
        n = cute.sym_int(divisibility=config.outputs_per_block)
        k = cute.sym_int(divisibility=config.block_size * config.vector_width)
        a = make_fake_tensor(
            element_type,
            (config.num_rows, k),
            divisibility=config.vector_width,
        )
        b = make_fake_tensor(
            element_type,
            (n, k),
            divisibility=config.vector_width,
        )
        c = make_fake_tensor(element_type, (config.num_rows, n), divisibility=1)
        residual = make_fake_tensor(element_type, (config.num_rows, n), divisibility=1)
        kernel = CuteSkinnyGemm(
            element_type=element_type,
            num_rows=config.num_rows,
            block_size=config.block_size,
            outputs_per_block=config.outputs_per_block,
            vector_width=config.vector_width,
            k_unroll=config.k_unroll,
            has_residual=has_residual,
            use_pdl=self._use_pdl(),
        )
        self._compiled[(dtype, config, has_residual)] = cute.compile(
            kernel,
            a,
            b,
            residual,
            c,
            self._stream(),
            options="--enable-tvm-ffi --ptxas-options -maxrregcount=64",
        )

    def request_warmup(
        self,
        dtype: torch.dtype,
        shapes: Iterable[tuple[int, int, int]],
    ) -> None:
        """Request compilation of ``(M, N, K)`` shapes before graph capture."""
        self.request_warmup_configs(
            dtype,
            (self._config(m, n, k) for m, n, k in shapes),
        )

    def request_warmup_configs(
        self,
        dtype: torch.dtype,
        configs: Iterable[SkinnyGemmConfig],
        *,
        has_residual: bool = False,
    ) -> None:
        """Request compilation of explicit measured configs before capture."""
        self._warmup_configs.update((dtype, config, has_residual) for config in configs)
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
                name=(
                    "shape-dynamic skinny GEMM with residual"
                    if has_residual
                    else "shape-dynamic skinny GEMM"
                ),
                key=("shape-dynamic-skinny-gemm", dtype, config, has_residual),
                compile=partial(self._compile, dtype, config, has_residual),
            )
            for dtype, config, has_residual in sorted(
                self._warmup_configs,
                key=lambda item: (
                    str(item[0]),
                    item[1].num_rows,
                    item[1].block_size,
                    item[1].outputs_per_block,
                    item[1].k_unroll,
                    item[1].vector_width,
                    item[2],
                ),
            )
        )

    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        config: SkinnyGemmConfig | None = None,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if a.dim() != 2 or b.dim() != 2:
            raise ValueError("a and b must be 2D tensors")
        if a.dtype not in (torch.bfloat16, torch.float16) or b.dtype != a.dtype:
            raise ValueError("a and b must have the same BF16 or FP16 dtype")
        if not a.is_cuda or not b.is_cuda or a.device != b.device:
            raise ValueError("a and b must be CUDA tensors on the same device")
        if not a.is_contiguous() or not b.is_contiguous():
            raise ValueError("a and b must be contiguous")
        if a.shape[1] != b.shape[1]:
            raise ValueError("a and b must have matching K dimensions")
        if not 1 <= a.shape[0] <= 16:
            raise ValueError("shape-dynamic skinny GEMM requires 1 <= M <= 16")
        if residual is not None:
            if residual.dim() != 2 or residual.shape != (a.shape[0], b.shape[0]):
                raise ValueError("residual must have shape (M, N)")
            if residual.dtype != a.dtype:
                raise ValueError("residual must have the same dtype as a and b")
            if residual.device != a.device or not residual.is_cuda:
                raise ValueError("residual must be on the same CUDA device")
            if not residual.is_contiguous():
                raise ValueError("residual must be contiguous")

        config = config or self._config(a.shape[0], b.shape[0], a.shape[1])
        if config.num_rows != a.shape[0]:
            raise ValueError("config num_rows must match M")
        if b.shape[0] % config.outputs_per_block != 0:
            raise ValueError("N must be divisible by outputs_per_block")
        if a.shape[1] % (config.block_size * config.vector_width) != 0:
            raise ValueError(
                "K must be divisible by block_size * vector_width for this config"
            )
        has_residual = residual is not None
        cache_key = (a.dtype, config, has_residual)
        if cache_key not in self._compiled:
            self._compile(a.dtype, config, has_residual)
        output = torch.empty((a.shape[0], b.shape[0]), dtype=a.dtype, device=a.device)
        residual_arg = output if residual is None else residual
        self._compiled[cache_key](a, b, residual_arg, output, self._stream())
        return output


shape_dynamic_skinny_gemm = ShapeDynamicSkinnyGemm()
