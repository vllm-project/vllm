# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_cutedsl_helper import (
    CuTeDSLLaunchSpec,
    VllmCuTeDSLJitKernel,
    compile_cutedsl,
)

logger = logging.getLogger(__name__)

_cutedsl_available: bool | None = None


@dataclass(frozen=True, slots=True)
class SkinnyGemmConfig:
    num_rows: int
    block_size: int
    outputs_per_block: int
    k_unroll: int = 1
    vector_width: int = 8
    static_k: int | None = None


class ShapeDynamicSkinnyGemm(
    VllmCuTeDSLJitKernel["ShapeDynamicSkinnyGemm.CompileKey"]
):
    compile_options = "--enable-tvm-ffi --ptxas-options -maxrregcount=64"

    @dataclass(frozen=True)
    class CompileKey:
        dtype: torch.dtype
        config: SkinnyGemmConfig
        has_residual: bool
        use_pdl: bool

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
    def _cutlass_dtype(dtype: torch.dtype) -> Any:
        from cutlass import BFloat16, Float16

        return BFloat16 if dtype == torch.bfloat16 else Float16

    @staticmethod
    def _stream() -> Any:
        from cuda.bindings.driver import CUstream

        from vllm.utils.torch_utils import current_stream

        return CUstream(current_stream().cuda_stream)

    @staticmethod
    def _use_pdl() -> bool:
        from vllm.platforms import current_platform

        return current_platform.is_arch_support_pdl()

    @staticmethod
    def kernel(compile_key: CompileKey) -> Any:
        from ._skinny_gemm import CuteSkinnyGemm

        config = compile_key.config
        return CuteSkinnyGemm(
            element_type=ShapeDynamicSkinnyGemm._cutlass_dtype(compile_key.dtype),
            num_rows=config.num_rows,
            block_size=config.block_size,
            outputs_per_block=config.outputs_per_block,
            vector_width=config.vector_width,
            k_unroll=config.k_unroll,
            has_residual=compile_key.has_residual,
            use_pdl=compile_key.use_pdl,
            static_k=config.static_k,
        )

    def dispatch(
        self,
        *,
        dtype: torch.dtype,
        config: SkinnyGemmConfig,
        has_residual: bool,
        use_pdl: bool,
    ) -> CompileKey:
        return self.CompileKey(
            dtype=dtype,
            config=config,
            has_residual=has_residual,
            use_pdl=use_pdl,
        )

    def get_warmup_keys(
        self,
        *,
        dtype: torch.dtype,
        configs: tuple[SkinnyGemmConfig, ...],
        has_residual: bool = False,
    ) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(
            dtype=dtype,
            config=configs,
            has_residual=has_residual,
            use_pdl=self._use_pdl(),
        )

    def request_warmup_configs(
        self,
        dtype: torch.dtype,
        configs: Iterable[SkinnyGemmConfig],
        *,
        has_residual: bool = False,
    ) -> None:
        """Compatibility entry point for existing skinny-GEMM selectors."""
        self.register_warmup(
            dtype=dtype,
            configs=tuple(configs),
            has_residual=has_residual,
        )

    def warmup_inputs(self, compile_key: CompileKey) -> tuple[Any, ...]:
        import cutlass.cute as cute
        from quack.compile_utils import make_fake_tensor

        config = compile_key.config
        element_type = self._cutlass_dtype(compile_key.dtype)
        n = cute.sym_int(divisibility=config.outputs_per_block)
        k = (
            config.static_k
            if config.static_k is not None
            else cute.sym_int(divisibility=config.block_size * config.vector_width)
        )
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
        return a, b, residual, c

    def compile(self, compile_key: CompileKey) -> None:
        if compile_key in self._compiled_cache:
            return
        self._compiled_cache[compile_key] = compile_cutedsl(
            self.kernel(compile_key),
            *self.warmup_inputs(compile_key),
            options=self.compile_options,
        )

    @kernel_launcher
    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        config: SkinnyGemmConfig | None = None,
        residual: torch.Tensor | None = None,
    ) -> CuTeDSLLaunchSpec[CompileKey]:
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
        if config.static_k is not None and a.shape[1] != config.static_k:
            raise ValueError("input K must match config static_k")
        output = torch.empty((a.shape[0], b.shape[0]), dtype=a.dtype, device=a.device)
        compile_key = self.dispatch(
            dtype=a.dtype,
            config=config,
            has_residual=residual is not None,
            use_pdl=self._use_pdl(),
        )
        return compile_key, (
            a,
            b,
            output if residual is None else residual,
            output,
            self._stream(),
        ), output


shape_dynamic_skinny_gemm = ShapeDynamicSkinnyGemm()
