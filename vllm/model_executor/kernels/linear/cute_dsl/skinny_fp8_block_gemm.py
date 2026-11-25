# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-M block-scaled FP8 GEMM (CuTe DSL) for decode-sized token counts.

Drop-in for the CUTLASS block-FP8 GEMM (`ops.cutlass_scaled_mm` with 1x128
activation and 128x128 weight scales) at the token counts a measured plan
lists; see skinny_fp8_block_gemm_plans.py.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    ShapeDynamicSkinnyGemm,
)


@dataclass(frozen=True, slots=True)
class SkinnyFp8BlockGemmConfig:
    num_rows: int
    block_size: int
    outputs_per_block: int
    static_k: int


class ShapeDynamicSkinnyFp8BlockGemm:
    def __init__(self) -> None:
        self._compiled: dict[tuple[torch.dtype, SkinnyFp8BlockGemmConfig], Any] = {}
        self._warmup_configs: set[tuple[torch.dtype, SkinnyFp8BlockGemmConfig]] = set()
        self._warmup_registered = False

    @staticmethod
    def is_available() -> bool:
        return ShapeDynamicSkinnyGemm.is_available()

    @staticmethod
    def _cutlass_dtype(dtype: torch.dtype):
        from cutlass import BFloat16, Float16

        return BFloat16 if dtype == torch.bfloat16 else Float16

    def _compile(
        self, out_dtype: torch.dtype, config: SkinnyFp8BlockGemmConfig
    ) -> None:
        import cutlass
        import cutlass.cute as cute
        from quack.compile_utils import make_fake_tensor

        from ._skinny_fp8_block_gemm import CuteSkinnyFp8BlockGemm

        m, k = config.num_rows, config.static_k
        n = cute.sym_int(divisibility=128)
        kernel = CuteSkinnyFp8BlockGemm(
            out_type=self._cutlass_dtype(out_dtype),
            num_rows=m,
            block_size=config.block_size,
            outputs_per_block=config.outputs_per_block,
            static_k=k,
            use_pdl=ShapeDynamicSkinnyGemm._use_pdl(),
        )
        self._compiled[(out_dtype, config)] = cute.compile(
            kernel,
            make_fake_tensor(cutlass.Float8E4M3FN, (m, k), divisibility=16),
            # Column-major activation scales, as the CUTLASS kernel takes them.
            make_fake_tensor(cutlass.Float32, (m, k // 128), leading_dim=0),
            make_fake_tensor(cutlass.Float8E4M3FN, (n, k), divisibility=16),
            make_fake_tensor(cutlass.Float32, (cute.sym_int(), k // 128)),
            make_fake_tensor(self._cutlass_dtype(out_dtype), (m, n), divisibility=1),
            ShapeDynamicSkinnyGemm._stream(),
            options="--enable-tvm-ffi --ptxas-options -maxrregcount=128",
        )

    def request_warmup_configs(
        self, out_dtype: torch.dtype, configs: Iterable[SkinnyFp8BlockGemmConfig]
    ) -> None:
        """Request compilation of planned configs before CUDA graph capture."""
        self._warmup_configs.update((out_dtype, config) for config in configs)
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
                name="shape-dynamic skinny block-FP8 GEMM",
                key=("shape-dynamic-skinny-fp8-block-gemm", out_dtype, config),
                compile=partial(self._compile, out_dtype, config),
            )
            for out_dtype, config in sorted(
                self._warmup_configs,
                key=lambda item: (
                    str(item[0]),
                    item[1].static_k,
                    item[1].num_rows,
                    item[1].block_size,
                    item[1].outputs_per_block,
                ),
            )
        )

    def __call__(
        self,
        a: torch.Tensor,
        a_scale: torch.Tensor,
        b: torch.Tensor,
        b_scale: torch.Tensor,
        config: SkinnyFp8BlockGemmConfig,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        """`(a * a_scale) @ (b * b_scale).T` in `out_dtype`.

        a: (M, K) float8_e4m3fn, row-major. a_scale: (M, K // 128) fp32,
        column-major. b: (N, K) float8_e4m3fn, contiguous. b_scale:
        (N // 128, K // 128) fp32, contiguous.
        """
        m, k = a.shape
        n = b.shape[0]
        assert m == config.num_rows and k == config.static_k
        assert n % 128 == 0 and n % config.outputs_per_block == 0
        cache_key = (out_dtype, config)
        if cache_key not in self._compiled:
            self._compile(out_dtype, config)
        out = torch.empty((m, n), dtype=out_dtype, device=a.device)
        self._compiled[cache_key](
            a, a_scale, b, b_scale, out, ShapeDynamicSkinnyGemm._stream()
        )
        return out


shape_dynamic_skinny_fp8_block_gemm = ShapeDynamicSkinnyFp8BlockGemm()
