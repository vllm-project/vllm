# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shape-dynamic W8A16 skinny GEMM for 128x128 block-scaled FP8 weights."""

from __future__ import annotations

from collections.abc import Iterable
from functools import partial

import torch

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    CuteDslGemm,
    SkinnyGemmConfig,
    a_rows_aligned,
)

QUANT_BLOCK = 128
# A is read in 128-bit vectors of BF16; B in 64- or 128-bit vectors of FP8.
_A_VECTOR = 8
_B_VECTOR_WIDTHS = (8, 16)


def can_use_skinny_fp8_gemm(
    a: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    config: SkinnyGemmConfig,
) -> bool:
    """Whether ``a @ dequant(weight, weight_scale).T`` fits ``config``.

    ``weight`` must be a row-major (N, K) FP8 E4M3 tensor and ``weight_scale``
    a row-major FP32 (ceil(N / 128), K / 128) tensor, the layout the SM90
    block-FP8 linear kernels keep after loading. ``config.static_k`` is not
    supported by this kernel.
    """
    if a.dim() != 2 or weight.dim() != 2 or weight_scale.dim() != 2:
        return False
    m, k = a.shape
    n = weight.shape[0]
    scale_shape = ((n + QUANT_BLOCK - 1) // QUANT_BLOCK, k // QUANT_BLOCK)
    return (
        m == config.num_rows
        and config.static_k is None
        and config.vector_width in _B_VECTOR_WIDTHS
        and a.dtype == torch.bfloat16
        and weight.dtype == torch.float8_e4m3fn
        and weight_scale.dtype == torch.float32
        and weight.shape[1] == k
        and weight.is_contiguous()
        and weight_scale.is_contiguous()
        and tuple(weight_scale.shape) == scale_shape
        and n % config.outputs_per_block == 0
        and k % (config.block_size * config.vector_width) == 0
        and a_rows_aligned(a, _A_VECTOR)
    )


class ShapeDynamicSkinnyFp8Gemm(CuteDslGemm):
    def __init__(self) -> None:
        super().__init__()
        self._warmup_configs: set[SkinnyGemmConfig] = set()

    def _compile(self, config: SkinnyGemmConfig) -> None:
        if config in self._compiled:
            return
        import cutlass
        import cutlass.cute as cute
        from quack.compile_utils import make_fake_tensor

        from ._skinny_fp8_gemm import CuteSkinnyFp8BlockGemm

        n = cute.sym_int(divisibility=config.outputs_per_block)
        k = cute.sym_int(divisibility=config.block_size * config.vector_width)
        n_scale = cute.sym_int()
        k_scale = cute.sym_int()
        a = make_fake_tensor(cutlass.BFloat16, (config.num_rows, k), divisibility=8)
        b = make_fake_tensor(
            cutlass.Float8E4M3FN, (n, k), divisibility=config.vector_width
        )
        s = make_fake_tensor(cutlass.Float32, (n_scale, k_scale), divisibility=1)
        c = make_fake_tensor(cutlass.BFloat16, (config.num_rows, n), divisibility=1)
        kernel = CuteSkinnyFp8BlockGemm(
            num_rows=config.num_rows,
            block_size=config.block_size,
            outputs_per_block=config.outputs_per_block,
            vector_width=config.vector_width,
            k_unroll=config.k_unroll,
            use_pdl=self._use_pdl(),
        )
        self._compiled[config] = cute.compile(
            kernel,
            a,
            b,
            s,
            c,
            self._stream(),
            options="--enable-tvm-ffi --ptxas-options -maxrregcount=64",
        )

    def request_warmup_configs(self, configs: Iterable[SkinnyGemmConfig]) -> None:
        """Request compilation of measured configs before CUDA graph capture."""
        self._warmup_configs.update(configs)
        self._register_warmup_provider()

    def get_cutedsl_warmup_compile_units(self):
        from vllm.model_executor.warmup.cutedsl_warmup import CuTeDSLCompileUnit

        return tuple(
            CuTeDSLCompileUnit(
                name="shape-dynamic block-FP8 skinny GEMM",
                key=("shape-dynamic-skinny-fp8-gemm", config),
                compile=partial(self._compile, config),
            )
            for config in sorted(
                self._warmup_configs,
                key=lambda c: (
                    c.num_rows,
                    c.block_size,
                    c.outputs_per_block,
                    c.k_unroll,
                    c.vector_width,
                ),
            )
        )

    def __call__(
        self,
        a: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        config: SkinnyGemmConfig,
    ) -> torch.Tensor:
        if not can_use_skinny_fp8_gemm(a, weight, weight_scale, config):
            raise ValueError("inputs do not fit the skinny FP8 GEMM config")
        self._compile(config)
        output = torch.empty(
            (a.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=a.device
        )
        self._compiled[config](a, weight, weight_scale, output, self._stream())
        return output


shape_dynamic_skinny_fp8_gemm = ShapeDynamicSkinnyFp8Gemm()
