# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped shape-dynamic W8A16 skinny GEMM for the routed experts of a MoE
layer with 128x128 block-scaled FP8 weights."""

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
_A_VECTOR = 8
_B_VECTOR_WIDTHS = (8, 16)


def can_use_skinny_fp8_moe_gemm(
    a: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    config: SkinnyGemmConfig,
    *,
    a_per_token: bool,
) -> bool:
    """Whether the grouped kernel can run ``config`` on these tensors.

    ``weight`` is the (E, N, K) FP8 E4M3 expert stack and ``weight_scale`` its
    (E, N / 128, K / 128) FP32 block scales, both contiguous. ``a`` holds one
    BF16 row per token (``a_per_token``) or per (token, expert) pair.
    ``config.num_rows`` must be 1 (one activation row per CTA) and
    ``config.static_k`` is not supported.
    """
    if a.dim() != 2 or weight.dim() != 3 or weight_scale.dim() != 3:
        return False
    if topk_ids.dim() != 2 or topk_ids.dtype != torch.int32:
        return False
    num_experts, n, k = weight.shape
    num_pairs = topk_ids.numel()
    a_rows = topk_ids.shape[0] if a_per_token else num_pairs
    return (
        config.num_rows == 1
        and config.static_k is None
        and config.vector_width in _B_VECTOR_WIDTHS
        and a.dtype == torch.bfloat16
        and weight.dtype == torch.float8_e4m3fn
        and weight_scale.dtype == torch.float32
        and a.shape == (a_rows, k)
        and weight.is_contiguous()
        and weight_scale.is_contiguous()
        and topk_ids.is_contiguous()
        and tuple(weight_scale.shape)
        == (num_experts, n // QUANT_BLOCK, k // QUANT_BLOCK)
        and n % QUANT_BLOCK == 0
        and n % config.outputs_per_block == 0
        and k % (config.block_size * config.vector_width) == 0
        and a_rows_aligned(a, _A_VECTOR)
    )


class ShapeDynamicSkinnyFp8MoeGemm(CuteDslGemm):
    """One compiled kernel per (config, a_per_token, mul_routed_weight)."""

    def __init__(self) -> None:
        super().__init__()
        self._warmup_configs: set[tuple[SkinnyGemmConfig, bool, bool]] = set()

    def _compile(self, key: tuple[SkinnyGemmConfig, bool, bool]) -> None:
        if key in self._compiled:
            return
        import cutlass
        import cutlass.cute as cute
        from quack.compile_utils import make_fake_tensor

        from ._skinny_fp8_moe_gemm import CuteSkinnyFp8BlockMoeGemm

        config, a_per_token, mul_routed_weight = key
        a_rows = cute.sym_int()
        k = cute.sym_int(divisibility=config.block_size * config.vector_width)
        en = cute.sym_int(divisibility=config.outputs_per_block)
        n_scale = cute.sym_int()
        k_scale = cute.sym_int()
        num_pairs = cute.sym_int()
        n = cute.sym_int(divisibility=config.outputs_per_block)
        a = make_fake_tensor(cutlass.BFloat16, (a_rows, k), divisibility=8)
        b = make_fake_tensor(
            cutlass.Float8E4M3FN, (en, k), divisibility=config.vector_width
        )
        s = make_fake_tensor(cutlass.Float32, (n_scale, k_scale), divisibility=1)
        ids = make_fake_tensor(cutlass.Int32, (num_pairs,), divisibility=1)
        w = make_fake_tensor(cutlass.Float32, (num_pairs,), divisibility=1)
        c = make_fake_tensor(cutlass.BFloat16, (num_pairs, n), divisibility=1)
        kernel = CuteSkinnyFp8BlockMoeGemm(
            block_size=config.block_size,
            outputs_per_block=config.outputs_per_block,
            vector_width=config.vector_width,
            k_unroll=config.k_unroll,
            a_per_token=a_per_token,
            mul_routed_weight=mul_routed_weight,
            use_pdl=self._use_pdl(),
        )
        self._compiled[key] = cute.compile(
            kernel,
            a,
            b,
            s,
            ids,
            w,
            c,
            cutlass.Int32(QUANT_BLOCK),
            cutlass.Int32(1),
            self._stream(),
            options="--enable-tvm-ffi --ptxas-options -maxrregcount=64",
        )

    def request_warmup_configs(
        self, keys: Iterable[tuple[SkinnyGemmConfig, bool, bool]]
    ) -> None:
        """Request compilation of measured configs before CUDA graph capture."""
        self._warmup_configs.update(keys)
        self._register_warmup_provider()

    def get_cutedsl_warmup_compile_units(self):
        from vllm.model_executor.warmup.cutedsl_warmup import CuTeDSLCompileUnit

        return tuple(
            CuTeDSLCompileUnit(
                name="shape-dynamic block-FP8 skinny MoE GEMM",
                key=("shape-dynamic-skinny-fp8-moe-gemm", key),
                compile=partial(self._compile, key),
            )
            for key in sorted(
                self._warmup_configs,
                key=lambda key: (
                    key[0].block_size,
                    key[0].outputs_per_block,
                    key[0].k_unroll,
                    key[0].vector_width,
                    key[1],
                    key[2],
                ),
            )
        )

    def __call__(
        self,
        a: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor | None,
        config: SkinnyGemmConfig,
        *,
        a_per_token: bool,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``out[p] = a[row(p)] @ dequant(weight[ids[p]]).T`` for every
        (token, expert) pair ``p``, times ``topk_weights[p]`` when given."""
        if not can_use_skinny_fp8_moe_gemm(
            a, weight, weight_scale, topk_ids, config, a_per_token=a_per_token
        ):
            raise ValueError("inputs do not fit the skinny FP8 MoE GEMM config")
        num_experts, n, k = weight.shape
        num_pairs = topk_ids.numel()
        mul_routed_weight = topk_weights is not None
        if mul_routed_weight:
            assert topk_weights.shape == topk_ids.shape
            assert topk_weights.dtype == torch.float32
            weights = topk_weights.reshape(num_pairs)
        else:
            # Unused by the kernel; only its type must match the compiled one.
            weights = torch.empty(num_pairs, dtype=torch.float32, device=a.device)
        key = (config, a_per_token, mul_routed_weight)
        self._compile(key)
        if out is None:
            out = torch.empty((num_pairs, n), dtype=torch.bfloat16, device=a.device)
        else:
            assert out.shape == (num_pairs, n) and out.dtype == torch.bfloat16
            assert out.is_contiguous()
        self._compiled[key](
            a,
            weight.view(num_experts * n, k),
            weight_scale.view(num_experts * (n // QUANT_BLOCK), k // QUANT_BLOCK),
            topk_ids.view(num_pairs),
            weights,
            out,
            n,
            topk_ids.shape[1],
            self._stream(),
        )
        return out


shape_dynamic_skinny_fp8_moe_gemm = ShapeDynamicSkinnyFp8MoeGemm()
