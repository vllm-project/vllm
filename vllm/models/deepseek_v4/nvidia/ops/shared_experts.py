# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused SwiGLU(-with-clamp) activation + row-major block-FP8 quantization
for DeepSeek V4 shared-expert MLP.

DSpark's shared-expert MLP forward is gate_up GEMM -> SiluAndMul(WithClamp)
-> down GEMM, where down_proj is FP8 block-scaled and (like every
BlockScaledMMLinearKernel) quantizes its BF16 input via a standalone
``per_token_group_quant_fp8`` kernel launch before the GEMM proper. That
quantize launch reads back the exact BF16 tensor the activation kernel just
wrote, immediately after writing it. This kernel fuses the two: the
clamp+silu+mul epilogue computes the activated value and quantizes it to FP8
in registers, skipping the intermediate BF16 round-trip through global
memory and the separate kernel launch.

This intentionally does not reuse
``vllm.model_executor.layers.quantization.utils.fp8_utils``'s existing
``silu_mul_quant_fp8_packed_triton`` / ``silu_mul_per_token_group_quant_fp8_colmajor``
kernels: those are wired into the routed-expert DeepGEMM MoE path and
produce column-major or UE8M0-packed scales for that consumer, and assume
token counts aligned to a fixed ``BLOCK_M`` (8). DSpark's decode batches are
small (single-digit token counts per proposal) and feed
``w8a8_triton_block_scaled_mm`` via ``down_proj``, which expects plain
row-major ``(tokens, K/128)`` FP32 scales (see the o_proj fusion in
``fp8_einsum.py`` for the same contract, validated against captured
fixtures). The clamp/silu/mul epilogue math below mirrors
``_silu_mul_per_token_group_quant_fp8_colmajor`` (narrow-then-widen at each
step to match the original C++ ``silu_and_mul``/``silu_and_mul_with_clamp``
kernels bit-for-bit) so this is a storage-layout variant, not a new
numerical recipe.
"""

import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    is_deep_gemm_e8m0_used,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_tokens"])
def _dspark_silu_mul_clamp_fp8_quant_kernel(
    x_ptr,
    out_fp8_ptr,
    out_scale_ptr,
    num_tokens,
    d: tl.constexpr,
    x_stride_token: tl.constexpr,
    x_stride_hidden: tl.constexpr,
    out_stride_token: tl.constexpr,
    out_stride_hidden: tl.constexpr,
    out_scale_stride_token: tl.constexpr,
    out_scale_stride_block: tl.constexpr,
    swiglu_limit: tl.constexpr,
    alpha: tl.constexpr,
    beta: tl.constexpr,
    has_clamp: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    use_ue8m0: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
) -> None:
    token_block = tl.program_id(0)
    out_block = tl.program_id(1)

    token_offsets = token_block * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    out_offsets = out_block * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    token_mask = token_offsets < num_tokens
    mask2d = token_mask[:, None]

    gate_ptrs = (
        x_ptr
        + token_offsets[:, None] * x_stride_token
        + out_offsets[None, :] * x_stride_hidden
    )
    up_ptrs = gate_ptrs + d * x_stride_hidden

    gate = tl.load(gate_ptrs, mask=mask2d, other=0.0)
    up = tl.load(up_ptrs, mask=mask2d, other=0.0)

    # Mirrors _silu_mul_per_token_group_quant_fp8_colmajor: clamp in fp32
    # then narrow back to the input dtype, so HAS_CLAMP True/False share the
    # same downstream multiplication path and match the original C++
    # silu_and_mul[_with_clamp] kernels' rounding bit-for-bit.
    if has_clamp:
        gate = tl.minimum(gate.to(tl.float32), swiglu_limit).to(x_ptr.dtype.element_ty)
        up = tl.clamp(up.to(tl.float32), -swiglu_limit, swiglu_limit).to(
            x_ptr.dtype.element_ty
        )

    gate_f32 = gate.to(tl.float32)
    glu = (gate_f32 / (1.0 + tl.exp(-gate_f32 * alpha))).to(x_ptr.dtype.element_ty)
    up_biased = (up.to(tl.float32) + beta).to(x_ptr.dtype.element_ty)
    y = (glu * up_biased).to(tl.float32)

    row_absmax = tl.maximum(tl.max(tl.abs(y), axis=1), eps)
    scale_raw = row_absmax * (1.0 / fp8_max)
    scale = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
    quant = tl.clamp(y / scale[:, None], fp8_min, fp8_max).to(tl.float8e4nv)

    tl.store(
        out_fp8_ptr
        + token_offsets[:, None] * out_stride_token
        + out_offsets[None, :] * out_stride_hidden,
        quant,
        mask=mask2d,
    )
    tl.store(
        out_scale_ptr
        + token_offsets * out_scale_stride_token
        + out_block * out_scale_stride_block,
        scale,
        mask=token_mask,
    )


def dspark_silu_mul_clamp_fp8_quant(
    gate_up: torch.Tensor,
    swiglu_limit: float | None,
    alpha: float,
    beta: float,
    out_fp8: torch.Tensor,
    out_scale: torch.Tensor,
) -> None:
    """``out = quantize_fp8_block128(swiglu_with_clamp(gate_up))``, fused.

    ``gate_up`` is ``(num_tokens, 2*d)`` BF16 (gate_up_proj's GEMM output).
    ``out_fp8``/``out_scale`` are ``(num_tokens, d)``/``(num_tokens, d//128)``,
    ready to pass directly as ``A``/``As`` to ``w8a8_triton_block_scaled_mm``
    for down_proj's GEMM.
    """
    num_tokens, n2 = gate_up.shape
    d = n2 // 2
    assert d % 128 == 0
    assert out_fp8.shape == (num_tokens, d)
    assert out_scale.shape == (num_tokens, d // 128)
    assert gate_up.is_contiguous()

    if num_tokens == 0:
        return

    fp8_min, fp8_max = get_fp8_min_max()
    block_tokens = 16
    block_out = 128
    grid = (triton.cdiv(num_tokens, block_tokens), d // block_out)
    _dspark_silu_mul_clamp_fp8_quant_kernel[grid](
        gate_up,
        out_fp8,
        out_scale,
        num_tokens,
        d,
        gate_up.stride(0),
        gate_up.stride(1),
        out_fp8.stride(0),
        out_fp8.stride(1),
        out_scale.stride(0),
        out_scale.stride(1),
        swiglu_limit=float(swiglu_limit) if swiglu_limit is not None else 0.0,
        alpha=float(alpha),
        beta=float(beta),
        has_clamp=swiglu_limit is not None,
        fp8_min=float(fp8_min),
        fp8_max=float(fp8_max),
        use_ue8m0=is_deep_gemm_e8m0_used(),
        eps=1e-10,
        BLOCK_TOKENS=block_tokens,
        BLOCK_OUT=block_out,
        num_warps=4,
        num_stages=2,
    )
