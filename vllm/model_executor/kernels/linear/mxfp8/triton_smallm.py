# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton MXFP8 GEMM optimized for small M (decode-batch regime).

The FlashInfer CUTLASS `mm_mxfp8` kernel is shape-optimal at M >= 128: it
processes a 128-row tile regardless of caller M, so at M=32 we pay for
128 rows of compute and discard 96 of them. This module provides a
Triton kernel that handles M < 128 natively without the padding waste.

Layout conventions (matches mxfp8_e4m3_quantize with is_sf_swizzled_layout=False):
    input_fp8:    [M, K] torch.float8_e4m3fn
    input_scale:  [M, K // 32] torch.uint8 (e8m0 biased exponent)
    weight_fp8:   [N, K] torch.float8_e4m3fn
    weight_scale: [N, K // 32] torch.uint8 (e8m0 biased exponent, un-swizzled)

Block scale interpretation:
    real_value = fp8_value * 2 ** (scale_uint8 - 127)
    so output[m, n] = sum over k_block of dot(in_fp8[m, blk], w_fp8[n, blk])
                      * 2^(in_scale[m, k_block] + w_scale[n, k_block] - 254)
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:  # triton always present in vLLM, but guard anyway
    _HAS_TRITON = False

MXFP8_BLOCK_SIZE = 32


if _HAS_TRITON:
    @triton.jit
    def _mxfp8_smallm_gemm_kernel(
        # Pointers
        input_ptr,         # [M, K] fp8_e4m3
        input_scale_ptr,   # [M, K/32] uint8
        weight_ptr,        # [N, K] fp8_e4m3
        weight_scale_ptr,  # [N, K/32] uint8
        output_ptr,        # [M, N] bf16
        # Sizes (runtime)
        M, N, K,
        # Strides
        stride_im, stride_ik,
        stride_ism, stride_isk,
        stride_wn, stride_wk,
        stride_wsn, stride_wsk,
        stride_om, stride_on,
        # Block shapes (compile-time)
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        SCALE_BLOCK: tl.constexpr,  # 32 — must match MXFP8_BLOCK_SIZE
    ):
        """One CTA computes one [BLOCK_M, BLOCK_N] tile of the output."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, SCALE_BLOCK)  # 32

        m_mask = offs_m < M
        n_mask = offs_n < N

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        num_k_blocks = K // SCALE_BLOCK

        for k_block in range(0, num_k_blocks):
            k_abs = k_block * SCALE_BLOCK + offs_k  # 32 element offsets

            # Load input tile [BLOCK_M, SCALE_BLOCK] in fp8
            in_block = tl.load(
                input_ptr
                + offs_m[:, None] * stride_im
                + k_abs[None, :] * stride_ik,
                mask=m_mask[:, None],
                other=0.0,
            ).to(tl.bfloat16)

            # Load weight tile [BLOCK_N, SCALE_BLOCK] in fp8
            w_block = tl.load(
                weight_ptr
                + offs_n[:, None] * stride_wn
                + k_abs[None, :] * stride_wk,
                mask=n_mask[:, None],
                other=0.0,
            ).to(tl.bfloat16)

            # Block dot product: [BLOCK_M, BLOCK_N] in fp32
            block_dot = tl.dot(in_block, w_block.T).to(tl.float32)

            # Load per-block scales (uint8 e8m0 biased exponents)
            in_scale_u = tl.load(
                input_scale_ptr + offs_m * stride_ism + k_block * stride_isk,
                mask=m_mask,
                other=0,
            ).to(tl.float32)
            w_scale_u = tl.load(
                weight_scale_ptr + offs_n * stride_wsn + k_block * stride_wsk,
                mask=n_mask,
                other=0,
            ).to(tl.float32)

            # Combined scale: 2^((in_scale - 127) + (w_scale - 127))
            #               = 2^(in_scale + w_scale - 254)
            # Broadcast to [BLOCK_M, BLOCK_N]
            log_scale = in_scale_u[:, None] + w_scale_u[None, :] - 254.0
            scale = tl.exp2(log_scale)

            acc += block_dot * scale

        # Write output [BLOCK_M, BLOCK_N] as bf16
        out_ptrs = (
            output_ptr
            + offs_m[:, None] * stride_om
            + offs_n[None, :] * stride_on
        )
        tl.store(
            out_ptrs,
            acc.to(tl.bfloat16),
            mask=m_mask[:, None] & n_mask[None, :],
        )


def mxfp8_smallm_gemm(
    input_fp8: torch.Tensor,
    input_scale: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Triton MXFP8 GEMM. Use only when M < 128 — for M >= 128, defer to
    FlashInfer's mm_mxfp8 which is faster at large batch.

    Args:
        input_fp8:    [M, K] torch.float8_e4m3fn
        input_scale:  [M, K // 32] torch.uint8 (un-swizzled, row-major)
        weight_fp8:   [N, K] torch.float8_e4m3fn
        weight_scale: [N, K // 32] torch.uint8 (un-swizzled, row-major)

    Returns:
        output: [M, N] torch.bfloat16
    """
    if not _HAS_TRITON:
        raise RuntimeError("triton not available for mxfp8_smallm_gemm")

    M, K = input_fp8.shape
    N, _Kw = weight_fp8.shape
    assert K == _Kw, f"K mismatch: input K={K} vs weight K={_Kw}"
    assert K % MXFP8_BLOCK_SIZE == 0, (
        f"K must be divisible by {MXFP8_BLOCK_SIZE}, got {K}"
    )
    assert input_scale.shape == (M, K // MXFP8_BLOCK_SIZE), (
        f"input_scale shape {tuple(input_scale.shape)} != ({M},{K//MXFP8_BLOCK_SIZE})"
    )
    assert weight_scale.shape == (N, K // MXFP8_BLOCK_SIZE), (
        f"weight_scale shape {tuple(weight_scale.shape)} != ({N},{K//MXFP8_BLOCK_SIZE})"
    )

    output = torch.empty(M, N, dtype=torch.bfloat16, device=input_fp8.device)

    # BLOCK_M: smallest power of 2 >= M, capped at 128
    BLOCK_M = max(16, 1 << (M - 1).bit_length()) if M > 0 else 16
    BLOCK_M = min(BLOCK_M, 128)
    BLOCK_N = 128
    SCALE_BLOCK = MXFP8_BLOCK_SIZE

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Ensure contiguous; inputs/scales are normally contiguous already.
    def _c(t: torch.Tensor) -> torch.Tensor:
        return t if t.is_contiguous() else t.contiguous()

    input_fp8_c = _c(input_fp8)
    weight_fp8_c = _c(weight_fp8)
    input_scale_c = _c(input_scale)
    weight_scale_c = _c(weight_scale)

    _mxfp8_smallm_gemm_kernel[grid](
        input_fp8_c, input_scale_c,
        weight_fp8_c, weight_scale_c,
        output,
        M, N, K,
        input_fp8_c.stride(0), input_fp8_c.stride(1),
        input_scale_c.stride(0), input_scale_c.stride(1),
        weight_fp8_c.stride(0), weight_fp8_c.stride(1),
        weight_scale_c.stride(0), weight_scale_c.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        SCALE_BLOCK=SCALE_BLOCK,
        num_warps=8,
        num_stages=4,
    )

    return output


__all__ = ["mxfp8_smallm_gemm", "MXFP8_BLOCK_SIZE"]
