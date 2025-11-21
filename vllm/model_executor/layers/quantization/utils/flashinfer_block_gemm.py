# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
FlashInfer FP8 Block-Scale GEMM wrapper for vLLM.

This module provides a thin wrapper around FlashInfer's FP8 block-scale GEMM
implementation, which uses TensorRT-LLM's optimized kernels for NVIDIA Hopper (SM90+).
"""

import torch


def flashinfer_block_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    scales_a: torch.Tensor | None,
    scales_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Wrapper for FlashInfer's FP8 block-scale GEMM.
    
    Computes: output = (input @ weight.T) with per-block scaling for quantization.
    
    Supports three modes:
    1. BF16 + BF16 → BF16: Both inputs BF16, internal quantization (scales_a=None, scales_b used internally)
    2. BF16 + FP8 → BF16: Weight-only quantization (scales_a=None, scales_b for weight)
    3. FP8 + FP8 → BF16: W8A8 full quantization (scales_a for input, scales_b for weight)
    
    Args:
        input: Input tensor (M, K) - BF16 or FP8 e4m3
        weight: Weight tensor (N, K) - BF16 or FP8 e4m3
        scales_a: Input scales (M, K//block_k) or None - FP32
                  None: input is BF16 (will be quantized internally for BF16+BF16 or left as-is for BF16+FP8)
                  Provided: input is pre-quantized FP8 (W8A8 mode)
        scales_b: Weight scales (N//block_n, K//block_k) - FP32
        out_dtype: Output dtype (typically torch.bfloat16)
        
    Returns:
        output: Result tensor (M, N) in out_dtype
        
    Note:
        - Requires SM90+ GPU (NVIDIA Hopper)
        - Uses TensorRT-LLM's optimized CUTLASS kernels via FlashInfer
        - For M < 32, automatically uses SwapAB kernel optimization
    """
    from flashinfer.gemm import fp8_blockscale_gemm_swapab
    
    return fp8_blockscale_gemm_swapab(
        input=input,
        weight=weight,
        input_scale=scales_a,
        weight_scale=scales_b,
        out_dtype=out_dtype,
    )

