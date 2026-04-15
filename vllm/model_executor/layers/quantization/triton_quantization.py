# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for fused activation + quantization.

Generated and optimized by KernelAgent
(https://github.com/meta-pytorch/KernelAgent), an autonomous GPU kernel
synthesis system that uses LLM-assisted generation with runtime
verification on NVIDIA Blackwell (GB200) GPUs.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def fused_silu_mul_per_token_quant_kernel(
    output_ptr,  # [num_tokens, hidden_size]  fp8
    scales_ptr,  # [num_tokens, 1]            fp32
    input_ptr,  # [num_tokens, hidden_size * 2]  bf16/fp16
    scale_ub_ptr,  # Optional: single float upper bound for scale
    hidden_size: tl.int64,
    stride_input_m: tl.int64,
    stride_output_m: tl.int64,
    stride_scale_m: tl.int64,
    FP8_MAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_ITERS: tl.constexpr,
    HAS_SCALE_UB: tl.constexpr,
):
    """Fused SiLU+Mul + per-token dynamic FP8 quantization.

    Each program handles one token (row). Two-pass:
    1. Compute silu(gate)*up per chunk, track row-wise amax
    2. Re-compute silu(gate)*up, quantize with the per-token scale
    """
    row_idx = tl.program_id(0)
    row_input_ptr = input_ptr + row_idx * stride_input_m
    row_output_ptr = output_ptr + row_idx * stride_output_m

    ub_val = tl.load(scale_ub_ptr) if HAS_SCALE_UB else 1e10

    # --- Pass 1: compute row-wise amax ---
    row_max = tl.zeros((), dtype=tl.float32)
    for i in range(0, NUM_ITERS):
        offsets = (i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
        mask = offsets < hidden_size
        gate = tl.load(row_input_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        up = tl.load(
            row_input_ptr + offsets + hidden_size, mask=mask, other=0.0
        ).to(tl.float32)
        silu_mul = gate * tl.sigmoid(gate) * up
        chunk_max = tl.max(tl.abs(silu_mul), axis=0)
        row_max = tl.maximum(row_max, chunk_max)

    # Scale convention: matches vLLM C++ dynamic_per_token_scaled_fp8_quant
    # scale = amax(row), clamped to min_scaling_factor = 1/(FP8_MAX * 512)
    # output = clamp(input / scale * FP8_MAX)
    MIN_SCALING_FACTOR: tl.constexpr = 1.0 / (FP8_MAX * 512.0)
    scale = tl.maximum(tl.minimum(row_max, ub_val), MIN_SCALING_FACTOR)
    tl.store(scales_ptr + row_idx * stride_scale_m, scale)

    # --- Pass 2: quantize and store ---
    inv_scale = FP8_MAX / scale
    for i in range(0, NUM_ITERS):
        offsets = (i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
        mask = offsets < hidden_size
        gate = tl.load(row_input_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        up = tl.load(
            row_input_ptr + offsets + hidden_size, mask=mask, other=0.0
        ).to(tl.float32)
        silu_mul = gate * tl.sigmoid(gate) * up
        quantized = tl.clamp(silu_mul * inv_scale, -FP8_MAX, FP8_MAX).to(
            output_ptr.dtype.element_ty
        )
        tl.store(row_output_ptr + offsets, quantized, mask=mask)


def fused_silu_mul_per_token_quant(
    input: torch.Tensor,
    quant_dtype: torch.dtype,
    scale_ub: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused SiLU+Mul + per-token dynamic FP8 quantization.

    Args:
        input: [num_tokens, hidden_size * 2] bf16/fp16 (gate || up layout)
        quant_dtype: output quantization dtype (e.g., torch.float8_e4m3fn)
        scale_ub: optional [1] float32 upper bound for scales

    Returns:
        output: [num_tokens, hidden_size] quant_dtype
        scales: [num_tokens, 1] float32 — one scale per token
    """
    num_tokens = input.size(0)
    hidden_size = input.size(1) // 2

    output = torch.empty(
        (num_tokens, hidden_size), device=input.device, dtype=quant_dtype
    )
    scales = torch.empty(
        (num_tokens, 1), device=input.device, dtype=torch.float32
    )

    FP8_MAX = torch.finfo(quant_dtype).max
    BLOCK_SIZE = min(triton.next_power_of_2(hidden_size), 8192)
    NUM_ITERS = triton.cdiv(hidden_size, BLOCK_SIZE)

    grid = (num_tokens,)
    fused_silu_mul_per_token_quant_kernel[grid](
        output,
        scales,
        input,
        scale_ub,
        hidden_size,
        input.stride(0),
        output.stride(0),
        scales.stride(0),
        FP8_MAX=FP8_MAX,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_ITERS=NUM_ITERS,
        HAS_SCALE_UB=(scale_ub is not None),
    )

    return output, scales
