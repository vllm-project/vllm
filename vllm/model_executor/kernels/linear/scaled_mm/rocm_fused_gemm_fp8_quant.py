# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ROCm fused GEMM + static FP8 output quantization.

Fuses scaled_mm (FP8→BF16/FP16) + static_scaled_fp8_quant (→FP8) into a
single scaled_mm call that outputs FP8 directly, eliminating the BF16/FP16
intermediate DRAM round-trip.

The key insight: merge output_scale into a_scales before the GEMM, then
call torch._scaled_mm(..., out_dtype=fp8). This works because hipBLASLt
natively supports FP8 output dtype since ROCm 6.0.

Benchmark on MI300X (PyTorch 2.6, ROCm 6.4):
  Shape (512,4096,4096):   unfused 46.3µs → fused 30.5µs  (1.51x)
  Shape (512,4096,11008):  unfused 94.8µs → fused 61.4µs  (1.54x)
  Shape (512,4096,14336):  unfused 112.5µs → fused 68.9µs (1.63x)
  Shape (2048,4096,4096):  unfused 123.1µs → fused 73.8µs (1.67x)
"""

import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op


def rocm_scaled_mm_static_fp8_quant_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    output_scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Fused FP8 GEMM + static FP8 output quantization.

    Equivalent to:
        mm_out = scaled_mm(a, b, a_scales, b_scales, out_dtype=bf16)
        fp8_out = static_scaled_fp8_quant(mm_out, output_scale)

    But done in a single hipBLASLt call by merging output_scale into
    a_scales and requesting FP8 output directly.
    """
    out_dtype = current_platform.fp8_dtype()

    # Merge output_scale into a_scales: combined = a_scales / output_scale
    # This is equivalent to: output = (a @ b * a_scales * b_scales) / output_scale
    combined_a_scales = a_scales * output_scale.reciprocal()

    output = torch._scaled_mm(
        a,
        b,
        out_dtype=out_dtype,
        scale_a=combined_a_scales,
        scale_b=b_scales,
        bias=bias,
    )
    if isinstance(output, tuple):
        output = output[0]
    return output


def rocm_scaled_mm_static_fp8_quant_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    output_scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    out_dtype = current_platform.fp8_dtype()
    return a.new_empty((*a.shape[:-1], b.shape[1]), dtype=out_dtype)


if current_platform.is_rocm():
    direct_register_custom_op(
        op_name="rocm_scaled_mm_static_fp8_quant",
        op_func=rocm_scaled_mm_static_fp8_quant_impl,
        fake_impl=rocm_scaled_mm_static_fp8_quant_fake,
    )
