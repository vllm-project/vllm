# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the W8A8 INT8 skinny GEMM kernel (wvSplitK_w8a8).

This kernel handles int8 weights with fused activation quantization:
- Activations passed as bf16/fp16, quantized to int8 inside the kernel
- Per-channel weight scale (fp16/bf16)
- Per-tensor activation scale (float32) for static quant, or None for dynamic
- Optional bias
"""

import math

import pytest
import torch

import vllm._custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.platform_utils import num_compute_units

DTYPES = [torch.bfloat16, torch.float16]
BIAS_MODES = [0, 1, 2]  # 0=no bias, 1=per-output [M], 2=per-batch [N,M]
SEEDS = [0]

# (N, K, M) test shapes: N=batch, K=inner dim, M=output features
# K must be divisible by 16, M must be divisible by YTILE (1 or 4)
NKM_FACTORS = [
    # Basic shapes
    (1, 32, 16),
    (1, 64, 64),
    (1, 128, 256),
    (1, 256, 512),
    (1, 512, 1024),
    # Typical LLM decode shapes
    (1, 4096, 4096),
    (1, 4096, 11008),
    (1, 11008, 4096),
    # Multiple batch sizes
    (2, 256, 256),
    (2, 4096, 4096),
    (3, 1024, 1024),
    (4, 4096, 4096),
    (5, 2048, 2048),
    # Extended K values
    (1, 9216, 512),
    (2, 10240, 1024),
    # Larger K (tests LDS capacity, int8 allows 2x vs fp16)
    (1, 16384, 1024),
    (2, 16384, 1024),
    (1, 32768, 1024),
]


def ref_w8a8_gemm(
    w_int8: torch.Tensor,
    a_int8: torch.Tensor,
    w_scale: torch.Tensor,
    a_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Reference implementation: dequantize and matmul.

    Args:
        w_int8: [M, K] int8 weights
        a_int8: [N, K] int8 activations
        w_scale: [M] per-channel weight scale (fp16/bf16)
        a_scale: scalar per-tensor activation scale (float32)
        bias: optional bias
        out_dtype: output dtype (fp16 or bf16)

    Returns:
        [N, M] output in out_dtype
    """
    # Dequantize to float32 for reference accuracy
    w_f32 = w_int8.float() * w_scale.float().unsqueeze(1)  # [M, K]
    a_f32 = a_int8.float() * a_scale.float()  # [N, K]

    # Matmul: [N, K] x [K, M] -> [N, M]
    out = torch.mm(a_f32, w_f32.t())

    if bias is not None:
        out = out + bias.float()

    return out.to(out_dtype)


def quantize_symmetric(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-channel int8 quantization.

    Args:
        tensor: [rows, cols] float tensor

    Returns:
        quantized: [rows, cols] int8
        scale: [rows] float32 per-channel scale
    """
    amax = tensor.abs().amax(dim=1)
    scale = amax / 127.0
    scale = scale.clamp(min=1e-10)
    quantized = (tensor / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale


def quantize_per_tensor(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-tensor int8 quantization.

    Args:
        tensor: float tensor

    Returns:
        quantized: int8 tensor
        scale: scalar float32 scale
    """
    amax = tensor.abs().max()
    scale = amax / 127.0
    scale = scale.clamp(min=1e-10)
    quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale.reshape(1)


@pytest.mark.parametrize("xnorm", [False, True])
@pytest.mark.parametrize("n,k,m", NKM_FACTORS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("bias_mode", BIAS_MODES)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@torch.inference_mode()
def test_rocm_wvsplitk_w8a8_kernel(xnorm, n, k, m, dtype, seed, bias_mode):
    """Test fused static quantization: pass bf16/fp16 activations with a_scale.

    The kernel quantizes activations to int8 internally using the provided
    per-tensor scale, then performs the int8 GEMM.
    """
    torch.manual_seed(seed)
    cu_count = num_compute_units()

    xavier = math.sqrt(2 / k) if xnorm else 1

    # Generate random data
    W_fp = (torch.rand(m, k, dtype=torch.float32, device="cuda") * 2 - 1) * xavier
    A_fp = (torch.rand(n, k, dtype=torch.float32, device="cuda") * 2 - 1) * xavier

    # Quantize weights per-channel, activations per-tensor (for reference)
    W_int8, w_scale = quantize_symmetric(W_fp)
    A_int8, a_scale = quantize_per_tensor(A_fp)

    # Convert weight scale to output dtype
    w_scale_typed = w_scale.to(dtype)

    BIAS = None
    if bias_mode == 1:
        BIAS = (torch.rand(m, dtype=dtype, device="cuda") * 2 - 1) * xavier
    elif bias_mode == 2:
        BIAS = (torch.rand(n, m, dtype=dtype, device="cuda") * 2 - 1) * xavier

    # Reference: dequantize and matmul in float32
    ref_out = ref_w8a8_gemm(W_int8, A_int8, w_scale_typed, a_scale, BIAS, dtype)

    # Kernel under test: pass bf16/fp16 activations (fused static quant)
    A_typed = A_fp.to(dtype)
    out = ops.wvSplitK_w8a8(W_int8, A_typed, w_scale_typed, a_scale, cu_count, BIAS)

    # Slightly looser tolerance: fused path quantizes from fp16/bf16 (less
    # precision than float32 quantization in the reference)
    if xnorm:
        atol = max(2e-3, torch.finfo(dtype).eps * math.sqrt(k) * 2)
        torch.testing.assert_close(out, ref_out, atol=atol, rtol=2e-2)
    else:
        atol = torch.finfo(dtype).eps * math.sqrt(k) * 2
        torch.testing.assert_close(out, ref_out, atol=atol, rtol=2e-2)


# --- Sweep tests (requires build with -DVLLM_SKINNY_GEMM_SWEEP) ---

SWEEP_HAS_OP = hasattr(torch.ops, "_rocm_C") and hasattr(
    torch.ops._rocm_C, "wvSplitK_w8a8_sweep"
)

# Subset of shapes for sweep (keep test time manageable)
SWEEP_NKM = [
    (1, 256, 512),
    (1, 4096, 4096),
    (2, 4096, 4096),
    (3, 1024, 1024),
    (4, 4096, 4096),
    (5, 2048, 2048),
]

SWEEP_PARAMS = [
    # (ytile, unrl, achunk, wvprgrp)
    (1, 1, 16, 16),  # baseline: matches production default for small M
    (1, 4, 16, 16),  # production default for large M
    (4, 1, 16, 16),  # production default for large M with N>=4
    (2, 2, 16, 16),  # mid-range ytile/unrl
    (1, 1, 8, 8),  # min achunk, min wvprgrp
    (1, 1, 32, 8),  # max achunk, min wvprgrp
    (4, 4, 8, 12),  # max ytile/unrl, different wvprgrp
    (2, 1, 32, 12),  # mixed
]


@pytest.mark.parametrize("ytile,unrl,achunk,wvprgrp", SWEEP_PARAMS)
@pytest.mark.parametrize("n,k,m", SWEEP_NKM)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@pytest.mark.skipif(not SWEEP_HAS_OP, reason="requires VLLM_SKINNY_GEMM_SWEEP build")
@torch.inference_mode()
def test_rocm_wvsplitk_w8a8_sweep(n, k, m, dtype, ytile, unrl, achunk, wvprgrp):
    """Sweep test with fused static quantization (bf16/fp16 activations)."""
    # Skip if M not divisible by ytile or K not divisible by achunk
    if m % ytile != 0:
        pytest.skip(f"M={m} not divisible by ytile={ytile}")
    if k % achunk != 0:
        pytest.skip(f"K={k} not divisible by achunk={achunk}")

    torch.manual_seed(0)
    cu_count = num_compute_units()

    xavier = math.sqrt(2 / k)

    W_fp = (torch.rand(m, k, dtype=torch.float32, device="cuda") * 2 - 1) * xavier
    A_fp = (torch.rand(n, k, dtype=torch.float32, device="cuda") * 2 - 1) * xavier

    W_int8, w_scale = quantize_symmetric(W_fp)
    A_int8, a_scale = quantize_per_tensor(A_fp)
    w_scale_typed = w_scale.to(dtype)

    ref_out = ref_w8a8_gemm(W_int8, A_int8, w_scale_typed, a_scale, None, dtype)

    # Pass bf16/fp16 activations (fused static quant)
    A_typed = A_fp.to(dtype)
    out = ops.wvSplitK_w8a8_sweep(
        W_int8,
        A_typed,
        w_scale_typed,
        a_scale,
        cu_count,
        ytile,
        unrl,
        achunk,
        wvprgrp,
    )

    atol = max(2e-3, torch.finfo(dtype).eps * math.sqrt(k) * 2)
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=2e-2)


# --- Fused quantization tests (bf16/fp16 activations passed directly) ---

# Subset of shapes for fused quant test (same constraints as skinny GEMM)
FUSED_NKM = [
    (1, 32, 16),
    (1, 64, 64),
    (1, 128, 256),
    (1, 256, 512),
    (1, 4096, 4096),
    (1, 4096, 11008),
    (2, 256, 256),
    (2, 4096, 4096),
    (3, 1024, 1024),
    (4, 4096, 4096),
    (5, 2048, 2048),
    (1, 16384, 1024),
]


@pytest.mark.parametrize("n,k,m", FUSED_NKM)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("bias_mode", BIAS_MODES)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@torch.inference_mode()
def test_rocm_wvsplitk_w8a8_fused_quant(n, k, m, dtype, seed, bias_mode):
    """Test fused activation quantization: pass fp16/bf16 activations directly.

    The kernel should quantize activations to int8 internally and produce
    the same result as the two-step path (scaled_int8_quant → wvSplitK_w8a8).
    """
    torch.manual_seed(seed)
    cu_count = num_compute_units()

    xavier = math.sqrt(2 / k)

    W_fp = (torch.rand(m, k, dtype=torch.float32, device="cuda") * 2 - 1) * xavier
    A_fp = (torch.rand(n, k, dtype=torch.float32, device="cuda") * 2 - 1) * xavier

    # Quantize weights per-channel
    W_int8, w_scale = quantize_symmetric(W_fp)
    w_scale_typed = w_scale.to(dtype)

    # Compute per-tensor activation scale (same as scaled_int8_quant would)
    A_amax = A_fp.abs().max()
    a_scale = (A_amax / 127.0).clamp(min=1e-10).reshape(1).to(torch.float32)

    # Two-step reference: quantize to int8, then run kernel
    A_int8, _ = quantize_per_tensor(A_fp)
    ref_out = ref_w8a8_gemm(W_int8, A_int8, w_scale_typed, a_scale, None, dtype)

    # Convert activations to target dtype (bf16/fp16) for fused path
    A_typed = A_fp.to(dtype)

    BIAS = None
    if bias_mode == 1:
        BIAS = (torch.rand(m, dtype=dtype, device="cuda") * 2 - 1) * xavier
    elif bias_mode == 2:
        BIAS = (torch.rand(n, m, dtype=dtype, device="cuda") * 2 - 1) * xavier

    if BIAS is not None:
        ref_out = ref_w8a8_gemm(W_int8, A_int8, w_scale_typed, a_scale, BIAS, dtype)

    # Fused path: pass fp16/bf16 activations directly
    out = ops.wvSplitK_w8a8(W_int8, A_typed, w_scale_typed, a_scale, cu_count, BIAS)

    # Slightly looser tolerance: fused path quantizes from fp16/bf16 (less
    # precision than float32 quantization in the reference), so small
    # rounding differences are expected.
    atol = max(2e-3, torch.finfo(dtype).eps * math.sqrt(k) * 2)
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=2e-2)


# --- Dynamic quantization tests (a_scale=None, bf16/fp16 activations) ---

DYNAMIC_NKM = [
    (1, 32, 16),
    (1, 64, 64),
    (1, 128, 256),
    (1, 256, 512),
    (1, 4096, 4096),
    (1, 4096, 11008),
    (2, 256, 256),
    (2, 4096, 4096),
    (3, 1024, 1024),
    (4, 4096, 4096),
    (5, 2048, 2048),
    (1, 16384, 1024),
]


def ref_dynamic_w8a8_gemm(
    w_int8: torch.Tensor,
    a_fp: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Reference for dynamic quantization: per-row absmax → int8 → matmul.

    Args:
        w_int8: [M, K] int8 weights
        a_fp: [N, K] fp16/bf16 activations (NOT pre-quantized)
        w_scale: [M] per-channel weight scale (fp16/bf16)
        bias: optional bias
        out_dtype: output dtype

    Returns:
        [N, M] output in out_dtype
    """
    a_f32 = a_fp.float()

    # Per-row dynamic quantization
    amax = a_f32.abs().amax(dim=1, keepdim=True)  # [N, 1]
    a_scale = (amax / 127.0).clamp(min=1e-10)  # [N, 1]
    a_int8 = (a_f32 / a_scale).round().clamp(-128, 127)

    # Dequantize and matmul
    w_f32 = w_int8.float() * w_scale.float().unsqueeze(1)  # [M, K]
    a_deq = a_int8 * a_scale  # [N, K]

    out = torch.mm(a_deq, w_f32.t())

    if bias is not None:
        out = out + bias.float()

    return out.to(out_dtype)


@pytest.mark.parametrize("n,k,m", DYNAMIC_NKM)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("bias_mode", BIAS_MODES)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@torch.inference_mode()
def test_rocm_wvsplitk_w8a8_dynamic_quant(n, k, m, dtype, seed, bias_mode):
    """Test fused dynamic quantization: pass bf16/fp16 activations, no a_scale.

    The kernel computes per-row absmax and quantizes internally.
    """
    torch.manual_seed(seed)
    cu_count = num_compute_units()

    xavier = math.sqrt(2 / k)

    W_fp = (torch.rand(m, k, dtype=torch.float32, device="cuda") * 2 - 1) * xavier
    A_fp = (torch.rand(n, k, dtype=torch.float32, device="cuda") * 2 - 1) * xavier

    # Quantize weights per-channel
    W_int8, w_scale = quantize_symmetric(W_fp)
    w_scale_typed = w_scale.to(dtype)

    # Convert activations to target dtype (bf16/fp16)
    A_typed = A_fp.to(dtype)

    BIAS = None
    if bias_mode == 1:
        BIAS = (torch.rand(m, dtype=dtype, device="cuda") * 2 - 1) * xavier
    elif bias_mode == 2:
        BIAS = (torch.rand(n, m, dtype=dtype, device="cuda") * 2 - 1) * xavier

    # Reference: dynamic per-row quantization then matmul
    ref_out = ref_dynamic_w8a8_gemm(W_int8, A_typed, w_scale_typed, BIAS, dtype)

    # Kernel under test: a_scale=None triggers dynamic quantization
    out = ops.wvSplitK_w8a8(W_int8, A_typed, w_scale_typed, None, cu_count, BIAS)

    # Looser tolerance: dynamic path quantizes from fp16/bf16 with
    # per-row scales computed inside the kernel.
    atol = max(2e-3, torch.finfo(dtype).eps * math.sqrt(k) * 2)
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=2e-2)
