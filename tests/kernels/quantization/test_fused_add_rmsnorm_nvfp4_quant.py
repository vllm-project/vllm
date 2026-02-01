# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for fused Add + RMSNorm + NVFP4 quantization kernel.
"""

import pytest
import torch

from tests.kernels.quantization.nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    dequantize_nvfp4_to_dtype,
)
from vllm._custom_ops import scaled_fp4_quant
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Nvfp4 Requires compute capability of 10 or above.",
        allow_module_level=True,
    )

FP4_DTYPE = torch.uint8
FP8_DTYPE = current_platform.fp8_dtype()

DTYPES = [torch.float16, torch.bfloat16]
SHAPES = [(128, 256), (128, 128), (256, 256), (256, 128)]
EPSILON = 1e-6


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@torch.inference_mode()
def test_fused_add_rms_norm_nvfp4_quant(
    default_vllm_config,
    dtype: torch.dtype,
    shape: tuple[int, int],
) -> None:
    """Test fused Add + RMSNorm + NVFP4 quantization kernel."""
    set_random_seed(42)
    device = "cuda:0"
    torch.set_default_device(device)

    num_tokens, hidden_size = shape
    x = torch.randn(shape, dtype=dtype)
    residual = torch.randn(shape, dtype=dtype)
    weight = torch.randn(hidden_size, dtype=dtype)

    # Reference: x + residual -> RMSNorm -> scaled_fp4_quant
    ref_residual = x + residual
    rms_norm = RMSNorm(hidden_size, EPSILON).to(dtype=dtype, device=device)
    rms_norm.weight.data.copy_(weight)
    ref_output = rms_norm.forward_native(ref_residual)

    ref_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.abs(
        ref_output
    ).max().to(torch.float32)
    ref_output_quant, ref_block_scale = scaled_fp4_quant(ref_output, ref_global_scale)

    # Fused op: fused_add_rms_norm_nvfp4_quant
    fused_output_quant = torch.empty_like(ref_output_quant)
    fused_block_scale = torch.empty_like(ref_block_scale)
    fused_residual = residual.clone()  # Will be updated in-place
    torch.ops._C.fused_add_rms_norm_nvfp4_quant(
        fused_output_quant,
        fused_block_scale,
        x,
        fused_residual,
        weight,
        ref_global_scale,
        EPSILON,
    )

    # Check residual is updated in-place
    torch.testing.assert_close(fused_residual, ref_residual, atol=1e-5, rtol=1e-5)

    # Check dtype
    assert ref_output_quant.dtype == FP4_DTYPE
    assert fused_output_quant.dtype == FP4_DTYPE
    assert ref_output_quant.shape == fused_output_quant.shape

    assert ref_block_scale.dtype == FP8_DTYPE
    assert fused_block_scale.dtype == FP8_DTYPE
    assert ref_block_scale.shape == fused_block_scale.shape

    # Check dequantized output
    ref_output_dequant = dequantize_nvfp4_to_dtype(
        ref_output_quant, ref_block_scale, ref_global_scale, dtype, device
    )
    fused_output_dequant = dequantize_nvfp4_to_dtype(
        fused_output_quant, fused_block_scale, ref_global_scale, dtype, device
    )

    atol, rtol = 3e-1, 3e-1
    torch.testing.assert_close(
        ref_output_dequant, fused_output_dequant, atol=atol, rtol=rtol
    )


@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_fused_add_rms_norm_nvfp4_quant_large(
    default_vllm_config,
    dtype: torch.dtype,
) -> None:
    """Test fused Add + RMSNorm + NVFP4 quantization kernel with larger shapes."""
    set_random_seed(42)
    device = "cuda:0"
    torch.set_default_device(device)

    # Test with larger hidden sizes (typical model dimensions)
    shape = (64, 4096)
    num_tokens, hidden_size = shape
    x = torch.randn(shape, dtype=dtype)
    residual = torch.randn(shape, dtype=dtype)
    weight = torch.randn(hidden_size, dtype=dtype)

    # Reference: x + residual -> RMSNorm -> scaled_fp4_quant
    ref_residual = x + residual
    rms_norm = RMSNorm(hidden_size, EPSILON).to(dtype=dtype, device=device)
    rms_norm.weight.data.copy_(weight)
    ref_output = rms_norm.forward_native(ref_residual)

    ref_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.abs(
        ref_output
    ).max().to(torch.float32)
    ref_output_quant, ref_block_scale = scaled_fp4_quant(ref_output, ref_global_scale)

    # Fused op: fused_add_rms_norm_nvfp4_quant
    fused_output_quant = torch.empty_like(ref_output_quant)
    fused_block_scale = torch.empty_like(ref_block_scale)
    fused_residual = residual.clone()
    torch.ops._C.fused_add_rms_norm_nvfp4_quant(
        fused_output_quant,
        fused_block_scale,
        x,
        fused_residual,
        weight,
        ref_global_scale,
        EPSILON,
    )

    # Check residual is updated in-place
    torch.testing.assert_close(fused_residual, ref_residual, atol=1e-5, rtol=1e-5)

    # Check dequantized output
    ref_output_dequant = dequantize_nvfp4_to_dtype(
        ref_output_quant, ref_block_scale, ref_global_scale, dtype, device
    )
    fused_output_dequant = dequantize_nvfp4_to_dtype(
        fused_output_quant, fused_block_scale, ref_global_scale, dtype, device
    )

    atol, rtol = 3e-1, 3e-1
    torch.testing.assert_close(
        ref_output_dequant, fused_output_dequant, atol=atol, rtol=rtol
    )
