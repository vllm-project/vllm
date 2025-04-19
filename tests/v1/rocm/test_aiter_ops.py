# SPDX-License-Identifier: Apache-2.0
# This is a test for the aiter ops.
# It tests if the aiter ops are
# 1. correctly registered as custom ops
# 2. correctly defined the relationship between
#    implementation and fake function
# 3. can be used with torch.compile
# This file will be skipped if aiter is not installed
# and the platform is not ROCm.

import importlib.util
import pytest
import torch
from vllm._aiter_ops import aiter_ops
from vllm.platforms import current_platform

# Check if aiter package is installed
aiter_available = importlib.util.find_spec("aiter") is not None

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and aiter_available),
    reason="AITER ops are only available on ROCm with aiter package installed")

def test_rocm_aiter_tuned_gemm_custom_op_registration():
    """Test that the custom op is correctly registered."""
    # Check if the op exists in torch.ops.vllm
    assert hasattr(torch.ops.vllm, 'rocm_aiter_tuned_gemm')
    
    # Check if the op is callable
    assert callable(torch.ops.vllm.rocm_aiter_tuned_gemm)

def test_rocm_aiter_tuned_gemm_torch_compile_compatibility():
    """Test that the op can be used with torch.compile."""
    # Create test tensors
    input_tensor = torch.randn(4, 3, dtype=torch.float16, device='cuda')
    weight_tensor = torch.randn(5, 3, dtype=torch.float16, device='cuda')
    
    # Define a function that uses the op
    def gemm_fn(x, w):
        return aiter_ops.rocm_aiter_tuned_gemm(x, w)
    
    # Compile the function
    compiled_fn = torch.compile(gemm_fn)
    
    # Run both compiled and uncompiled versions
    result_original = gemm_fn(input_tensor, weight_tensor)
    result_compiled = compiled_fn(input_tensor, weight_tensor)
    
    # Verify results match
    assert torch.allclose(result_original, result_compiled)

def test_rocm_aiter_tuned_gemm_torch_compile_fp8_compatibility():
    """Test that the op can be used with torch.compile for FP8 (E4M3FNuz) matrices."""
    # Create test tensors in FP8 format
    input_tensor = torch.randn(4, 3, dtype=torch.float16, device='cuda')
    weight_tensor = torch.randn(5, 3, dtype=torch.float16, device='cuda')
    
    # Convert to FP8 (E4M3FNuz)
    input_fp8 = input_tensor.to(torch.float8_e4m3fnuz)
    weight_fp8 = weight_tensor.to(torch.float8_e4m3fnuz)
    
    # Create per-tensor scales
    scale_a = torch.tensor(5.0, device='cuda')  # Scale for input
    scale_b = torch.tensor(0.5, device='cuda')  # Scale for weight
    
    # Define a function that uses the op with FP8 and scales
    def gemm_fp8_fn(x, w, scale_a, scale_b):
        return aiter_ops.rocm_aiter_tuned_gemm(
            x, w, 
            out_dtype=torch.float16,
            scale_a=scale_a,
            scale_b=scale_b
        )
    
    # Compile the function
    compiled_fp8_fn = torch.compile(gemm_fp8_fn)
    
    # Run both compiled and uncompiled versions
    result_original = gemm_fp8_fn(input_fp8, weight_fp8, scale_a, scale_b)
    result_compiled = compiled_fp8_fn(input_fp8, weight_fp8, scale_a, scale_b)
    
    # Verify results match and have correct properties
    assert torch.allclose(result_original, result_compiled)
    assert result_original.dtype == torch.float16
    assert result_compiled.dtype == torch.float16
    assert result_original.shape == (4, 5)
    assert result_compiled.shape == (4, 5)
    
    # Verify that scaling was applied correctly by comparing with unscaled version
    unscaled_result = aiter_ops.rocm_aiter_tuned_gemm(
        input_fp8, weight_fp8, out_dtype=torch.float16)
    # Results should be different due to scaling
    assert not torch.allclose(result_original, unscaled_result)  

