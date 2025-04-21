# SPDX-License-Identifier: Apache-2.0
# This is a test for the aiter ops.
# It tests if the aiter ops are
# 1. correctly registered as custom ops
# 2. correctly defined the relationship between
#    implementation and fake function
# 3. can be used with torch.compile
# This file will be skipped if aiter is not installed
# and the platform is not ROCm.
#
# NOTE:
# This unit tests is by no means to check the
# correctness of the aiter ops. It only checks if the
# aiter ops are correctly registered and if torch.compile
# can be used with the aiter ops.
# The correctness of the aiter ops is tested in the
# https://github.com/ROCm/aiter

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
    input_tensor = torch.randn(64, 32, dtype=torch.float16, device='cuda')
    weight_tensor = torch.randn(16, 32, dtype=torch.float16, device='cuda')

    # Define a function that uses the op
    def gemm_fn(x, w):
        return aiter_ops.rocm_aiter_tuned_gemm(x, w)

    # Verify the op's fake implementation
    torch.library.opcheck(torch.ops.vllm.rocm_aiter_tuned_gemm,
                          (input_tensor, weight_tensor),
                          test_utils=("test_schema", "test_faketensor"))

    # Compile the function with appropriate settings based on
    # vllm/compilation/wrapper.py
    compiled_fn = torch.compile(gemm_fn,
                                fullgraph=True,
                                backend="inductor",
                                mode="reduce-overhead",
                                dynamic=False)

    # Run both compiled (V1 graph mode) and uncompiled versions (V1 eager mode)
    result_original = gemm_fn(input_tensor, weight_tensor)
    result_compiled = compiled_fn(input_tensor, weight_tensor)

    # Verify results match
    assert torch.allclose(result_original, result_compiled)


def test_rocm_aiter_tuned_gemm_torch_compile_fp8_compatibility():

    input_tensor = torch.randn(64, 32, dtype=torch.float16, device='cuda')
    weight_tensor = torch.randn(16, 32, dtype=torch.float16, device='cuda')

    input_fp8 = input_tensor.to(current_platform.fp8_dtype())
    weight_fp8 = weight_tensor.to(current_platform.fp8_dtype())

    scale_a = torch.tensor(10.0, device='cuda')
    scale_b = torch.tensor(0.5, device='cuda')

    # Define a function that uses the op with FP8 and scales
    def gemm_fp8_fn(x, w, scale_a, scale_b):
        return aiter_ops.rocm_aiter_tuned_gemm(x,
                                               w,
                                               out_dtype=torch.float16,
                                               scale_a=scale_a,
                                               scale_b=scale_b)

    # Verify the op's fake implementation with FP8 inputs
    # Disable test_schema as fp8 datatype is not supported by
    # torch.library.opcheck
    # Related error:
    #      OpCheckError: opcheck(op, ...): test_schema failed with
    #      "mul_cuda" not implemented for 'Float8_e4m3fnuz'
    torch.library.opcheck(torch.ops.vllm.rocm_aiter_tuned_gemm,
                          (input_fp8, weight_fp8),
                          kwargs={
                              "out_dtype": torch.float16,
                              "scale_a": scale_a,
                              "scale_b": scale_b
                          },
                          test_utils=("test_faketensor"))

    # Compile the function with appropriate settings based on
    # vllm/compilation/wrapper.py
    compiled_fp8_fn = torch.compile(gemm_fp8_fn,
                                    fullgraph=True,
                                    backend="inductor",
                                    mode="reduce-overhead",
                                    dynamic=False)

    # Run both compiled (V1 graph mode) and uncompiled versions (V1 eager mode)
    result_original = gemm_fp8_fn(input_fp8, weight_fp8, scale_a, scale_b)
    result_compiled = compiled_fp8_fn(input_fp8, weight_fp8, scale_a, scale_b)

    # Verify results match and have correct properties
    assert torch.allclose(result_original, result_compiled)
    assert result_original.dtype == torch.float16
    assert result_compiled.dtype == torch.float16
    assert result_original.shape == (64, 16)
    assert result_compiled.shape == (64, 16)

    # Get unscaled result
    unscaled_result = aiter_ops.rocm_aiter_tuned_gemm(
        input_fp8.to(torch.float16),
        weight_fp8.to(torch.float16),
        out_dtype=torch.float16)

    # Verify that scaling was applied correctly
    # The scaled result should be approximately equal to the
    # unscaled result multiplied by the scales
    expected_scaled = unscaled_result * (scale_a * scale_b)
    assert torch.allclose(result_original,
                          expected_scaled,
                          rtol=1e-2,
                          atol=1e-2)

    # Verify that scaled and unscaled results are different
    assert not torch.allclose(
        result_original, unscaled_result, rtol=1e-2, atol=1e-2)
