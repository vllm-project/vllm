# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER grouped FP8 quantization tests for ROCm.

This file checks that the ROCm AITER grouped FP8 quant op:
1. exposes a correct fake implementation
2. works through torch.compile in cudagraph mode
3. preserves expected shapes and dtypes across representative inputs
"""

import importlib.util

import pytest
import torch

# this import statement is needed to ensure the ops are registered
from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms import current_platform

# Check if aiter package is installed
aiter_available = importlib.util.find_spec("aiter") is not None

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and aiter_available),
    reason="AITER ops are only available on ROCm with aiter package installed",
)


def test_rocm_aiter_group_fp8_quant_fake_implementation():
    """Test that the fake implementation is correctly defined."""
    m = 128
    n = 4096
    group_size = 128

    input_tensor = torch.randn((m, n), dtype=torch.bfloat16, device="cuda")

    # Verify that the fake function returns tensors with the right metadata.
    torch.library.opcheck(
        torch.ops.vllm.rocm_aiter_group_fp8_quant,
        (input_tensor, group_size),
        test_utils=("test_faketensor",),
    )


def test_rocm_aiter_group_fp8_quant_torch_compile_with_cudagraph():
    """Test that the op works with torch.compile in cudagraph mode."""
    m = 128
    n = 4096
    group_size = 128

    input_tensor = torch.randn((m, n), dtype=torch.bfloat16, device="cuda")

    def group_fp8_quant_fn(x):
        return rocm_aiter_ops.group_fp8_quant(x, group_size)

    # Compile the op in the same low-overhead mode we care about for runtime use.
    compiled_fn = torch.compile(
        group_fp8_quant_fn,
        fullgraph=True,
        backend="inductor",
        mode="reduce-overhead",
        dynamic=False,
    )

    # First run triggers compilation; later runs should reuse the compiled path.
    x_fp8_eager, scales_eager = group_fp8_quant_fn(input_tensor)
    x_fp8_compiled, scales_compiled = compiled_fn(input_tensor)

    assert x_fp8_compiled.shape == x_fp8_eager.shape
    assert scales_compiled.shape == scales_eager.shape
    assert x_fp8_compiled.shape == (m, n)
    assert scales_compiled.shape == (m, (n + group_size - 1) // group_size)

    assert torch.allclose(
        x_fp8_compiled.to(torch.float32),
        x_fp8_eager.to(torch.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    assert torch.allclose(scales_compiled, scales_eager, rtol=1e-3, atol=1e-3)

    input_tensor_2 = torch.randn((m, n), dtype=torch.bfloat16, device="cuda")
    x_fp8_eager_2, scales_eager_2 = group_fp8_quant_fn(input_tensor_2)
    x_fp8_compiled_2, scales_compiled_2 = compiled_fn(input_tensor_2)

    assert torch.allclose(
        x_fp8_compiled_2.to(torch.float32),
        x_fp8_eager_2.to(torch.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    assert torch.allclose(scales_compiled_2, scales_eager_2, rtol=1e-3, atol=1e-3)


def test_rocm_aiter_group_fp8_quant_different_shapes():
    """Test the op with different input shapes."""
    group_size = 128
    test_shapes = [
        (64, 2048),
        (256, 8192),
        (32, 1024),
        (512, 4096),
    ]

    for m, n in test_shapes:
        input_tensor = torch.randn((m, n), dtype=torch.bfloat16, device="cuda")

        x_fp8, scales = rocm_aiter_ops.group_fp8_quant(input_tensor, group_size)

        # Each row gets one scale per 128-column group.
        assert x_fp8.shape == (m, n)
        assert scales.shape == (m, (n + group_size - 1) // group_size)

        from aiter import dtypes

        assert x_fp8.dtype == dtypes.fp8
        assert scales.dtype == torch.float32
