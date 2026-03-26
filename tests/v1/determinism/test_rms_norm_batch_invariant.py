# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test batch-invariant RMS normalization against standard implementations.

This test compares the Triton-based batch-invariant RMS norm implementation
with the standard CUDA-based implementation to ensure numerical accuracy.
"""

import pytest
import torch
from utils import skip_unsupported

from vllm.model_executor.layers.batch_invariant import rms_norm as triton_rms_norm
from vllm.model_executor.layers.layernorm import RMSNorm


@skip_unsupported
@pytest.mark.parametrize("batch_size", [1, 4, 16, 64])
@pytest.mark.parametrize("hidden_size", [512, 2048, 4096, 8192])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_rms_norm_batch_invariant_vs_standard(
    default_vllm_config,
    batch_size: int,
    hidden_size: int,
    dtype: torch.dtype,
    eps: float,
):
    """
    Compare batch-invariant Triton RMS norm against standard CUDA implementation.

    Tests that the Triton-based batch-invariant RMS norm produces numerically
    equivalent results to the standard CUDA implementation across various
    configurations.
    """
    device = torch.device("cuda")

    # Create test input and weight
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    # Standard implementation (CUDA ops)
    rms_norm_layer = RMSNorm(hidden_size, eps=eps, dtype=dtype).to(device)
    rms_norm_layer.weight.data = weight.clone()

    standard_output = rms_norm_layer.forward_cuda(input_tensor)

    # Batch-invariant implementation (Triton)
    triton_output = triton_rms_norm(input_tensor, weight, eps=eps)

    # Compare outputs
    # Use looser tolerance for bfloat16 due to its lower precision
    if dtype == torch.bfloat16:
        rtol, atol = 1e-1, 1e-1  # 10% relative tolerance for bfloat16
    else:
        rtol, atol = 1e-2, 1e-2  # 1% for float16/float32

    torch.testing.assert_close(
        triton_output,
        standard_output,
        rtol=rtol,
        atol=atol,
        msg=f"RMS norm mismatch for batch_size={batch_size}, "
        f"hidden_size={hidden_size}, "
        f"dtype={dtype}, eps={eps}",
    )


@skip_unsupported
@pytest.mark.parametrize("batch_size", [1, 16, 128])
@pytest.mark.parametrize("seq_len", [1, 32, 512])
@pytest.mark.parametrize("hidden_size", [2048, 4096])
def test_rms_norm_3d_input(
    default_vllm_config, batch_size: int, seq_len: int, hidden_size: int
):
    """
    Test RMS norm with 3D input tensors (batch, seq_len, hidden_size).

    Ensures that the batch-invariant RMS norm correctly handles multi-dimensional
    inputs that are common in transformer models.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    eps = 1e-6

    torch.manual_seed(42)
    input_tensor = torch.randn(
        batch_size, seq_len, hidden_size, dtype=dtype, device=device
    )
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    # Standard implementation
    rms_norm_layer = RMSNorm(hidden_size, eps=eps, dtype=dtype).to(device)
    rms_norm_layer.weight.data = weight.clone()
    standard_output = rms_norm_layer.forward_cuda(input_tensor)

    # Batch-invariant implementation
    triton_output = triton_rms_norm(input_tensor, weight, eps=eps)

    # Use looser tolerance for bfloat16
    rtol, atol = 1e-1, 1e-1  # 10% tolerance for bfloat16

    torch.testing.assert_close(
        triton_output,
        standard_output,
        rtol=rtol,
        atol=atol,
        msg=f"RMS norm mismatch for 3D input with batch_size={batch_size}, "
        f"seq_len={seq_len}, hidden_size={hidden_size}",
    )


@skip_unsupported
def test_rms_norm_numerical_stability(default_vllm_config):
    """
    Test RMS norm numerical stability with extreme values.

    Ensures that both implementations handle edge cases like very small or large
    values without producing NaN or Inf.
    """
    device = torch.device("cuda")
    dtype = torch.float16
    eps = 1e-6
    hidden_size = 2048

    # Test cases with extreme values
    test_cases = [
        # Very small values
        torch.ones(4, hidden_size, dtype=dtype, device=device) * 1e-5,
        # Very large values
        torch.ones(4, hidden_size, dtype=dtype, device=device) * 1e4,
        # Mixed small and large
        torch.randn(4, hidden_size, dtype=dtype, device=device) * 100,
        # Values near zero
        torch.randn(4, hidden_size, dtype=dtype, device=device) * 1e-6,
    ]

    weight = torch.ones(hidden_size, dtype=dtype, device=device)

    for idx, input_tensor in enumerate(test_cases):
        # Standard implementation
        rms_norm_layer = RMSNorm(hidden_size, eps=eps, dtype=dtype).to(device)
        rms_norm_layer.weight.data = weight.clone()
        standard_output = rms_norm_layer.forward_cuda(input_tensor)

        # Batch-invariant implementation
        triton_output = triton_rms_norm(input_tensor, weight, eps=eps)

        # Check for NaN or Inf
        assert not torch.isnan(standard_output).any(), (
            f"Standard RMS norm produced NaN for test case {idx}"
        )
        assert not torch.isinf(standard_output).any(), (
            f"Standard RMS norm produced Inf for test case {idx}"
        )
        assert not torch.isnan(triton_output).any(), (
            f"Triton RMS norm produced NaN for test case {idx}"
        )
        assert not torch.isinf(triton_output).any(), (
            f"Triton RMS norm produced Inf for test case {idx}"
        )

        # Compare outputs - very lenient for extreme values with float16
        torch.testing.assert_close(
            triton_output,
            standard_output,
            rtol=2e-1,  # 20% tolerance for extreme values
            atol=2e-1,
            msg=f"RMS norm mismatch for extreme value test case {idx}",
        )


@skip_unsupported
def test_rms_norm_formula(default_vllm_config):
    """
    Test that RMS norm follows the correct mathematical formula.

    Verifies: output = input / sqrt(mean(input^2) + eps) * weight
    """
    device = torch.device("cuda")
    dtype = torch.float32  # Use float32 for higher precision in formula check
    eps = 1e-6
    hidden_size = 1024

    torch.manual_seed(42)
    input_tensor = torch.randn(8, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    # Compute expected output using the formula
    variance = (input_tensor.pow(2).mean(dim=-1, keepdim=True)).to(dtype)
    expected_output = input_tensor * torch.rsqrt(variance + eps) * weight

    # Batch-invariant implementation
    triton_output = triton_rms_norm(input_tensor, weight, eps=eps)

    # Compare against formula
    torch.testing.assert_close(
        triton_output,
        expected_output,
        rtol=1e-4,
        atol=1e-4,
        msg="Triton RMS norm doesn't match expected formula",
    )


@skip_unsupported
@pytest.mark.parametrize("hidden_size", [128, 1024, 4096, 16384])
def test_rms_norm_different_hidden_sizes(default_vllm_config, hidden_size: int):
    """
    Test RMS norm with various hidden sizes to ensure block size handling.

    The Triton kernel uses a fixed BLOCK_SIZE=1024, so this tests that it
    correctly handles hidden sizes both smaller and larger than the block size.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    eps = 1e-6
    batch_size = 16

    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    # Standard implementation
    rms_norm_layer = RMSNorm(hidden_size, eps=eps, dtype=dtype).to(device)
    rms_norm_layer.weight.data = weight.clone()
    standard_output = rms_norm_layer.forward_cuda(input_tensor)

    # Batch-invariant implementation
    triton_output = triton_rms_norm(input_tensor, weight, eps=eps)

    # Use looser tolerance for bfloat16
    rtol, atol = 1e-1, 1e-1  # 10% tolerance for bfloat16

    torch.testing.assert_close(
        triton_output,
        standard_output,
        rtol=rtol,
        atol=atol,
        msg=f"RMS norm mismatch for hidden_size={hidden_size}",
    )


@skip_unsupported
def test_rms_norm_determinism(default_vllm_config):
    """
    Test that batch-invariant RMS norm produces deterministic results.

    Runs the same input through the kernel multiple times and verifies
    identical outputs.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    eps = 1e-6
    hidden_size = 4096
    batch_size = 32

    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    # Run multiple times
    outputs = []
    for _ in range(5):
        output = triton_rms_norm(input_tensor.clone(), weight, eps=eps)
        outputs.append(output)

    # All outputs should be identical
    reference = outputs[0]
    for idx, output in enumerate(outputs[1:], start=1):
        torch.testing.assert_close(
            output,
            reference,
            rtol=0.0,
            atol=0.0,
            msg=f"RMS norm not deterministic: run {idx} differs from reference",
        )


if __name__ == "__main__":
    # Run a quick smoke test
    print("Running quick smoke test of RMS norm implementations...")

    device = torch.device("cuda")
    batch_size = 8
    hidden_size = 4096
    dtype = torch.bfloat16
    eps = 1e-6

    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    # Standard implementation
    rms_norm_layer = RMSNorm(hidden_size, eps=eps, dtype=dtype).to(device)
    rms_norm_layer.weight.data = weight.clone()
    standard_output = rms_norm_layer.forward_cuda(input_tensor)

    # Batch-invariant implementation
    triton_output = triton_rms_norm(input_tensor, weight, eps=eps)

    # Compare
    max_diff = (triton_output - standard_output).abs().max().item()
    mean_diff = (triton_output - standard_output).abs().mean().item()

    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")
    print(f"Standard output sample: {standard_output[0, :5].tolist()}")
    print(f"Triton output sample: {triton_output[0, :5].tolist()}")

    if max_diff < 1e-3:
        print("✓ Smoke test passed!")
    else:
        print("✗ Smoke test failed - differences too large")
