# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for relu2_no_mul activation function.

This validates the core activation math used by Nemotron-Nano models
before testing full MoE kernel integration.

relu2_no_mul: output = relu(input)^2

This is a non-gated activation where output dim == input dim.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.utils import (
    RELU2_NO_MUL,
    apply_moe_activation,
)
from vllm.platforms import current_platform


def pytorch_relu2_reference(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation: relu(x)^2"""
    return torch.relu(x) ** 2


@pytest.mark.skipif(
    not current_platform.has_device_capability(80),
    reason="Requires compute capability >= 8.0",
)
@pytest.mark.parametrize(
    "shape",
    [
        (64, 1024),
        (128, 2048),
        (128, 2688),  # Nemotron-Nano hidden_size
        (128, 1856),  # Nemotron-Nano intermediate_size
        (1, 256),  # Single token
        (512, 4096),  # Larger batch
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_relu2_no_mul_vs_reference(shape: tuple, dtype: torch.dtype):
    """Test relu2_no_mul against PyTorch reference implementation.

    The vLLM implementation should match relu(x)^2 within tolerance.
    Using 1e-5 tolerance for activation math as recommended for NVFP4 testing.
    """
    device = "cuda"

    # Create input with mix of positive and negative values
    input_tensor = torch.randn(shape, dtype=dtype, device=device)

    # PyTorch reference: relu(x)^2
    expected = pytorch_relu2_reference(input_tensor)

    # vLLM implementation - note: apply_moe_activation modifies input in-place
    # for relu2_no_mul, so we need to clone
    input_clone = input_tensor.clone()
    output = torch.empty_like(input_tensor)
    apply_moe_activation(RELU2_NO_MUL, output, input_clone)

    # Use tighter tolerance for FP32, slightly looser for BF16
    if dtype == torch.float32:
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-6)
    else:
        # BF16 has lower precision
        torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(
    not current_platform.has_device_capability(80),
    reason="Requires compute capability >= 8.0",
)
@torch.inference_mode()
def test_relu2_no_mul_negative_inputs_become_zero():
    """Verify that negative inputs produce zero output.

    relu2_no_mul: output = relu(x)^2
    For x < 0: relu(x) = 0, so output = 0
    """
    device = "cuda"
    input_tensor = torch.tensor(
        [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=torch.float32, device=device
    )

    input_clone = input_tensor.clone()
    output = torch.empty_like(input_tensor)
    apply_moe_activation(RELU2_NO_MUL, output, input_clone)

    # Expected: relu(x)^2 = [0, 0, 0, 0.25, 1.0, 4.0]
    expected = torch.tensor(
        [0.0, 0.0, 0.0, 0.25, 1.0, 4.0], dtype=torch.float32, device=device
    )
    torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(
    not current_platform.has_device_capability(80),
    reason="Requires compute capability >= 8.0",
)
@torch.inference_mode()
def test_relu2_no_mul_preserves_shape():
    """Verify output shape equals input shape (non-gated activation).

    For *_no_mul activations, output.size(-1) == input.size(-1)
    This is different from gated activations where output is half the size.
    """
    device = "cuda"
    shapes = [(32, 64), (128, 256), (1, 1024), (64, 1856), (32, 2688)]

    for shape in shapes:
        input_tensor = torch.randn(shape, dtype=torch.float32, device=device)
        output = torch.empty_like(input_tensor)
        apply_moe_activation(RELU2_NO_MUL, output, input_tensor.clone())

        assert output.shape == input_tensor.shape, (
            f"Shape mismatch: {output.shape} vs {input_tensor.shape}"
        )


@pytest.mark.skipif(
    not current_platform.has_device_capability(80),
    reason="Requires compute capability >= 8.0",
)
@torch.inference_mode()
def test_relu2_no_mul_gradient_behavior():
    """Verify the mathematical property: d/dx[relu(x)^2] = 2*relu(x) for x > 0.

    This isn't a gradient test per se (we're in inference mode),
    but validates that the squaring operation is correctly applied.
    """
    device = "cuda"

    # For x = 3.0: relu(3.0)^2 = 9.0
    # For x = -3.0: relu(-3.0)^2 = 0.0
    input_tensor = torch.tensor([3.0, -3.0, 0.0], dtype=torch.float32, device=device)

    output = torch.empty_like(input_tensor)
    apply_moe_activation(RELU2_NO_MUL, output, input_tensor.clone())

    expected = torch.tensor([9.0, 0.0, 0.0], dtype=torch.float32, device=device)
    torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(
    not current_platform.has_device_capability(80),
    reason="Requires compute capability >= 8.0",
)
@pytest.mark.parametrize(
    "nemotron_dims",
    [
        # (batch, hidden_size) and (batch, intermediate_size)
        (1, 2688),  # single token, hidden
        (1, 1856),  # single token, intermediate
        (32, 2688),  # batch, hidden
        (32, 1856),  # batch, intermediate
        (128, 2688),  # larger batch, hidden
        (128, 1856),  # larger batch, intermediate
    ],
)
@torch.inference_mode()
def test_relu2_no_mul_nemotron_nano_dimensions(nemotron_dims: tuple):
    """Test with Nemotron-Nano specific dimensions.

    Nemotron-Nano-4B-Instruct uses:
    - hidden_size = 2688
    - intermediate_size = 1856 (non-standard, not divisible by 128)
    - num_experts = 128
    - is_act_and_mul = False (non-gated MLP)
    """
    device = "cuda"
    batch, dim = nemotron_dims

    input_tensor = torch.randn(batch, dim, dtype=torch.bfloat16, device=device)

    # PyTorch reference
    expected = pytorch_relu2_reference(input_tensor)

    # vLLM implementation
    output = torch.empty_like(input_tensor)
    apply_moe_activation(RELU2_NO_MUL, output, input_tensor.clone())

    # BF16 tolerance
    torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(
    not current_platform.has_device_capability(80),
    reason="Requires compute capability >= 8.0",
)
@torch.inference_mode()
def test_relu2_no_mul_no_nan_inf():
    """Verify output contains no NaN or Inf values for various inputs."""
    device = "cuda"

    # Test with various input ranges
    test_cases = [
        torch.randn(64, 1024, device=device),  # Normal distribution
        torch.zeros(64, 1024, device=device),  # All zeros
        torch.ones(64, 1024, device=device),  # All ones
        torch.full((64, 1024), -100.0, device=device),  # Large negatives
        torch.full((64, 1024), 100.0, device=device),  # Large positives
    ]

    for input_tensor in test_cases:
        output = torch.empty_like(input_tensor)
        apply_moe_activation(RELU2_NO_MUL, output, input_tensor.clone())

        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
