# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the shuffle_rows function

Run `pytest tests/kernels/test_shuffle_rows.py`.
"""

import pytest
import torch

from vllm._custom_ops import shuffle_rows
from vllm.platforms import current_platform


@pytest.mark.parametrize("num_tokens", [1, 16, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize("hidden_size", [128, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32])
def test_shuffle_rows_basic(num_tokens: int, hidden_size: int,
                            dtype: torch.dtype):
    """Test basic functionality of shuffle_rows with various tensor sizes and
    dtypes."""
    if not current_platform.is_cuda():
        pytest.skip("shuffle_rows requires CUDA")

    # Create input tensor
    input_tensor = torch.randn(num_tokens,
                               hidden_size,
                               device="cuda",
                               dtype=dtype)

    # Create a simple permutation map (identity mapping)
    dst2src_map = torch.arange(num_tokens, device="cuda", dtype=torch.int32)

    # Test shuffle_rows
    output = shuffle_rows(input_tensor, dst2src_map)

    # With identity mapping, output should be identical to input
    torch.testing.assert_close(output, input_tensor, atol=0, rtol=0)

    # Check output shape
    assert output.shape == (num_tokens, hidden_size)
    assert output.dtype == dtype
    assert output.device == input_tensor.device


@pytest.mark.parametrize("num_tokens", [16, 64, 128])
@pytest.mark.parametrize("hidden_size", [128, 512, 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_shuffle_rows_permutation(num_tokens: int, hidden_size: int,
                                  dtype: torch.dtype):
    """Test shuffle_rows with actual permutation."""
    if not current_platform.is_cuda():
        pytest.skip("shuffle_rows requires CUDA")

    # Create input tensor
    input_tensor = torch.randn(num_tokens,
                               hidden_size,
                               device="cuda",
                               dtype=dtype)

    # Create a reverse permutation map
    dst2src_map = torch.arange(num_tokens - 1,
                               -1,
                               -1,
                               device="cuda",
                               dtype=torch.int32)

    # Test shuffle_rows
    output = shuffle_rows(input_tensor, dst2src_map)

    # Check that the output is the reverse of the input
    expected_output = torch.flip(input_tensor, dims=[0])
    torch.testing.assert_close(output, expected_output, atol=1e-6, rtol=1e-5)

    # Check output shape and properties
    assert output.shape == (num_tokens, hidden_size)
    assert output.dtype == dtype
    assert output.device == input_tensor.device


@pytest.mark.parametrize("num_tokens", [32, 64])
@pytest.mark.parametrize("hidden_size", [256, 512])
def test_shuffle_rows_expansion(num_tokens: int, hidden_size: int):
    """Test shuffle_rows with expansion (more output tokens than input
    tokens)."""
    if not current_platform.is_cuda():
        pytest.skip("shuffle_rows requires CUDA")

    dtype = torch.float16

    # Create input tensor
    input_tensor = torch.randn(num_tokens,
                               hidden_size,
                               device="cuda",
                               dtype=dtype)

    # Create a mapping that duplicates some tokens (expansion)
    expanded_size = num_tokens * 2
    dst2src_map = torch.randint(0,
                                num_tokens, (expanded_size, ),
                                device="cuda",
                                dtype=torch.int32)

    # Test shuffle_rows
    output = shuffle_rows(input_tensor, dst2src_map)

    # Check output shape
    assert output.shape == (expanded_size, hidden_size)
    assert output.dtype == dtype
    assert output.device == input_tensor.device

    # Verify that each output row matches the corresponding input row
    for i in range(expanded_size):
        src_idx = dst2src_map[i].item()
        torch.testing.assert_close(output[i],
                                   input_tensor[src_idx],
                                   atol=1e-6,
                                   rtol=1e-5)


@pytest.mark.parametrize("num_tokens", [16, 64])
@pytest.mark.parametrize("hidden_size", [128, 512])
def test_shuffle_rows_random_permutation(num_tokens: int, hidden_size: int):
    """Test shuffle_rows with random permutation."""
    if not current_platform.is_cuda():
        pytest.skip("shuffle_rows requires CUDA")

    dtype = torch.float16

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Create input tensor
    input_tensor = torch.randn(num_tokens,
                               hidden_size,
                               device="cuda",
                               dtype=dtype)

    # Create a random permutation map
    dst2src_map = torch.randperm(num_tokens, device="cuda", dtype=torch.int32)

    # Test shuffle_rows
    output = shuffle_rows(input_tensor, dst2src_map)

    # Check output shape and properties
    assert output.shape == (num_tokens, hidden_size)
    assert output.dtype == dtype
    assert output.device == input_tensor.device

    # Verify that each output row matches the corresponding input row
    for i in range(num_tokens):
        src_idx = dst2src_map[i].item()
        torch.testing.assert_close(output[i],
                                   input_tensor[src_idx],
                                   atol=1e-6,
                                   rtol=1e-5)


def test_shuffle_rows_edge_cases():
    """Test shuffle_rows with edge cases."""
    if not current_platform.is_cuda():
        pytest.skip("shuffle_rows requires CUDA")

    dtype = torch.float16

    # Test with single token
    input_tensor = torch.randn(1, 128, device="cuda", dtype=dtype)
    dst2src_map = torch.tensor([0], device="cuda", dtype=torch.int32)
    output = shuffle_rows(input_tensor, dst2src_map)
    torch.testing.assert_close(output, input_tensor, atol=0, rtol=0)

    # Test with single feature dimension
    input_tensor = torch.randn(16, 1, device="cuda", dtype=dtype)
    dst2src_map = torch.arange(16, device="cuda", dtype=torch.int32)
    output = shuffle_rows(input_tensor, dst2src_map)
    torch.testing.assert_close(output, input_tensor, atol=0, rtol=0)


def test_shuffle_rows_moe_like_scenario():
    """Test shuffle_rows in a scenario similar to MoE usage."""
    if not current_platform.is_cuda():
        pytest.skip("shuffle_rows requires CUDA")

    dtype = torch.float16
    batch_size = 32
    hidden_size = 1024
    topk = 2

    # Simulate input tokens
    input_tensor = torch.randn(batch_size,
                               hidden_size,
                               device="cuda",
                               dtype=dtype)

    # Simulate expert assignment (each token goes to topk experts)
    # This creates a mapping where tokens are duplicated for multiple experts
    total_tokens = batch_size * topk
    dst2src_map = torch.zeros(total_tokens, device="cuda", dtype=torch.int32)

    # Fill the mapping to simulate MoE token distribution
    for i in range(batch_size):
        for k in range(topk):
            dst2src_map[i * topk + k] = i

    # Test shuffle_rows
    output = shuffle_rows(input_tensor, dst2src_map)

    # Check output shape
    assert output.shape == (total_tokens, hidden_size)
    assert output.dtype == dtype
    assert output.device == input_tensor.device

    # Verify that tokens are correctly duplicated
    for i in range(batch_size):
        for k in range(topk):
            output_idx = i * topk + k
            torch.testing.assert_close(output[output_idx],
                                       input_tensor[i],
                                       atol=1e-6,
                                       rtol=1e-5)


@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32])
def test_shuffle_rows_dtype_consistency(dtype: torch.dtype):
    """Test that shuffle_rows preserves dtype correctly."""
    if not current_platform.is_cuda():
        pytest.skip("shuffle_rows requires CUDA")

    num_tokens = 64
    hidden_size = 512

    # Create input tensor with specific dtype
    input_tensor = torch.randn(num_tokens,
                               hidden_size,
                               device="cuda",
                               dtype=dtype)
    dst2src_map = torch.arange(num_tokens, device="cuda", dtype=torch.int32)

    # Test shuffle_rows
    output = shuffle_rows(input_tensor, dst2src_map)

    # Verify dtype is preserved
    assert output.dtype == dtype
    assert output.device == input_tensor.device
    torch.testing.assert_close(output, input_tensor, atol=1e-6, rtol=1e-5)


def test_shuffle_rows_device_consistency():
    """Test that shuffle_rows maintains device consistency."""
    if not current_platform.is_cuda():
        pytest.skip("shuffle_rows requires CUDA")

    num_tokens = 32
    hidden_size = 256
    dtype = torch.float16

    # Create input tensor on CUDA
    input_tensor = torch.randn(num_tokens,
                               hidden_size,
                               device="cuda",
                               dtype=dtype)
    dst2src_map = torch.arange(num_tokens, device="cuda", dtype=torch.int32)

    # Test shuffle_rows
    output = shuffle_rows(input_tensor, dst2src_map)

    # Verify device is maintained
    assert output.device == input_tensor.device
    assert output.device.type == "cuda"


def test_shuffle_rows_contiguous_output():
    """Test that shuffle_rows produces contiguous output."""
    if not current_platform.is_cuda():
        pytest.skip("shuffle_rows requires CUDA")

    num_tokens = 64
    hidden_size = 512
    dtype = torch.float16

    # Create input tensor
    input_tensor = torch.randn(num_tokens,
                               hidden_size,
                               device="cuda",
                               dtype=dtype)
    dst2src_map = torch.arange(num_tokens, device="cuda", dtype=torch.int32)

    # Test shuffle_rows
    output = shuffle_rows(input_tensor, dst2src_map)

    # Verify output is contiguous
    assert output.is_contiguous()
