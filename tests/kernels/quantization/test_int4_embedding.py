# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for int4 per-channel embedding quantization."""

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm.model_executor.layers.quantization.online.int4 import (
    Int4PerChannelEmbeddingMethod,
)
from vllm.platforms import current_platform


@pytest.mark.parametrize("num_embeddings", [1000, 10000])
@pytest.mark.parametrize("embedding_dim", [256, 512])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_int4_quantization_accuracy(num_embeddings, embedding_dim, dtype):
    """Test that int4 quantization maintains reasonable accuracy."""
    torch.manual_seed(42)

    # Create a random embedding weight
    weight = torch.randn(num_embeddings, embedding_dim, dtype=dtype)

    # Compute int4 quantization manually
    max_abs = weight.abs().max(dim=0, keepdim=True).values
    scale = (max_abs / 7.0).to(dtype)
    q = (weight / scale).round().clamp(-7, 7).to(torch.int8)

    # Dequantize
    q_float = q.to(dtype)
    dequantized = q_float * scale

    # Check that the quantization error is reasonable
    # Int4 quantization should have max error of scale/2 per element
    max_error = (weight - dequantized).abs().max()
    expected_max_error = scale.max() / 2

    # Use a more relaxed tolerance for bfloat16 due to its lower precision
    tolerance = 1.10 if dtype == torch.bfloat16 else 1.01
    assert max_error <= expected_max_error * tolerance

    # Check that quantization preserves the general structure
    # Int4 has limited precision, so we just check it's not completely broken
    correlation = torch.corrcoef(
        torch.stack([weight.flatten(), dequantized.flatten()])
    )[0, 1]
    assert correlation > 0.9  # High correlation expected


@pytest.mark.parametrize("num_embeddings", [1000, 10000])
@pytest.mark.parametrize("embedding_dim", [256, 512])
def test_int4_packing(num_embeddings, embedding_dim):
    """Test that int4 values are correctly packed and unpacked."""
    torch.manual_seed(42)

    # Create random int4 values in range [-7, 7]
    q = torch.randint(-7, 8, (num_embeddings, embedding_dim), dtype=torch.int8)

    # Pack to uint8 (two int4 values per byte)
    q_uint4 = (q + 8).to(torch.uint8)  # Shift to [0, 15]
    packed = q_uint4[:, ::2] | (q_uint4[:, 1::2] << 4)

    # Unpack
    low = (packed & 0xF).to(torch.int16)
    high = (packed >> 4).to(torch.int16)
    q_uint4_unpacked = torch.stack([low, high], dim=-1).view(packed.shape[0], -1)
    q_unpacked = q_uint4_unpacked.to(torch.int8) - 8

    # Verify packing/unpacking is lossless
    assert torch.equal(q, q_unpacked)


def test_int4_embedding_method():
    """Test Int4PerChannelEmbeddingMethod basic functionality."""
    torch.manual_seed(42)

    # Create a mock layer with weight - must be nn.Module
    class MockLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.randn(100, 256, dtype=torch.float32), requires_grad=False
            )
            self.params_dtype = torch.float32

    layer = MockLayer()
    method = Int4PerChannelEmbeddingMethod()

    # Process weights
    method.process_weights_after_loading(layer)

    # Check that weight is now packed
    assert layer.weight.dtype == torch.uint8
    assert layer.weight.shape == (100, 128)  # Packed: 256/2 = 128
    assert hasattr(layer, "weight_scale")
    assert layer.weight_scale.shape == (1, 256)

    # Test embedding lookup
    indices = torch.randint(0, 100, (10,), dtype=torch.long)
    embeddings = method.embedding(layer, indices)

    assert embeddings.shape == (10, 256)
    assert embeddings.dtype == torch.float32


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="int4 embedding kernels require CUDA"
)
@pytest.mark.parametrize("vocab_size", [1024, 151936])
@pytest.mark.parametrize("hidden_size", [512, 1024])
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_bias", [False, True])
def test_int4_lm_head_gemv(vocab_size, hidden_size, batch_size, dtype, use_bias):
    """Compare the int4 lm-head kernel against the Python reference."""
    torch.manual_seed(42)
    device = "cuda"
    packed = torch.randint(
        1, 16, (vocab_size, hidden_size // 2), dtype=torch.uint8, device=device
    )
    scale = torch.rand(1, hidden_size, dtype=dtype, device=device) * 0.01
    hidden = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    bias = (
        torch.randn(vocab_size, dtype=dtype, device=device) * 0.1 if use_bias else None
    )

    # Python reference
    low = (packed & 0xF).to(torch.int16)
    high = (packed >> 4).to(torch.int16)
    q_uint4 = torch.stack([low, high], dim=-1).view(vocab_size, hidden_size)
    w = (q_uint4.to(dtype) - 8.0) * scale
    expected = hidden @ w.t()
    if bias is not None:
        expected = expected + bias

    from vllm import _custom_ops as ops

    got = ops.int4_lm_head_gemv(packed, scale, hidden, bias)
    torch.testing.assert_close(got, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="int4 embedding kernels require CUDA"
)
@pytest.mark.parametrize("num_embeddings", [1024, 151936])
@pytest.mark.parametrize("hidden_size", [512, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_int4_embedding_lookup(num_embeddings, hidden_size, dtype):
    """Compare the int4 embedding lookup kernel against the Python reference."""
    torch.manual_seed(42)
    device = "cuda"
    packed = torch.randint(
        1, 16, (num_embeddings, hidden_size // 2), dtype=torch.uint8, device=device
    )
    scale = torch.rand(1, hidden_size, dtype=dtype, device=device) * 0.01
    ids = torch.randint(0, num_embeddings, (17,), dtype=torch.long, device=device)

    # Python reference
    low = (packed[ids] & 0xF).to(torch.int16)
    high = (packed[ids] >> 4).to(torch.int16)
    q_uint4 = torch.stack([low, high], dim=-1).view(ids.numel(), hidden_size)
    expected = (q_uint4.to(dtype) - 8.0) * scale

    from vllm import _custom_ops as ops

    got = ops.int4_embedding_lookup(packed, scale, ids, dtype)
    torch.testing.assert_close(got, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="int4 embedding kernels require CUDA"
)
def test_int4_embedding_lookup_2d_ids():
    """Check that 2D input_ids preserve their shape in the output."""
    torch.manual_seed(0)
    device = "cuda"
    packed = torch.randint(1, 16, (128, 32), dtype=torch.uint8, device=device)
    scale = torch.rand(1, 64, dtype=torch.bfloat16, device=device) * 0.01
    ids = torch.randint(0, 128, (3, 5), dtype=torch.long, device=device)

    from vllm import _custom_ops as ops

    got = ops.int4_embedding_lookup(packed, scale, ids, torch.bfloat16)
    assert got.shape == (3, 5, 64)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="int4 embedding kernels require CUDA"
)
def test_int4_embedding_lookup_opcheck():
    device = "cuda"
    packed = torch.randint(1, 16, (128, 32), dtype=torch.uint8, device=device)
    scale = torch.rand(1, 64, dtype=torch.bfloat16, device=device) * 0.01
    ids = torch.arange(4, dtype=torch.long, device=device)
    opcheck(torch.ops._C.int4_embedding_lookup, (packed, scale, ids, torch.bfloat16))


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="int4 embedding kernels require CUDA"
)
def test_int4_lm_head_gemv_opcheck():
    device = "cuda"
    packed = torch.randint(1, 16, (128, 32), dtype=torch.uint8, device=device)
    scale = torch.rand(1, 64, dtype=torch.bfloat16, device=device) * 0.01
    hidden = torch.randn(2, 64, dtype=torch.bfloat16, device=device)
    opcheck(torch.ops._C.int4_lm_head_gemv, (packed, scale, hidden, None))


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="int4 embedding kernels require CUDA"
)
def test_int4_embedding_odd_hidden_size():
    """Odd hidden sizes must raise a clear runtime error."""
    device = "cuda"
    packed = torch.randint(1, 16, (16, 3), dtype=torch.uint8, device=device)
    scale = torch.rand(1, 7, dtype=torch.bfloat16, device=device) * 0.01
    ids = torch.arange(4, dtype=torch.long, device=device)

    from vllm import _custom_ops as ops

    with pytest.raises(RuntimeError, match="even"):
        ops.int4_embedding_lookup(packed, scale, ids, torch.bfloat16)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="int4 embedding kernels require CUDA"
)
def test_int4_embedding_lookup_empty_ids():
    """Empty input_ids should return an empty output tensor."""
    device = "cuda"
    packed = torch.randint(1, 16, (64, 8), dtype=torch.uint8, device=device)
    scale = torch.rand(1, 16, dtype=torch.bfloat16, device=device) * 0.01
    ids = torch.empty((0,), dtype=torch.long, device=device)

    from vllm import _custom_ops as ops

    got = ops.int4_embedding_lookup(packed, scale, ids, torch.bfloat16)
    assert got.shape == (0, 16)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="int4 embedding kernels require CUDA"
)
def test_int4_embedding_lookup_min_hidden_size():
    """Minimum even hidden size (H=2) should work."""
    device = "cuda"
    packed = torch.randint(1, 16, (16, 1), dtype=torch.uint8, device=device)
    scale = torch.rand(1, 2, dtype=torch.bfloat16, device=device) * 0.01
    ids = torch.tensor([0, 5, 15], dtype=torch.long, device=device)

    from vllm import _custom_ops as ops

    got = ops.int4_embedding_lookup(packed, scale, ids, torch.bfloat16)
    assert got.shape == (3, 2)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="int4 embedding kernels require CUDA"
)
def test_int4_lm_head_gemv_noncontiguous_hidden():
    """Non-contiguous hidden_states must produce correct logits."""
    torch.manual_seed(0)
    device = "cuda"
    packed = torch.randint(1, 16, (128, 32), dtype=torch.uint8, device=device)
    scale = torch.rand(1, 64, dtype=torch.bfloat16, device=device) * 0.01
    hidden = torch.randn(4, 64, dtype=torch.bfloat16, device=device)
    # Make the hidden tensor non-contiguous by slicing every other row.
    strided = hidden[::2]
    assert not strided.is_contiguous()

    from vllm import _custom_ops as ops

    got = ops.int4_lm_head_gemv(packed, scale, strided, None)

    # Python reference using the same (now materialized) non-contiguous slice.
    low = (packed & 0xF).to(torch.int16)
    high = (packed >> 4).to(torch.int16)
    q_uint4 = torch.stack([low, high], dim=-1).view(128, 64)
    w = (q_uint4.to(torch.bfloat16) - 8.0) * scale
    expected = strided @ w.t()
    torch.testing.assert_close(got, expected, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    # Run basic tests
    test_int4_quantization_accuracy(1000, 256, torch.float32)
    test_int4_packing(1000, 256)
    test_int4_embedding_method()
    print("All basic tests passed!")
