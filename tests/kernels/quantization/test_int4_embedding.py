# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for int4 per-channel embedding quantization."""

import pytest
import torch

from vllm.model_executor.layers.quantization.online.int4 import (
    Int4PerChannelEmbeddingMethod,
    Int4VocabParallelEmbedding,
)


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

    assert max_error <= expected_max_error * 1.01  # 1% tolerance for rounding

    # Check that quantization preserves the general structure
    # Int4 has limited precision, so we just check it's not completely broken
    correlation = torch.corrcoef(torch.stack([
        weight.flatten(),
        dequantized.flatten()
    ]))[0, 1]
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
                torch.randn(100, 256, dtype=torch.float32),
                requires_grad=False
            )
            self.params_dtype = torch.float32

    layer = MockLayer()
    method = Int4PerChannelEmbeddingMethod(layer)

    # Process weights
    method.process_weights_after_loading(layer)

    # Check that weight is now packed
    assert layer.weight.dtype == torch.uint8
    assert layer.weight.shape == (100, 128)  # Packed: 256/2 = 128
    assert hasattr(layer, 'weight_scale')
    assert layer.weight_scale.shape == (1, 256)

    # Test embedding lookup
    indices = torch.randint(0, 100, (10,), dtype=torch.long)
    embeddings = method.embedding(layer, indices)

    assert embeddings.shape == (10, 256)
    assert embeddings.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_int4_vocab_parallel_embedding():
    """Test Int4VocabParallelEmbedding with tensor parallelism."""
    torch.manual_seed(42)

    num_embeddings = 1000
    embedding_dim = 256

    # Create embedding layer
    embedding = Int4VocabParallelEmbedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        params_dtype=torch.float32,
    )

    # Check initial shape
    assert embedding.weight.dtype == torch.uint8
    assert embedding.weight.shape[1] == embedding_dim // 2

    # Simulate weight loading
    loaded_weight = torch.randn(num_embeddings, embedding_dim, dtype=torch.float32)
    embedding._int4_weight_loader(embedding.weight, loaded_weight)

    # Test forward pass
    indices = torch.randint(0, num_embeddings, (10,), dtype=torch.long)
    output = embedding(indices)

    assert output.shape == (10, embedding_dim)
    assert output.dtype == torch.float32


if __name__ == "__main__":
    # Run basic tests
    test_int4_quantization_accuracy(1000, 256, torch.float32)
    test_int4_packing(1000, 256)
    test_int4_embedding_method()
    print("All basic tests passed!")
