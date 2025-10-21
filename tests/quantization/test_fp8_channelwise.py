# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for FP8 channelwise quantization utilities.

Run `pytest tests/quantization/test_fp8_channelwise.py`.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    fp8_channelwise_quantize,
)
from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="FP8 channelwise quantization requires CUDA",
)
class TestFp8ChannelwiseQuantize:
    """Test suite for fp8_channelwise_quantize function."""

    def test_basic_quantization(self):
        """Test basic quantization with standard input."""
        x = torch.randn(4, 8, dtype=torch.float32, device="cuda")
        quantized, scale = fp8_channelwise_quantize(x)

        # Check output types and shapes
        assert quantized.dtype == current_platform.fp8_dtype()
        assert scale.dtype == torch.float32
        assert quantized.shape == x.shape
        assert scale.shape == (4, 1)  # Per-channel scale

    def test_different_dtypes(self):
        """Test quantization with different float input dtypes."""
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x = torch.randn(4, 8, dtype=dtype, device="cuda")
            quantized, scale = fp8_channelwise_quantize(x)

            assert quantized.dtype == current_platform.fp8_dtype()
            assert scale.dtype == dtype

    def test_custom_channel_dim(self):
        """Test quantization with custom channel dimension."""
        x = torch.randn(4, 8, 16, dtype=torch.float32, device="cuda")
        quantized, scale = fp8_channelwise_quantize(x, channel_dim=1)

        assert quantized.shape == x.shape
        assert scale.shape == (4, 1, 16)

    def test_dequantization_accuracy(self):
        """Test that quantization-dequantization has reasonable accuracy."""
        x = torch.randn(4, 8, dtype=torch.float32, device="cuda")
        quantized, scale = fp8_channelwise_quantize(x.clone())

        # Dequantize
        dequantized = quantized.to(torch.float32) * scale

        # Check relative error
        relative_error = torch.abs(x - dequantized) / (torch.abs(x) + 1e-6)
        assert torch.mean(relative_error) < 0.1  # 10% average error

    def test_per_channel_independence(self):
        """Test that channels are quantized independently."""
        x = torch.randn(4, 8, dtype=torch.float32, device="cuda")

        # Make channels have very different scales
        x[0] = x[0] * 100.0
        x[1] = x[1] * 0.01

        quantized, scale = fp8_channelwise_quantize(x)

        # Scales should be different across channels
        assert scale[0] > scale[1]  # First channel should have larger scale


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="FP8 channelwise quantization requires CUDA",
)
class TestFp8ChannelwiseQuantizeValidation:
    """Test suite for input validation of fp8_channelwise_quantize."""

    def test_empty_tensor_raises(self):
        """Test that empty tensor raises assertion error."""
        x = torch.tensor([], dtype=torch.float32, device="cuda")
        with pytest.raises(AssertionError, match="Input tensor must not be empty"):
            fp8_channelwise_quantize(x)

    def test_scalar_tensor_raises(self):
        """Test that scalar tensor raises assertion error."""
        x = torch.tensor(1.0, dtype=torch.float32, device="cuda")
        with pytest.raises(
            AssertionError, match="Input tensor must have at least 1 dimension"
        ):
            fp8_channelwise_quantize(x)
