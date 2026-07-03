# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for NVFP4 per-partition output rescaling.

Tests the rescale logic in CompressedTensorsW4A4Fp4 when fused linear
layers (e.g. gate_up_proj) have non-uniform weight_global_scale values.

Run: pytest tests/quantization/test_nvfp4_partition_rescale.py -v
"""

from unittest.mock import MagicMock

import pytest
import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4 import (  # noqa: E501
    CompressedTensorsW4A4Fp4,
)


class FakeKernel:
    """Minimal stub that returns a fixed output tensor."""

    def __init__(self, output: torch.Tensor):
        self._output = output

    def apply_weights(self, layer, x, bias=None):
        out = self._output.clone()
        if bias is not None:
            out = out + bias
        return out

    def process_weights_after_loading(self, layer):
        pass


def _make_layer(logical_widths, weight_global_scale_values, use_a16=True):
    """Build a minimal nn.Module that mimics a CT-loaded NVFP4 layer."""
    layer = torch.nn.Module()
    layer.logical_widths = logical_widths
    total_out = sum(logical_widths)

    layer.weight_packed = Parameter(
        torch.zeros(total_out, 8, dtype=torch.uint8), requires_grad=False
    )
    layer.weight_global_scale = Parameter(
        torch.tensor(weight_global_scale_values, dtype=torch.float32),
        requires_grad=False,
    )
    layer.weight_scale = Parameter(
        torch.ones(total_out, 1, dtype=torch.float32), requires_grad=False
    )

    if not use_a16:
        layer.input_global_scale = Parameter(
            torch.tensor(
                [1.0] * len(logical_widths), dtype=torch.float32
            ),
            requires_grad=False,
        )

    return layer


class TestNonUniformTwoPartitionRescale:
    """Non-uniform two-partition layers should get per-partition rescaling."""

    def test_numerical_recovery(self):
        """Rescaled output matches per-partition independent GEMM."""
        logical_widths = [2, 2]
        # CT divisors: gate=100, up=200
        layer = _make_layer(logical_widths, [100.0, 200.0])

        fake_output = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
        scheme = CompressedTensorsW4A4Fp4(use_a16=True)
        scheme.kernel = FakeKernel(fake_output)

        scheme.process_weights_after_loading(layer)

        assert layer._output_partition_rescale is not None
        rescale = layer._output_partition_rescale.data
        # base = min(100, 200) = 100
        # gate rescale = 100/100 = 1.0, up rescale = 100/200 = 0.5
        assert torch.allclose(rescale, torch.tensor([1.0, 1.0, 0.5, 0.5]))

        x_dummy = torch.randn(1, 16)
        out = scheme.apply_weights(layer, x_dummy)
        expected = torch.tensor([[10.0, 20.0, 15.0, 20.0]])
        assert torch.allclose(out, expected)

    def test_bias_not_rescaled(self):
        """Bias is added after rescaling, so it is not multiplied."""
        logical_widths = [2, 2]
        layer = _make_layer(logical_widths, [100.0, 200.0])

        fake_output = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
        scheme = CompressedTensorsW4A4Fp4(use_a16=True)
        scheme.kernel = FakeKernel(fake_output)

        scheme.process_weights_after_loading(layer)

        bias = torch.tensor([1.0, 1.0, 1.0, 1.0])
        x_dummy = torch.randn(1, 16)
        out = scheme.apply_weights(layer, x_dummy, bias=bias)
        # [10*1+1, 20*1+1, 30*0.5+1, 40*0.5+1]
        expected = torch.tensor([[11.0, 21.0, 16.0, 21.0]])
        assert torch.allclose(out, expected)

    def test_unequal_partition_widths(self):
        """Non-equal partition widths should be handled correctly."""
        logical_widths = [3, 5]
        layer = _make_layer(logical_widths, [100.0, 200.0])

        fake_output = torch.ones(1, 8) * 10.0
        scheme = CompressedTensorsW4A4Fp4(use_a16=True)
        scheme.kernel = FakeKernel(fake_output)

        scheme.process_weights_after_loading(layer)

        rescale = layer._output_partition_rescale.data
        assert rescale.shape == (8,)
        assert torch.allclose(rescale[:3], torch.tensor([1.0, 1.0, 1.0]))
        assert torch.allclose(rescale[3:], torch.tensor([0.5] * 5))


class TestUniformTwoPartition:
    """Uniform two-partition layers should NOT get rescaling."""

    def test_no_rescale_when_uniform(self):
        logical_widths = [2, 2]
        layer = _make_layer(logical_widths, [100.0, 100.0])

        fake_output = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
        scheme = CompressedTensorsW4A4Fp4(use_a16=True)
        scheme.kernel = FakeKernel(fake_output)

        scheme.process_weights_after_loading(layer)

        assert layer._output_partition_rescale is None

        x_dummy = torch.randn(1, 16)
        bias = torch.tensor([1.0, 1.0, 1.0, 1.0])
        out = scheme.apply_weights(layer, x_dummy, bias=bias)
        # Kernel receives bias directly
        expected = torch.tensor([[11.0, 21.0, 31.0, 41.0]])
        assert torch.allclose(out, expected)


class TestThreePartitionUnaffected:
    """Three-partition layers (e.g. QKV) should not enter rescale path."""

    def test_qkv_no_rescale(self):
        logical_widths = [4, 2, 2]
        layer = _make_layer(logical_widths, [100.0, 120.0, 140.0])

        fake_output = torch.ones(1, 8) * 10.0
        scheme = CompressedTensorsW4A4Fp4(use_a16=True)
        scheme.kernel = FakeKernel(fake_output)

        scheme.process_weights_after_loading(layer)

        assert layer._output_partition_rescale is None

        # weight_global_scale should use max() = 140.0
        assert torch.allclose(
            layer.weight_global_scale,
            torch.tensor(1.0 / 140.0),
        )
