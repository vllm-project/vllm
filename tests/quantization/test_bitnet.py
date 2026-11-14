# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only correctness tests for the Stage 1 BitNet ternary quantization
path (see vllm/model_executor/layers/quantization/bitnet.py).

These tests exercise the pack/unpack round-trip and the linear apply()
path directly with plain tensors -- they do not build a full vLLM engine
or load a real checkpoint, so they run on CPU without a GPU and without
downloading any model. End-to-end validation against a real checkpoint is
tracked as follow-up work (see the module docstring).
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.bitnet import (
    BITNET_VALUES_PER_BYTE,
    bitnet_dequantize_weight,
    bitnet_linear,
    bitnet_quantize_weight,
)


def test_pack_unpack_round_trip_is_ternary():
    torch.manual_seed(0)
    out_features, in_features = 17, 37  # deliberately not a multiple of 4
    weight = torch.randn(out_features, in_features)

    packed, scale = bitnet_quantize_weight(weight)

    expected_packed_cols = -(-in_features // BITNET_VALUES_PER_BYTE)
    assert packed.dtype == torch.uint8
    assert packed.shape == (out_features, expected_packed_cols)
    assert torch.isclose(scale, weight.abs().mean())

    dequantized = bitnet_dequantize_weight(packed, scale, out_features, in_features)
    assert dequantized.shape == weight.shape

    # Every dequantized entry must be exactly -scale, 0, or +scale.
    allowed = torch.stack([-scale, torch.zeros_like(scale), scale]).view(-1, 1, 1)
    close_to_allowed = torch.isclose(
        dequantized.unsqueeze(0), allowed, atol=1e-6
    ).any(dim=0)
    assert bool(close_to_allowed.all())


def test_quantize_weight_rejects_non_2d_input():
    bad_weight = torch.randn(4, 4, 4)
    with pytest.raises(ValueError):
        bitnet_quantize_weight(bad_weight)


def test_bitnet_linear_matches_manual_dequantized_matmul():
    torch.manual_seed(1)
    batch, in_features, out_features = 5, 23, 11
    weight = torch.randn(out_features, in_features)
    bias = torch.randn(out_features)
    x = torch.randn(batch, in_features)

    packed, scale = bitnet_quantize_weight(weight)
    actual = bitnet_linear(x, packed, scale, out_features, in_features, bias)

    reference_weight = bitnet_dequantize_weight(
        packed, scale, out_features, in_features
    )
    expected = x @ reference_weight.T + bias

    assert torch.allclose(actual, expected, atol=1e-5)


def test_bitnet_linear_without_bias():
    torch.manual_seed(2)
    batch, in_features, out_features = 3, 9, 6
    weight = torch.randn(out_features, in_features)
    x = torch.randn(batch, in_features)

    packed, scale = bitnet_quantize_weight(weight)
    out = bitnet_linear(x, packed, scale, out_features, in_features, bias=None)

    assert out.shape == (batch, out_features)
