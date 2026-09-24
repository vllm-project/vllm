# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for dequantizing block-quantized FP8 weights back to bf16.

Layers that consume their weight directly instead of through ``apply_weights``
(such as DeepSeek-V4's ``o_proj``, which sets ``layer.is_bmm`` and
``bmm_batch_size``) cannot use a kernel that repacks the weight or rewrites the
block scales. These tests cover both the helper and the guard in
``Fp8LinearMethod.process_weights_after_loading`` that decides to use it.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    dequantize_fp8_block_weight_to_bf16,
)

# Importing the FP8 linear method pulls in vllm's compiled CUDA extensions, which
# are absent in a source-only checkout. Only the guard tests below need it; the
# helper tests run everywhere.
try:
    from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod

    HAS_FP8_LINEAR_METHOD = True
except ImportError:  # pragma: no cover - depends on the build
    Fp8LinearMethod = None
    HAS_FP8_LINEAR_METHOD = False

requires_compiled_extensions = pytest.mark.skipif(
    not HAS_FP8_LINEAR_METHOD, reason="requires the compiled vLLM extensions"
)


def _block_weight(shape, block_size, scale=None, dtype=torch.float32):
    block_n, block_k = block_size
    n, k = shape
    weight = (torch.randn(n, k) * 4).to(torch.float8_e4m3fn)
    if scale is None:
        scale = torch.rand(n // block_n, k // block_k, dtype=dtype) * 0.5 + 0.25
    return weight, scale


# --------------------------------------------------------------------------
# the helper
# --------------------------------------------------------------------------


@pytest.mark.parametrize("block_size", [(128, 128), (128, 64), (1, 32)])
def test_dequantize_matches_manual_expansion(block_size):
    block_n, block_k = block_size
    torch.manual_seed(0)
    weight, scale = _block_weight((2 * block_n, 3 * block_k), block_size)

    out = dequantize_fp8_block_weight_to_bf16(weight, scale, block_size)

    assert out.shape == weight.shape
    assert out.dtype == torch.bfloat16

    expected = (
        weight.to(torch.float32)
        * scale.to(torch.float32)
        .repeat_interleave(block_n, dim=0)
        .repeat_interleave(block_k, dim=1)
    ).to(torch.bfloat16)
    torch.testing.assert_close(out.float(), expected.float(), rtol=0, atol=0)


def test_dequantize_accepts_e8m0_scales():
    """DeepSeek-V4-Flash stores block scales as e8m0 by default."""
    weight, scale = _block_weight((128, 256), (128, 128), dtype=torch.float32)
    # 2**-1 and 2**-2, exactly representable in e8m0.
    scale = torch.tensor([[0.5, 0.25]], dtype=torch.float32)
    e8m0 = scale.to(torch.float8_e8m0fnu)

    out = dequantize_fp8_block_weight_to_bf16(weight, e8m0, (128, 128))
    expected = dequantize_fp8_block_weight_to_bf16(weight, scale, (128, 128))

    torch.testing.assert_close(out.float(), expected.float(), rtol=0, atol=0)


def test_dequantize_scale_is_a_multiplier_not_a_reciprocal():
    """The block scale is a scale; its ``_inv`` name is legacy.

    A weight quantized as ``w / s`` must come back as approximately ``w``.
    """
    s = 0.5
    quantized = (torch.full((128, 128), 2.0) / s).to(torch.float8_e4m3fn)
    scale = torch.full((1, 1), s, dtype=torch.float32)

    out = dequantize_fp8_block_weight_to_bf16(quantized, scale, (128, 128))

    torch.testing.assert_close(
        out.float(), torch.full((128, 128), 2.0), rtol=0.05, atol=0.05
    )


@pytest.mark.parametrize(
    "weight,scale,block_size",
    [
        # not an fp8 weight
        (
            torch.zeros(128, 128, dtype=torch.bfloat16),
            torch.ones(1, 1),
            (128, 128),
        ),
        # scale grid does not match the weight
        (
            torch.zeros(128, 128, dtype=torch.float8_e4m3fn),
            torch.ones(2, 2),
            (128, 128),
        ),
        # shape not divisible by the block
        (
            torch.zeros(130, 128, dtype=torch.float8_e4m3fn),
            torch.ones(1, 1),
            (128, 128),
        ),
        # a scale container we refuse to guess about
        (
            torch.zeros(128, 128, dtype=torch.float8_e4m3fn),
            torch.ones(1, 1, dtype=torch.uint8),
            (128, 128),
        ),
    ],
)
def test_dequantize_rejects_bad_inputs(weight, scale, block_size):
    with pytest.raises(ValueError):
        dequantize_fp8_block_weight_to_bf16(weight, scale, block_size)


# --------------------------------------------------------------------------
# the guard in Fp8LinearMethod
# --------------------------------------------------------------------------


class _FakeMarlinKernel:
    """Stands in for MarlinFP8ScaledMMLinearKernel and records its calls."""

    def __init__(self):
        self.called = False

    def process_weights_after_loading(self, layer):
        self.called = True
        # What Marlin would do: repack and rename the scales.
        layer.weight = torch.nn.Parameter(
            torch.zeros(4, dtype=torch.int32), requires_grad=False
        )
        layer.weight_scale_inv = torch.nn.Parameter(torch.ones(1))


def _make_method(block_size=(128, 128)):
    """Build an Fp8LinearMethod without running its heavy __init__."""
    method = Fp8LinearMethod.__new__(Fp8LinearMethod)
    method.use_marlin = True
    method.block_quant = True
    method.weight_block_size = block_size
    method.marlin_input_dtype = None
    method.fp8_linear = _FakeMarlinKernel()
    return method


def _make_direct_weight_layer(block_size=(128, 128)):
    block_n, block_k = block_size
    weight, scale = _block_weight((2 * block_n, 2 * block_k), block_size)
    layer = SimpleNamespace(
        weight=torch.nn.Parameter(weight, requires_grad=False),
        weight_scale_inv=torch.nn.Parameter(scale, requires_grad=False),
        weight_block_size=list(block_size),
        is_bmm=True,
    )
    return layer, weight, scale


@requires_compiled_extensions
def test_guard_dequantizes_and_skips_marlin():
    block_size = (128, 128)
    method = _make_method(block_size)
    layer, weight, scale = _make_direct_weight_layer(block_size)

    method.process_weights_after_loading(layer)

    assert layer.weight.dtype == torch.bfloat16
    assert layer.input_scale is None
    # Marlin must not have touched the weight.
    assert method.fp8_linear.called is False
    torch.testing.assert_close(
        layer.weight.float(),
        dequantize_fp8_block_weight_to_bf16(weight, scale, block_size).float(),
        rtol=0,
        atol=0,
    )


@requires_compiled_extensions
def test_guard_leaves_other_layers_to_marlin():
    """A layer that is not a direct consumer keeps the previous behaviour."""
    block_size = (128, 128)
    method = _make_method(block_size)
    layer, _, _ = _make_direct_weight_layer(block_size)
    layer.is_bmm = False

    method.process_weights_after_loading(layer)

    assert method.fp8_linear.called is True
