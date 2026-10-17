# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for compressed-tensors transform linear method selection (CPU-only)."""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.model_executor.layers.fusion.quant_activation import get_input_quant_key

# Import the config module first so the transform modules below do not trip
# the compressed_tensors <-> transform import cycle.
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501, F401
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A4Fp4,
)
from vllm.model_executor.layers.quantization.compressed_tensors.transform import (
    linear as transform_linear,
)
from vllm.model_executor.layers.quantization.compressed_tensors.transform.linear import (  # noqa: E501
    CompressedTensorsLinearTransformMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.transform.schemes.linear_qutlass_nvfp4 import (  # noqa: E501
    QutlassNvFP4LinearMethod,
)

pytestmark = pytest.mark.cpu_test


def _hadamard_tfm(location: str = "input") -> MagicMock:
    tfm = MagicMock()
    tfm.scheme_name = "v"
    tfm.scheme.head_dim = 16
    tfm.args.location = location
    return tfm


@pytest.mark.parametrize(
    "compute_capability,expect_qutlass",
    [
        ((9, 0), False),
        ((10, 0), True),
        ((10, 3), True),
        ((12, 0), False),
        ((12, 1), False),
    ],
)
def test_qutlass_scheme_requires_sm10x(
    monkeypatch: pytest.MonkeyPatch,
    compute_capability: tuple[int, int],
    expect_qutlass: bool,
) -> None:
    """QuTLASS is only built for SM 10.x; SM12x must fall back to the generic
    transform method even though it satisfies has_device_capability(100)."""
    capability = compute_capability[0] * 10 + compute_capability[1]
    monkeypatch.setattr(
        transform_linear.current_platform,
        "is_device_capability_family",
        lambda family, device_id=0: capability // 10 == family // 10,
    )
    scheme = MagicMock(spec=CompressedTensorsW4A4Fp4)

    method = CompressedTensorsLinearTransformMethod.from_schemes(
        MagicMock(), scheme, {0: _hadamard_tfm()}, {}
    )

    assert isinstance(method, QutlassNvFP4LinearMethod) is expect_qutlass


def _create_transform_layer(
    monkeypatch: pytest.MonkeyPatch,
    input_tfms: dict,
    output_tfms: dict,
) -> torch.nn.Module:
    monkeypatch.setattr(
        transform_linear,
        "HadamardTransform",
        lambda *args, **kwargs: torch.nn.Identity(),
    )
    layer = torch.nn.Module()
    layer.weight_loader = lambda *args, **kwargs: None
    method = CompressedTensorsLinearTransformMethod(
        MagicMock(), input_tfms, output_tfms
    )
    method.create_weights(
        layer,
        input_size_per_partition=64,
        output_partition_sizes=[64],
        input_size=64,
        output_size=64,
        params_dtype=torch.bfloat16,
        weight_loader=layer.weight_loader,
    )
    # Simulate the wrapped W4A4 kernel advertising a pre-quantized input key.
    layer._input_quant_key = "nvfp4"
    return layer


def test_input_transform_hides_input_quant_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """An input transform needs the original activation, so producers must not
    fuse the wrapped kernel's input quantization."""
    layer = _create_transform_layer(monkeypatch, {0: _hadamard_tfm("input")}, {})
    assert get_input_quant_key(layer) is None


def test_output_transform_keeps_input_quant_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _create_transform_layer(monkeypatch, {}, {0: _hadamard_tfm("output")})
    assert get_input_quant_key(layer) == "nvfp4"
