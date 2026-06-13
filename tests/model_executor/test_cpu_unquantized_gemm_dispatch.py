# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CPU unquantized GEMM dispatch behavior."""

import pytest
import torch

from vllm.model_executor.layers import utils
from vllm.platforms import current_platform


@pytest.fixture(scope="module")
def _mock_zentorch_linear_unary():
    """Register a mock zentorch_linear_unary op when zentorch is not installed.

    Allows the dispatch tests to run in CI without a real zentorch build.
    Skips registration when zentorch is already available.
    """
    if hasattr(torch.ops.zentorch, "zentorch_linear_unary"):
        yield
        return

    lib_def = torch.library.Library("zentorch", "DEF")
    lib_def.define(
        "zentorch_linear_unary("
        "Tensor input, "
        "Tensor weight, "
        "Tensor? bias, "
        "bool is_weight_prepacked=False"
        ") -> Tensor"
    )

    lib_impl = torch.library.Library("zentorch", "IMPL", "CPU")
    lib_impl.impl(
        "zentorch_linear_unary",
        lambda input, weight, bias, is_weight_prepacked=False: (
            torch.nn.functional.linear(input, weight, bias)
        ),
    )

    yield

    lib_impl._destroy()
    lib_def._destroy()


@pytest.mark.usefixtures("_mock_zentorch_linear_unary")
def test_dispatch_cpu_unquantized_gemm_uses_zentorch_on_zen(monkeypatch):
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)

    layer = torch.nn.Linear(16, 8, bias=True)
    x = torch.randn(4, 16)
    expected = torch.nn.functional.linear(x, layer.weight, layer.bias)

    utils.dispatch_cpu_unquantized_gemm(layer, remove_weight=False)
    output = layer.cpu_linear(x, layer.weight, layer.bias)

    torch.testing.assert_close(output, expected)


@pytest.mark.usefixtures("_mock_zentorch_linear_unary")
def test_dispatch_cpu_unquantized_gemm_zen_remove_weight(monkeypatch):
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)

    layer = torch.nn.Linear(16, 8, bias=True)
    utils.dispatch_cpu_unquantized_gemm(layer, remove_weight=True)

    assert layer.weight.numel() == 0
