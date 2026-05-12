# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CPU unquantized GEMM dispatch via the w16a16 kernel abstraction."""

import pytest
import torch

import vllm.model_executor.kernels.linear.base.w16a16 as w16a16
import vllm.model_executor.kernels.linear.zentorch.w16a16 as zentorch_w16a16
from vllm.model_executor.kernels.linear import choose_w16a16_kernel
from vllm.platforms import PlatformEnum, current_platform


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


@pytest.fixture
def _force_zen_cpu(monkeypatch):
    """Make `current_platform` look like a Zen CPU for dispatch selection."""
    monkeypatch.setattr(current_platform, "_enum", PlatformEnum.CPU)
    monkeypatch.setattr(current_platform, "is_cpu", lambda: True)
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)


def _config(layer: torch.nn.Linear, *, clear_weight: bool) -> w16a16.Config:
    return w16a16.Config(
        weight_dtype=layer.weight.dtype,
        weight_shape=tuple(layer.weight.shape),
        clear_weight_after_processing=clear_weight,
    )


@pytest.mark.usefixtures("_mock_zentorch_linear_unary", "_force_zen_cpu")
def test_choose_w16a16_kernel_uses_zentorch_on_zen():
    layer = torch.nn.Linear(16, 8, bias=True)
    x = torch.randn(4, 16)
    expected = torch.nn.functional.linear(x, layer.weight, layer.bias)

    kernel = choose_w16a16_kernel(_config(layer, clear_weight=False))
    assert isinstance(kernel, zentorch_w16a16.Kernel)

    kernel.process_weights_after_loading(layer)
    output = kernel.apply_weights(layer, x, layer.bias)

    torch.testing.assert_close(output, expected)


@pytest.mark.usefixtures("_mock_zentorch_linear_unary", "_force_zen_cpu")
def test_zentorch_kernel_clears_weight_when_configured():
    layer = torch.nn.Linear(16, 8, bias=True)

    kernel = choose_w16a16_kernel(_config(layer, clear_weight=True))
    kernel.process_weights_after_loading(layer)

    assert layer.weight.numel() == 0
