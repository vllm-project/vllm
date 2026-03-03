"""Tests for CPU unquantized GEMM dispatch behavior."""

import torch

from vllm.model_executor.layers import utils
from vllm.platforms import current_platform

_TEST_ZENTORCH_LIBS: list[torch.library.Library] = []


def _ensure_test_zentorch_linear_unary() -> None:
    """Register a mock zentorch_linear_unary op when zentorch is not installed.

    Allows the dispatch tests to run in CI without a real zentorch build.
    Skips registration when zentorch is already available.
    """
    if hasattr(torch.ops.zentorch, "zentorch_linear_unary"):
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
        lambda input, weight, bias, is_weight_prepacked=False: torch.nn.functional.linear(
            input, weight, bias
        ),
    )

    _TEST_ZENTORCH_LIBS.extend((lib_def, lib_impl))


def test_dispatch_cpu_unquantized_gemm_uses_zentorch_on_zen(monkeypatch):
    _ensure_test_zentorch_linear_unary()
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)

    layer = torch.nn.Linear(16, 8, bias=True)
    x = torch.randn(4, 16)
    expected = torch.nn.functional.linear(x, layer.weight, layer.bias)

    utils.dispatch_cpu_unquantized_gemm(layer, remove_weight=False)
    output = layer.cpu_linear(x, layer.weight, layer.bias)

    torch.testing.assert_close(output, expected)


def test_dispatch_cpu_unquantized_gemm_zen_remove_weight(monkeypatch):
    _ensure_test_zentorch_linear_unary()
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)

    layer = torch.nn.Linear(16, 8, bias=True)
    utils.dispatch_cpu_unquantized_gemm(layer, remove_weight=True)

    assert layer.weight.numel() == 0
