# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CPU unquantized GEMM dispatch behavior."""

import pytest
import torch

from vllm.model_executor.layers import utils
from vllm.platforms import CpuArchEnum, current_platform


def test_dispatch_prepacks_arm_bf16_causal_conv(monkeypatch):
    monkeypatch.setattr(
        current_platform, "get_cpu_architecture", lambda: CpuArchEnum.ARM
    )
    monkeypatch.setattr(torch.cpu, "get_capabilities", lambda: {"bf16": True})
    monkeypatch.setattr(torch.cpu, "_is_avx512_bf16_supported", lambda: False)

    packed = torch.randn(32, 4, dtype=torch.bfloat16)
    pack_inputs = []

    def pack(weight):
        pack_inputs.append(weight.clone())
        return packed

    monkeypatch.setattr(utils.ops, "causal_conv1d_weight_pack", pack)
    original = torch.randn(32, 1, 4, dtype=torch.bfloat16)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(original.clone(), requires_grad=False)

    utils.dispatch_cpu_unquantized_gemm(layer, remove_weight=False)

    assert len(pack_inputs) == 1
    # Check causal_conv1d_weight_pack received the expected weights
    torch.testing.assert_close(pack_inputs[0], original.view(32, 4))
    # Check we have correctly stashed the unpacked weights
    torch.testing.assert_close(layer._cpu_unpacked_conv_weight, original.view(32, 4))
    # Check we have correctly stored the packed weights
    assert layer.weight.data_ptr() == packed.data_ptr()


def test_dispatch_does_not_pack_3d_expert_weight(monkeypatch):
    monkeypatch.setattr(
        current_platform, "get_cpu_architecture", lambda: CpuArchEnum.ARM
    )
    monkeypatch.setattr(torch.cpu, "get_capabilities", lambda: {"bf16": True})
    monkeypatch.setattr(torch.cpu, "_is_avx512_bf16_supported", lambda: False)
    pack_calls = []
    monkeypatch.setattr(
        utils.ops,
        "causal_conv1d_weight_pack",
        lambda weight: pack_calls.append(weight),
    )

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.randn(8, 32, 4, dtype=torch.bfloat16), requires_grad=False
    )
    utils.dispatch_cpu_unquantized_gemm(layer, remove_weight=False)
    assert pack_calls == []


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


@pytest.mark.usefixtures("_mock_zentorch_linear_unary")
def test_dispatch_cpu_unquantized_gemm_logs_zentorch_dispatch(monkeypatch):
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)
    expected_prepacked = bool(utils.envs.VLLM_ZENTORCH_WEIGHT_PREPACK) and hasattr(
        torch.ops.zentorch, "zentorch_weight_prepack_for_linear"
    )

    log_calls = []
    monkeypatch.setattr(
        utils.logger, "debug_once", lambda *args: log_calls.append(args)
    )

    layer = torch.nn.Linear(16, 8, bias=True)
    utils.dispatch_cpu_unquantized_gemm(layer, remove_weight=False)

    assert log_calls == [
        (
            "CPU unquantized GEMM dispatch: using zentorch_linear_unary (prepacked=%s)",
            expected_prepacked,
        )
    ]
