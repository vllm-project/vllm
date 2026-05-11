# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the W8A8 zentorch dispatch path in
``CPUInt8ScaledMMLinearKernel``.

Mirrors the TorchAO DA8W8 zentorch tests in
``tests/quantization/test_torchao.py`` but exercises the LLM-Compressor
W8A8 dynamic-symmetric route in
``vllm/model_executor/kernels/linear/scaled_mm/cpu.py``.

Coverage:
* ``_zentorch_w8a8_eligible``       — eligibility predicate (Zen + op + scheme).
* ``_process_weights_for_zentorch`` — weight/scale prep, fused-module expansion.
* ``_apply_weights_zentorch``       — op dispatch + bias pass-through.
* ``process_weights_after_loading`` — fall-through to oneDNN/sgl when ineligible.

Run ``pytest tests/kernels/quantization/test_cpu_int8_zentorch_dispatch.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from tests.quantization._zentorch_helpers import zentorch_ops_mock  # noqa: F401
from vllm.model_executor.kernels.linear.scaled_mm import (
    CPUInt8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)
from vllm.platforms import current_platform

# Zen CPU dispatch unit tests for compressed-tensors W8A8 dynamic-symmetric.


_LAYER_PARAM_NAMES: Sequence[str] = (
    "weight",  # w_q
    "weight_scale",  # w_s
    "input_scale",  # i_s
    "input_zero_point",  # i_zp
    "azp_adj",  # azp_adj
)


def _make_kernel(
    *,
    is_static: bool = False,
    is_channelwise: bool = True,
    input_symmetric: bool = True,
) -> CPUInt8ScaledMMLinearKernel:
    """Construct a kernel without invoking ``is_supported``/``can_implement``
    assertions, which gate on ``current_platform.is_cpu()``.
    """
    kernel = CPUInt8ScaledMMLinearKernel.__new__(CPUInt8ScaledMMLinearKernel)
    kernel.config = Int8ScaledMMLinearLayerConfig(
        is_static_input_scheme=is_static,
        is_channelwise=is_channelwise,
        input_symmetric=input_symmetric,
    )
    kernel.layer_param_names = _LAYER_PARAM_NAMES
    return kernel


def _make_layer(
    *,
    n: int,
    k: int,
    weight_scale: torch.Tensor,
    logical_widths: list[int] | None = None,
) -> torch.nn.Module:
    """Build a minimal ``nn.Module`` that mimics what
    ``CPUInt8ScaledMMLinearKernel.process_weights_after_loading`` reads.

    A real ``nn.Module`` (not ``SimpleNamespace``) is required because
    ``replace_parameter`` calls ``layer.register_parameter``.
    """
    layer = torch.nn.Module()
    weight = torch.randint(-128, 127, (n, k), dtype=torch.int8)
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(weight, requires_grad=False),
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(weight_scale, requires_grad=False),
    )
    layer.logical_widths = logical_widths if logical_widths is not None else [n]
    return layer


def _layer_ready_for_apply(
    n: int, k: int, *, with_bias: bool = False
) -> torch.nn.Module:
    """A layer pre-populated as if ``_process_weights_for_zentorch`` already
    ran successfully (int8 weight + bf16 (N,) scale)."""
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(
            torch.randint(-128, 127, (n, k), dtype=torch.int8),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(torch.randn(n, dtype=torch.bfloat16), requires_grad=False),
    )
    layer.bias = torch.randn(n, dtype=torch.bfloat16) if with_bias else None
    return layer


# ----- _zentorch_w8a8_eligible: predicate ----------------------------------


def test_eligible_predicate_per_channel_dynamic_symmetric_returns_true(
    monkeypatch,
    zentorch_ops_mock,  # noqa: F811
):
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)

    layer = _make_layer(n=8, k=16, weight_scale=torch.randn(8, dtype=torch.float32))
    kernel = _make_kernel(is_static=False, is_channelwise=True, input_symmetric=True)
    assert kernel._zentorch_w8a8_eligible(layer) is True


def test_eligible_predicate_fused_per_tensor_returns_false(
    monkeypatch,
    zentorch_ops_mock,  # noqa: F811
):
    """Fused module + per-tensor weight scales fall back to oneDNN: zentorch's
    per-channel kernel adds no benefit over oneDNN's per-tensor path."""
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)

    layer = _make_layer(
        n=12,
        k=16,
        weight_scale=torch.tensor([0.1, 0.2, 0.4]).reshape(-1, 1),
        logical_widths=[4, 4, 4],
    )
    kernel = _make_kernel(is_static=False, is_channelwise=False, input_symmetric=True)
    assert kernel._zentorch_w8a8_eligible(layer) is False


def test_eligible_predicate_off_zen_returns_false(monkeypatch):
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: False)

    layer = _make_layer(n=8, k=16, weight_scale=torch.randn(8, dtype=torch.float32))
    kernel = _make_kernel(is_static=False, is_channelwise=True, input_symmetric=True)
    assert kernel._zentorch_w8a8_eligible(layer) is False


def test_eligible_predicate_op_missing_returns_false(monkeypatch):
    """Zen CPU but ``zentorch_dynamic_qlinear`` is not registered."""
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)
    if hasattr(torch.ops, "zentorch") and hasattr(
        torch.ops.zentorch, "zentorch_dynamic_qlinear"
    ):
        pytest.skip(
            "real zentorch build registers the op; can't test the missing-op "
            "fallback hermetically here"
        )

    layer = _make_layer(n=8, k=16, weight_scale=torch.randn(8, dtype=torch.float32))
    kernel = _make_kernel(is_static=False, is_channelwise=True, input_symmetric=True)
    assert kernel._zentorch_w8a8_eligible(layer) is False


def test_eligible_predicate_static_input_scheme_returns_false(
    monkeypatch,
    zentorch_ops_mock,  # noqa: F811
):
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)

    layer = _make_layer(n=8, k=16, weight_scale=torch.randn(8, dtype=torch.float32))
    kernel = _make_kernel(is_static=True, is_channelwise=True, input_symmetric=True)
    assert kernel._zentorch_w8a8_eligible(layer) is False


def test_eligible_predicate_asymmetric_input_returns_false(
    monkeypatch,
    zentorch_ops_mock,  # noqa: F811
):
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)

    layer = _make_layer(n=8, k=16, weight_scale=torch.randn(8, dtype=torch.float32))
    kernel = _make_kernel(is_static=False, is_channelwise=True, input_symmetric=False)
    assert kernel._zentorch_w8a8_eligible(layer) is False


def test_eligible_predicate_non_fused_per_tensor_returns_false(
    monkeypatch,
    zentorch_ops_mock,  # noqa: F811
):
    """Non-fused per-tensor cannot be reduced to a (N,) scale."""
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)

    layer = _make_layer(
        n=8,
        k=16,
        weight_scale=torch.tensor(0.123, dtype=torch.float32),
        logical_widths=[8],
    )
    kernel = _make_kernel(is_static=False, is_channelwise=False, input_symmetric=True)
    assert kernel._zentorch_w8a8_eligible(layer) is False


# ----- process_weights_after_loading: success paths ------------------------


def test_process_weights_after_loading_per_channel_caches_zentorch_attrs(
    monkeypatch,
    zentorch_ops_mock,  # noqa: F811
):
    """Dynamic-symmetric, per-channel, non-fused module → zentorch path."""
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)

    n, k = 8, 16
    layer = _make_layer(
        n=n,
        k=k,
        weight_scale=torch.randn(n, dtype=torch.float32),
        logical_widths=[n],
    )

    kernel = _make_kernel(is_static=False, is_channelwise=True, input_symmetric=True)
    kernel.process_weights_after_loading(layer)

    assert kernel.linear_method == kernel._apply_weights_zentorch
    assert layer.weight.shape == (n, k)
    assert layer.weight.is_contiguous()
    assert layer.weight_scale.shape == (n,)
    assert layer.weight_scale.dtype == torch.bfloat16
    assert layer.weight_scale.is_contiguous()
    assert getattr(layer, "_zentorch_processed_weights", False) is True
    assert getattr(layer, "_zentorch_kind", None) == "compressed_tensors_w8a8"


def test_process_weights_after_loading_fused_per_channel_squeezes_n1_scale(
    monkeypatch,
    zentorch_ops_mock,  # noqa: F811
):
    """Fused module + per-channel strategy → ``(N, 1)`` scale (the shape
    ``ChannelQuantScaleParameter`` emits in compressed-tensors) is squeezed
    to ``(N,)``."""
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)

    logical_widths = [4, 4]
    n = sum(logical_widths)
    k = 16
    scale_fp32 = torch.linspace(0.1, 0.8, n, dtype=torch.float32)

    layer = _make_layer(
        n=n,
        k=k,
        weight_scale=scale_fp32.reshape(n, 1).clone(),
        logical_widths=logical_widths,
    )

    kernel = _make_kernel(is_static=False, is_channelwise=True, input_symmetric=True)
    kernel.process_weights_after_loading(layer)

    assert kernel.linear_method == kernel._apply_weights_zentorch
    assert getattr(layer, "_zentorch_processed_weights", False) is True
    assert layer.weight_scale.shape == (n,)
    assert layer.weight_scale.dtype == torch.bfloat16
    torch.testing.assert_close(layer.weight_scale.data, scale_fp32.to(torch.bfloat16))


# ----- process_weights_after_loading: fall-through paths preserve weight ---


def _stub_non_zentorch_paths(monkeypatch):
    """Replace oneDNN/SGL processing with spies so we don't need their
    backing C++ kernels at test time."""
    calls: dict[str, int] = {"onednn": 0, "sgl": 0}

    def _onednn(self, layer):
        calls["onednn"] += 1

    def _sgl(self, layer):
        calls["sgl"] += 1

    monkeypatch.setattr(
        CPUInt8ScaledMMLinearKernel, "process_weights_for_onednn", _onednn
    )
    monkeypatch.setattr(CPUInt8ScaledMMLinearKernel, "process_weights_for_sgl", _sgl)
    return calls


def test_process_weights_after_loading_off_zen_falls_through_to_onednn(
    monkeypatch,
):
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: False)
    calls = _stub_non_zentorch_paths(monkeypatch)

    n, k = 8, 16
    layer = _make_layer(n=n, k=k, weight_scale=torch.randn(n, dtype=torch.float32))

    kernel = _make_kernel(is_static=False, is_channelwise=True, input_symmetric=True)
    kernel.process_weights_after_loading(layer)

    assert kernel.linear_method != kernel._apply_weights_zentorch
    assert calls["onednn"] + calls["sgl"] == 1
    # Original int8 weight dtype is preserved by the fall-through path.
    assert layer.weight.dtype == torch.int8
    assert getattr(layer, "_zentorch_processed_weights", False) is False


def test_process_weights_after_loading_op_missing_falls_through_to_onednn(
    monkeypatch,
):
    """Zen CPU but ``zentorch_dynamic_qlinear`` is not registered."""
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)
    calls = _stub_non_zentorch_paths(monkeypatch)

    n, k = 8, 16
    layer = _make_layer(n=n, k=k, weight_scale=torch.randn(n, dtype=torch.float32))

    kernel = _make_kernel(is_static=False, is_channelwise=True, input_symmetric=True)

    if hasattr(torch.ops, "zentorch") and hasattr(
        torch.ops.zentorch, "zentorch_dynamic_qlinear"
    ):
        pytest.skip(
            "real zentorch build registers the op; can't test the missing-op "
            "fallback hermetically here"
        )

    kernel.process_weights_after_loading(layer)

    assert kernel.linear_method != kernel._apply_weights_zentorch
    assert calls["onednn"] + calls["sgl"] == 1


def test_process_weights_after_loading_static_input_falls_through_to_onednn(
    monkeypatch,
    zentorch_ops_mock,  # noqa: F811
):
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)
    calls = _stub_non_zentorch_paths(monkeypatch)

    n, k = 8, 16
    layer = _make_layer(n=n, k=k, weight_scale=torch.randn(n, dtype=torch.float32))

    kernel = _make_kernel(is_static=True, is_channelwise=True, input_symmetric=True)
    kernel.process_weights_after_loading(layer)

    assert kernel.linear_method != kernel._apply_weights_zentorch
    assert calls["onednn"] + calls["sgl"] == 1


def test_process_weights_after_loading_asymmetric_input_falls_through_to_onednn(
    monkeypatch,
    zentorch_ops_mock,  # noqa: F811
):
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)
    calls = _stub_non_zentorch_paths(monkeypatch)

    n, k = 8, 16
    layer = _make_layer(n=n, k=k, weight_scale=torch.randn(n, dtype=torch.float32))

    kernel = _make_kernel(is_static=False, is_channelwise=True, input_symmetric=False)
    kernel.process_weights_after_loading(layer)

    assert kernel.linear_method != kernel._apply_weights_zentorch
    assert calls["onednn"] + calls["sgl"] == 1


def test_process_weights_after_loading_non_fused_per_tensor_falls_through_to_onednn(
    monkeypatch,
    zentorch_ops_mock,  # noqa: F811
):
    """Per-tensor weight quantization is rejected by the predicate (zentorch
    is per-channel only); the layer falls through to oneDNN."""
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)
    calls = _stub_non_zentorch_paths(monkeypatch)

    n, k = 8, 16
    layer = _make_layer(
        n=n,
        k=k,
        weight_scale=torch.tensor(0.123, dtype=torch.float32),
        logical_widths=[n],
    )

    kernel = _make_kernel(is_static=False, is_channelwise=False, input_symmetric=True)
    kernel.process_weights_after_loading(layer)

    assert kernel.linear_method != kernel._apply_weights_zentorch
    assert calls["onednn"] + calls["sgl"] == 1


def test_process_weights_after_loading_fused_per_tensor_falls_through_to_onednn(
    monkeypatch,
    zentorch_ops_mock,  # noqa: F811
):
    """Fused module quantized per-tensor is also rejected: zentorch's
    per-channel kernel offers no benefit when the scale is constant within
    each logical sub-module, so the layer falls through to oneDNN."""
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)
    calls = _stub_non_zentorch_paths(monkeypatch)

    logical_widths = [4, 4, 8]
    n = sum(logical_widths)
    k = 16
    per_tensor_scales = torch.tensor([0.1, 0.2, 0.4], dtype=torch.float32).reshape(
        -1, 1
    )

    layer = _make_layer(
        n=n,
        k=k,
        weight_scale=per_tensor_scales,
        logical_widths=logical_widths,
    )

    kernel = _make_kernel(is_static=False, is_channelwise=False, input_symmetric=True)
    kernel.process_weights_after_loading(layer)

    assert kernel.linear_method != kernel._apply_weights_zentorch
    assert calls["onednn"] + calls["sgl"] == 1


# ----- apply(): zentorch dispatch ------------------------------------------


def test_apply_dispatches_to_zentorch_dynamic_qlinear(
    monkeypatch,
    zentorch_ops_mock,  # noqa: F811
):
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)

    batch, k, n = 4, 16, 8
    layer = _layer_ready_for_apply(n=n, k=k)
    kernel = _make_kernel(is_static=False, is_channelwise=True, input_symmetric=True)
    kernel.linear_method = kernel._apply_weights_zentorch

    captured: dict = {}

    def spy(
        inp,
        weight,
        weight_scales,
        bias=None,
        zentorch_op_name="zentorch::zentorch_dynamic_qlinear",
    ):
        captured["called"] = True
        captured["bias"] = bias
        captured["op_name"] = zentorch_op_name
        return torch.zeros(inp.shape[:-1] + (weight.shape[0],), dtype=inp.dtype)

    monkeypatch.setattr(torch.ops.zentorch, "zentorch_dynamic_qlinear", spy)

    x = torch.randn(batch, k, dtype=torch.bfloat16)
    out = kernel.apply_weights(layer, x)

    assert captured.get("called") is True
    assert captured["bias"] is None
    assert captured["op_name"] == "zentorch::zentorch_dynamic_qlinear"
    assert out.shape == (batch, n)


def test_apply_passes_bias_through_to_zentorch_dynamic_qlinear(
    monkeypatch,
    zentorch_ops_mock,  # noqa: F811
):
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)

    batch, k, n = 4, 16, 8
    layer = _layer_ready_for_apply(n=n, k=k)
    kernel = _make_kernel(is_static=False, is_channelwise=True, input_symmetric=True)
    kernel.linear_method = kernel._apply_weights_zentorch

    captured: dict = {}

    def spy(
        inp,
        weight,
        weight_scales,
        bias=None,
        zentorch_op_name="zentorch::zentorch_dynamic_qlinear",
    ):
        captured["bias"] = bias
        captured["op_name"] = zentorch_op_name
        return torch.zeros(inp.shape[:-1] + (weight.shape[0],), dtype=inp.dtype)

    monkeypatch.setattr(torch.ops.zentorch, "zentorch_dynamic_qlinear", spy)

    x = torch.randn(batch, k, dtype=torch.bfloat16)
    bias = torch.randn(n, dtype=torch.bfloat16)

    kernel.apply_weights(layer, x, bias=bias)

    assert captured["bias"] is bias
    assert captured["op_name"] == "zentorch::zentorch_dynamic_qlinear"


if __name__ == "__main__":
    pytest.main([__file__])
