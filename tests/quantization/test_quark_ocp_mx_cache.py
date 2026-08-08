# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.model_executor.kernels.linear.mxfp4.emulation import (
    EmulationMxfp4LinearKernel,
)
from vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx import (
    QuarkOCP_MX,
)


def _make_method(cache: bool, dtype: torch.dtype = torch.bfloat16):
    method = object.__new__(QuarkOCP_MX)
    method.cache_dequant_weight = cache
    method.dynamic_mxfp4_quant = False
    method._dequant_dtype = dtype
    kernel = object.__new__(EmulationMxfp4LinearKernel)
    kernel.quant_dequant_func = MagicMock(side_effect=lambda x: x)
    method.ocp_mx_linear = kernel
    return method


def _make_layer():
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight", torch.nn.Parameter(torch.ones((2, 2), dtype=torch.uint8), False)
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(torch.ones((2, 1), dtype=torch.uint8), False),
    )
    return layer


@pytest.mark.parametrize(("env_value", "expected"), [("0", False), ("1", True)])
def test_cache_opt_in(monkeypatch, env_value, expected):
    monkeypatch.setenv("VLLM_QUARK_MX_CACHE_DEQUANT_WEIGHT", env_value)
    method = QuarkOCP_MX(
        weight_quant_spec={"dtype": "fp4"},
        input_quant_spec={"dtype": "fp4", "is_dynamic": True},
    )

    assert method.cache_dequant_weight is expected


def test_cache_disabled_preserves_kernel_path():
    method = _make_method(cache=False)
    layer = _make_layer()
    method.ocp_mx_linear.process_weights_after_loading = MagicMock()

    method.process_weights_after_loading(layer)

    method.ocp_mx_linear.process_weights_after_loading.assert_called_once_with(layer)
    assert layer.weight_scale is not None


def test_cache_dequantizes_once_and_releases_scale():
    method = _make_method(cache=True)
    layer = _make_layer()
    dequantized = torch.ones((2, 4), dtype=torch.bfloat16)

    with patch(
        "vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx.dequant_mxfp4",
        return_value=dequantized,
    ) as dequant:
        method.process_weights_after_loading(layer)
        method.process_weights_after_loading(layer)

    dequant.assert_called_once()
    assert layer.weight_scale is None
    assert layer.weight.dtype == torch.bfloat16
    assert layer.weight.device == dequantized.device
    assert not layer.weight.requires_grad


def test_cached_output_matches_uncached_and_keeps_activation_qdq():
    cached = _make_method(cache=True)
    uncached = _make_method(cache=False)
    cached.ocp_mx_linear.quant_dequant_func.side_effect = lambda x: x * 0.5
    uncached.ocp_mx_linear.quant_dequant_func.side_effect = lambda x: x * 0.5
    cached_layer = _make_layer()
    uncached_layer = _make_layer()
    dequantized = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    x = torch.tensor([[2.0, 4.0]], dtype=torch.float32)

    with (
        patch(
            "vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx.dequant_mxfp4",
            return_value=dequantized,
        ),
        patch(
            "vllm.model_executor.kernels.linear.mxfp4.emulation.dequant_mxfp4",
            return_value=dequantized,
        ),
    ):
        cached.process_weights_after_loading(cached_layer)
        actual = cached.apply_weights(cached_layer, x)
        expected = uncached.apply_weights(uncached_layer, x)

    torch.testing.assert_close(actual, expected)
    cached.ocp_mx_linear.quant_dequant_func.assert_called_once_with(x)
    uncached.ocp_mx_linear.quant_dequant_func.assert_called_once_with(x)
