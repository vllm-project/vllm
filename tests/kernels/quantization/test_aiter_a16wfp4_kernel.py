# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.kernels.linear import (
    AiterA16Wfp4LinearKernel,
    MxFp4LinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Dynamic,
)

pytestmark = pytest.mark.cpu_test


def _make_kernel() -> AiterA16Wfp4LinearKernel:
    with patch.object(
        AiterA16Wfp4LinearKernel,
        "is_supported",
        return_value=(True, None),
    ):
        return AiterA16Wfp4LinearKernel(
            MxFp4LinearLayerConfig(activation_quant_key=kMxfp4Dynamic)
        )


def _make_layer(
    *,
    n: int = 256,
    k: int = 256,
    params_dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.params_dtype = params_dtype
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(torch.ones((n, k // 2), dtype=torch.uint8), False),
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(torch.ones((n, k // 32), dtype=torch.uint8), False),
    )
    return layer


def test_is_supported_requires_gfx950_rocm_and_aiter():
    module = "vllm.model_executor.kernels.linear.mxfp4.aiter_a16wfp4"
    with (
        patch(f"{module}.current_platform.is_rocm", return_value=True),
        patch(f"{module}.is_aiter_found_and_supported", return_value=True),
    ):
        assert AiterA16Wfp4LinearKernel.is_supported(95) == (True, None)
        supported, reason = AiterA16Wfp4LinearKernel.is_supported(94)
        assert not supported
        assert "gfx950" in reason

    with patch(f"{module}.current_platform.is_rocm", return_value=False):
        supported, reason = AiterA16Wfp4LinearKernel.is_supported(95)
        assert not supported
        assert "ROCm" in reason


def test_fused_route_prepares_once_flattens_and_restores_bias():
    kernel = _make_kernel()
    kernel.quant_dequant_func = MagicMock(side_effect=lambda x: x * 0.5)
    layer = _make_layer()
    prepared = SimpleNamespace(
        weight=layer.weight.detach().clone(),
        scale=layer.weight_scale.detach().clone(),
        n=256,
        k=256,
    )
    x = torch.full((2, 3, 256), 2.0, dtype=torch.bfloat16)
    bias = torch.arange(256, dtype=torch.bfloat16)

    gemm = MagicMock(
        side_effect=lambda x_2d, weight, scale, n, k: torch.zeros(
            (x_2d.shape[0], n), dtype=x_2d.dtype
        )
    )
    with (
        patch.object(envs, "VLLM_MXFP4_EMULATION_DEQUANT_AT_LOAD", True),
        patch(
            "vllm.model_executor.kernels.linear.mxfp4.aiter_a16wfp4."
            "_prepare_aiter_a16wfp4_weight",
            return_value=prepared,
        ) as prepare,
        patch(
            "torch.ops.vllm.aiter_a16wfp4_gemm",
            gemm,
            create=True,
        ),
    ):
        kernel.process_weights_after_loading(layer)
        actual = kernel.apply_weights(layer, x, bias)

    prepare.assert_called_once()
    kernel.quant_dequant_func.assert_called_once_with(x)
    assert layer._aiter_a16wfp4_prepared
    assert gemm.call_args.args[0].shape == (6, 256)
    assert gemm.call_args.args[0].is_contiguous()
    assert actual.shape == (2, 3, 256)
    torch.testing.assert_close(actual, bias.expand_as(actual))


@pytest.mark.parametrize(
    ("params_dtype", "n", "k"),
    [
        (torch.float16, 256, 256),
        (torch.bfloat16, 128, 256),
        (torch.bfloat16, 256, 384),
    ],
)
def test_ineligible_routes_cache_bf16_and_run_fp16(
    params_dtype: torch.dtype,
    n: int,
    k: int,
):
    kernel = _make_kernel()
    kernel.quant_dequant_func = MagicMock(side_effect=lambda x: x * 0.5)
    layer = _make_layer(n=n, k=k, params_dtype=params_dtype)
    decoded = torch.arange(n * k, dtype=torch.float32).reshape(n, k).to(torch.bfloat16)
    x = torch.ones((2, k), dtype=torch.float16)

    with (
        patch.object(envs, "VLLM_MXFP4_EMULATION_DEQUANT_AT_LOAD", True),
        patch(
            "vllm.model_executor.kernels.linear.mxfp4.aiter_a16wfp4."
            "_prepare_aiter_a16wfp4_weight"
        ) as prepare,
        patch(
            "vllm.model_executor.kernels.linear.mxfp4.aiter_a16wfp4.dequant_mxfp4",
            return_value=decoded,
        ) as dequant,
    ):
        kernel.process_weights_after_loading(layer)
        actual = kernel.apply_weights(layer, x)

    prepare.assert_not_called()
    dequant.assert_called_once()
    assert not layer._aiter_a16wfp4_prepared
    assert layer.weight.dtype == torch.bfloat16
    expected = torch.nn.functional.linear(x * 0.5, decoded.to(torch.float16))
    torch.testing.assert_close(actual, expected)


def test_missing_flydsl_or_preparation_failure_caches_weight():
    kernel = _make_kernel()
    layer = _make_layer()
    decoded = torch.zeros((256, 256), dtype=torch.bfloat16)

    with (
        patch.object(envs, "VLLM_MXFP4_EMULATION_DEQUANT_AT_LOAD", True),
        patch(
            "vllm.model_executor.kernels.linear.mxfp4.aiter_a16wfp4."
            "_prepare_aiter_a16wfp4_weight",
            side_effect=ModuleNotFoundError("no FlyDSL API"),
        ),
        patch(
            "vllm.model_executor.kernels.linear.mxfp4.aiter_a16wfp4.dequant_mxfp4",
            return_value=decoded,
        ) as dequant,
    ):
        kernel.process_weights_after_loading(layer)

    dequant.assert_called_once()
    assert not layer._aiter_a16wfp4_prepared
    assert layer.weight.dtype == torch.bfloat16


def test_dequant_at_load_disabled_retains_packed_weight_per_call():
    kernel = _make_kernel()
    kernel.quant_dequant_func = MagicMock(side_effect=lambda x: x)
    layer = _make_layer()
    decoded = torch.ones((256, 256), dtype=torch.bfloat16)
    x = torch.ones((1, 256), dtype=torch.bfloat16)

    with (
        patch.object(envs, "VLLM_MXFP4_EMULATION_DEQUANT_AT_LOAD", False),
        patch(
            "vllm.model_executor.kernels.linear.mxfp4.aiter_a16wfp4."
            "_prepare_aiter_a16wfp4_weight"
        ) as prepare,
        patch(
            "vllm.model_executor.kernels.linear.mxfp4.aiter_a16wfp4.dequant_mxfp4",
            return_value=decoded,
        ) as dequant,
    ):
        kernel.process_weights_after_loading(layer)
        first = kernel.apply_weights(layer, x)
        second = kernel.apply_weights(layer, x)

    prepare.assert_not_called()
    assert layer.weight.dtype == torch.uint8
    assert dequant.call_count == 2
    torch.testing.assert_close(first, second)
