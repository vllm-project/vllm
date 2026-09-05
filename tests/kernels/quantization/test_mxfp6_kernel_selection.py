# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MXFP6 linear kernel selection logic (CPU-only)

Run `pytest tests/kernels/quantization/test_mxfp6_kernel_selection.py`.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.kernels.linear import (
    EmulationMxfp6LinearKernel,
    MxFp6LinearKernel,
    MxFp6LinearLayerConfig,
    Mxfp6Sm120LinearKernel,
    init_mxfp6_linear_kernel,
    register_linear_kernel,
)
from vllm.model_executor.kernels.linear.mxfp6 import sm120 as sm120_module
from vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx import (
    QuarkOCP_MX,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Dynamic,
    kMxfp4Static,
    kMxfp6E2M3Dynamic,
    kMxfp6E2M3Static,
    kMxfp6E3M2Dynamic,
    kMxfp6E3M2Static,
    kMxfp8Dynamic,
)
from vllm.platforms import PlatformEnum

pytestmark = pytest.mark.cpu_test

_WEIGHT_QUANT_KEYS = [kMxfp6E3M2Static, kMxfp6E2M3Static]


def test_can_implement_is_abstract():
    """Test that can_implement()/is_supported() are properly defined."""
    assert hasattr(MxFp6LinearKernel, "can_implement")
    assert hasattr(MxFp6LinearKernel, "is_supported")


def test_emulation_kernel_rejects_non_mxfp6_weights():
    """EmulationMxfp6LinearKernel must not implement a non-MXFP6 weight
    format."""
    config = MxFp6LinearLayerConfig(weight_quant_key=kMxfp4Static)
    can_implement, reason = EmulationMxfp6LinearKernel.can_implement(config)
    assert not can_implement
    assert reason


@pytest.mark.parametrize("weight_quant_key", _WEIGHT_QUANT_KEYS)
@pytest.mark.parametrize(
    "activation_quant_key",
    [None, kMxfp4Dynamic, kMxfp6E3M2Dynamic, kMxfp6E2M3Dynamic, kMxfp8Dynamic],
)
def test_emulation_kernel_accepts_any_supported_config(
    weight_quant_key, activation_quant_key
):
    """Emulation must preserve a fallback for each recognized MX format."""
    config = MxFp6LinearLayerConfig(
        weight_quant_key=weight_quant_key, activation_quant_key=activation_quant_key
    )
    can_implement, reason = EmulationMxfp6LinearKernel.can_implement(config)
    assert can_implement, reason


@pytest.mark.parametrize("weight_quant_key", _WEIGHT_QUANT_KEYS)
def test_emulation_kernel_rejects_non_mxfp4_or_mxfp6_activation(weight_quant_key):
    config = MxFp6LinearLayerConfig(
        weight_quant_key=weight_quant_key, activation_quant_key=kMxfp4Static
    )
    can_implement, reason = EmulationMxfp6LinearKernel.can_implement(config)
    assert not can_implement
    assert reason


def test_quark_ocp_mx_recognizes_dense_w6a8():
    scheme = QuarkOCP_MX(
        weight_quant_key=kMxfp6E3M2Static,
        activation_quant_key=kMxfp8Dynamic,
    )

    assert scheme.weight_quant_key == kMxfp6E3M2Static
    assert scheme.activation_quant_key == kMxfp8Dynamic


@pytest.mark.parametrize(
    ("weight_quant_key", "activation_quant_key", "expected"),
    [
        (kMxfp6E3M2Static, kMxfp8Dynamic, True),
        (kMxfp6E2M3Static, kMxfp8Dynamic, False),
        (kMxfp6E3M2Static, kMxfp6E3M2Dynamic, False),
        (kMxfp6E3M2Static, None, False),
    ],
)
def test_sm120_kernel_accepts_only_native_w6a8_format(
    weight_quant_key, activation_quant_key, expected
):
    config = MxFp6LinearLayerConfig(
        weight_quant_key=weight_quant_key,
        activation_quant_key=activation_quant_key,
    )

    can_implement, reason = Mxfp6Sm120LinearKernel.can_implement(config)

    assert can_implement is expected
    assert (reason is None) is expected


def test_sm120_kernel_rejects_non_sm120_compute_capability():
    supported, reason = Mxfp6Sm120LinearKernel.is_supported(compute_capability=90)

    assert not supported
    assert reason == "requires SM120"


def test_sm120_availability_rechecks_after_an_incompatible_package():
    package_api = {
        "is_available": lambda: True,
        "load_library": lambda: None,
        "pack_scales": lambda scales: scales,
    }
    old_package = SimpleNamespace(__version__="0.2.0", **package_api)
    current_package = SimpleNamespace(__version__="0.2.1", **package_api)
    torch_ops = SimpleNamespace(mxfp6=SimpleNamespace(gemm_w6a8=object()))

    with (
        patch.object(sm120_module.current_platform, "is_cuda", return_value=True),
        patch.object(
            sm120_module.current_platform,
            "is_device_capability",
            return_value=True,
        ),
        patch.object(
            sm120_module,
            "_import_mxfp6",
            side_effect=[old_package, current_package],
        ),
        patch.object(sm120_module.torch, "ops", torch_ops),
    ):
        assert not sm120_module.is_mxfp6_sm120_available()
        assert sm120_module.is_mxfp6_sm120_available()


@patch(
    "vllm.model_executor.kernels.linear.mxfp6.sm120.is_mxfp6_sm120_available",
    return_value=False,
)
@patch("vllm.model_executor.kernels.linear.current_platform")
def test_selector_falls_back_when_native_package_is_unavailable(
    platform_mock, _availability
):
    platform_mock._enum = PlatformEnum.CUDA

    kernel = init_mxfp6_linear_kernel(kMxfp6E3M2Static, kMxfp8Dynamic)

    assert isinstance(kernel, EmulationMxfp6LinearKernel)


@patch(
    "vllm.model_executor.kernels.linear.mxfp6.sm120.is_mxfp6_sm120_available",
    return_value=True,
)
@patch("vllm.model_executor.kernels.linear.current_platform")
def test_selector_falls_back_for_unsupported_native_format(
    platform_mock, _availability
):
    platform_mock._enum = PlatformEnum.CUDA

    kernel = init_mxfp6_linear_kernel(kMxfp6E2M3Static, kMxfp8Dynamic)

    assert isinstance(kernel, EmulationMxfp6LinearKernel)


@patch(
    "vllm.model_executor.kernels.linear.mxfp6.sm120.is_mxfp6_sm120_available",
    return_value=True,
)
def test_sm120_kernel_packs_supported_weight_scales(_availability):
    config = MxFp6LinearLayerConfig(kMxfp6E3M2Static, kMxfp8Dynamic)
    kernel = Mxfp6Sm120LinearKernel(config)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.zeros((8, 96), dtype=torch.uint8), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.arange(32, dtype=torch.uint8).reshape(8, 4), requires_grad=False
    )
    package = SimpleNamespace(pack_scales=lambda scale: scale.flatten())

    with patch(
        "vllm.model_executor.kernels.linear.mxfp6.sm120._import_mxfp6",
        return_value=package,
    ):
        kernel.process_weights_after_loading(layer)

    assert layer._mxfp6_sm120_native
    assert layer.weight_scale.shape == (32,)
    assert torch.equal(layer.weight_scale, torch.arange(32, dtype=torch.uint8))


@patch(
    "vllm.model_executor.kernels.linear.mxfp6.sm120.is_mxfp6_sm120_available",
    return_value=True,
)
def test_sm120_kernel_pads_an_ocp_partition_to_native_k_alignment(_availability):
    config = MxFp6LinearLayerConfig(kMxfp6E3M2Static, kMxfp8Dynamic)
    kernel = Mxfp6Sm120LinearKernel(config)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.full((8, 24), 3, dtype=torch.uint8), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.full((8, 1), 128, dtype=torch.uint8), requires_grad=False
    )
    package = SimpleNamespace(pack_scales=lambda scale: scale)

    with patch(
        "vllm.model_executor.kernels.linear.mxfp6.sm120._import_mxfp6",
        return_value=package,
    ):
        kernel.process_weights_after_loading(layer)

    assert layer._mxfp6_sm120_native
    assert layer._mxfp6_sm120_input_features == 32
    assert layer._mxfp6_sm120_padded_input_features == 128
    assert layer.weight.shape == (8, 96)
    assert torch.equal(layer.weight[:, :24], torch.full((8, 24), 3, dtype=torch.uint8))
    assert torch.count_nonzero(layer.weight[:, 24:]) == 0
    assert layer.weight_scale.shape == (8, 4)
    assert torch.all(layer.weight_scale[:, :1] == 128)
    assert torch.all(layer.weight_scale[:, 1:] == 127)


@patch(
    "vllm.model_executor.kernels.linear.mxfp6.sm120.is_mxfp6_sm120_available",
    return_value=True,
)
def test_sm120_kernel_keeps_emulation_fallback_for_unsupported_shape(_availability):
    config = MxFp6LinearLayerConfig(kMxfp6E3M2Static, kMxfp8Dynamic)
    kernel = Mxfp6Sm120LinearKernel(config)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.zeros((7, 96), dtype=torch.uint8), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.zeros((7, 4), dtype=torch.uint8), requires_grad=False
    )

    kernel.process_weights_after_loading(layer)

    assert not layer._mxfp6_sm120_native
    assert layer.weight_scale.shape == (7, 4)


@patch(
    "vllm.model_executor.kernels.linear.mxfp6.sm120.is_mxfp6_sm120_available",
    return_value=True,
)
def test_sm120_kernel_quantizes_and_restores_input_shape(_availability):
    config = MxFp6LinearLayerConfig(kMxfp6E3M2Static, kMxfp8Dynamic)
    kernel = Mxfp6Sm120LinearKernel(config)
    layer = SimpleNamespace(
        _mxfp6_sm120_native=True,
        _mxfp6_sm120_input_features=128,
        _mxfp6_sm120_padded_input_features=128,
        _mxfp6_sm120_output_features=8,
        weight=torch.zeros((8, 96), dtype=torch.uint8),
        weight_scale=torch.zeros(32, dtype=torch.uint8),
    )
    x = torch.randn((2, 3, 128), dtype=torch.bfloat16)
    quantized = torch.empty((6, 128), dtype=torch.float8_e4m3fn)
    input_scale = torch.empty((128,), dtype=torch.uint8)
    expected = torch.arange(48, dtype=torch.bfloat16).reshape(6, 8)

    with (
        patch(
            "vllm.model_executor.kernels.linear.mxfp6.sm120.mxfp8_e4m3_quantize",
            return_value=(quantized, input_scale),
        ) as quantize,
        patch(
            "vllm.model_executor.kernels.linear.mxfp6.sm120.mxfp6_sm120_gemm",
            return_value=expected,
        ) as gemm,
    ):
        output = kernel.apply_weights(layer, x)

    assert output.shape == (2, 3, 8)
    quantize.assert_called_once()
    quantize_input = quantize.call_args.args[0]
    assert torch.equal(quantize_input, x.reshape(-1, 128))
    assert quantize_input.is_contiguous()
    assert quantize.call_args.kwargs == {"is_sf_swizzled_layout": True}
    gemm.assert_called_once_with(
        quantized,
        input_scale,
        layer.weight,
        layer.weight_scale,
        8,
        torch.bfloat16,
    )
    assert torch.equal(output, expected.reshape(2, 3, 8))


class OOTMxFp6LinearKernel(MxFp6LinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        return True, None

    @classmethod
    def can_implement(cls, config: MxFp6LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pass


@patch("vllm.model_executor.kernels.linear.current_platform")
def test_init_mxfp6_linear_kernel_dispatches_to_registered_kernel(platform_mock):
    """init_mxfp6_linear_kernel should select a registered kernel that
    reports itself as supported/able to implement the given config, and
    construct it with that exact config."""
    platform_mock._enum = PlatformEnum.OOT
    register_linear_kernel(OOTMxFp6LinearKernel, PlatformEnum.OOT, "mxfp6")

    kernel = init_mxfp6_linear_kernel(
        weight_quant_key=kMxfp6E3M2Static, activation_quant_key=kMxfp6E3M2Dynamic
    )

    assert isinstance(kernel, OOTMxFp6LinearKernel)
    assert kernel.config == MxFp6LinearLayerConfig(
        weight_quant_key=kMxfp6E3M2Static, activation_quant_key=kMxfp6E3M2Dynamic
    )


class UnsupportedMxFp6LinearKernel(MxFp6LinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        return False, "never supported"

    @classmethod
    def can_implement(cls, config: MxFp6LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pass


@patch("vllm.model_executor.kernels.linear.current_platform")
def test_init_mxfp6_linear_kernel_raises_when_no_kernel_matches(platform_mock):
    platform_mock._enum = PlatformEnum.UNSPECIFIED
    register_linear_kernel(
        UnsupportedMxFp6LinearKernel, PlatformEnum.UNSPECIFIED, "mxfp6"
    )

    with pytest.raises(ValueError, match="Failed to find a kernel"):
        init_mxfp6_linear_kernel(weight_quant_key=kMxfp6E3M2Static)
