# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MXFP4 linear kernel selection logic (CPU-only)

Run `pytest tests/kernels/quantization/test_mxfp4_kernel_selection.py`.
"""

from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.kernels.linear import (
    AiterMxfp4LinearKernel,
    EmulationOcpMxLinearKernel,
    FlashInferMxFp4LinearKernel,
    HummingMxFp4LinearKernel,
    MarlinMxFp4LinearKernel,
    MxFp4LinearKernel,
    MxFp4LinearLayerConfig,
    XPUMxFp4LinearKernel,
    init_mxfp4_linear_kernel,
    register_linear_kernel,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    dequant_mxfp4,
    quant_dequant_mxfp4,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Dynamic,
    kMxfp4Static,
    kMxfp6E3M2Dynamic,
    kMxfp6E3M2Static,
)
from vllm.platforms import PlatformEnum

pytestmark = pytest.mark.cpu_test

# Kernels that quantize activations themselves (true W4A4): they only accept
# no activation-format expectation, or an exact MXFP4-dynamic match.
_TRUE_W4A4_KERNELS = [
    FlashInferMxFp4LinearKernel,
    XPUMxFp4LinearKernel,
    AiterMxfp4LinearKernel,
]

# Weight-only (A16) kernels: they never quantize activations, so an
# unquantized MXFP4-static activation key is tolerated (and ignored) in
# addition to None/dynamic.
_WEIGHT_ONLY_KERNELS = [MarlinMxFp4LinearKernel, HummingMxFp4LinearKernel]


def test_can_implement_is_abstract():
    """Test that can_implement()/is_supported() are properly defined."""
    assert hasattr(MxFp4LinearKernel, "can_implement")
    assert hasattr(MxFp4LinearKernel, "is_supported")


@pytest.mark.parametrize(
    "kernel_cls", _TRUE_W4A4_KERNELS + _WEIGHT_ONLY_KERNELS + [AiterMxfp4LinearKernel]
)
def test_all_kernels_reject_non_mxfp4_weights(kernel_cls):
    """No MXFP4 kernel should implement a non-MXFP4 weight format."""
    config = MxFp4LinearLayerConfig(weight_quant_key=kMxfp6E3M2Static)
    can_implement, reason = kernel_cls.can_implement(config)
    assert not can_implement
    assert reason


@pytest.mark.parametrize("kernel_cls", _TRUE_W4A4_KERNELS)
@pytest.mark.parametrize("activation_quant_key", [None, kMxfp4Dynamic])
def test_true_w4a4_kernels_accept_native_or_unset_activation(
    kernel_cls, activation_quant_key
):
    config = MxFp4LinearLayerConfig(
        weight_quant_key=kMxfp4Static, activation_quant_key=activation_quant_key
    )
    can_implement, reason = kernel_cls.can_implement(config)
    assert can_implement, reason


@pytest.mark.parametrize("kernel_cls", _TRUE_W4A4_KERNELS)
def test_true_w4a4_kernels_reject_explicit_non_mxfp4_activation(kernel_cls):
    """FlashInfer/XPU/Aiter quantize activations to MXFP4 internally, so an
    explicit request for a different activation format must be rejected."""
    config = MxFp4LinearLayerConfig(
        weight_quant_key=kMxfp4Static, activation_quant_key=kMxfp6E3M2Dynamic
    )
    can_implement, reason = kernel_cls.can_implement(config)
    assert not can_implement
    assert reason


@pytest.mark.parametrize("kernel_cls", _WEIGHT_ONLY_KERNELS)
@pytest.mark.parametrize(
    "activation_quant_key", [None, kMxfp4Dynamic, kMxfp4Static]
)
def test_weight_only_kernels_accept_unquantized_or_mxfp4_activation(
    kernel_cls, activation_quant_key
):
    """Marlin/Humming never quantize activations, so an unset activation key,
    or one that already describes MXFP4-shaped data, is tolerated."""
    config = MxFp4LinearLayerConfig(
        weight_quant_key=kMxfp4Static, activation_quant_key=activation_quant_key
    )
    can_implement, reason = kernel_cls.can_implement(config)
    assert can_implement, reason


@pytest.mark.parametrize("kernel_cls", _WEIGHT_ONLY_KERNELS)
def test_weight_only_kernels_reject_non_mxfp4_activation(kernel_cls):
    config = MxFp4LinearLayerConfig(
        weight_quant_key=kMxfp4Static, activation_quant_key=kMxfp6E3M2Dynamic
    )
    can_implement, reason = kernel_cls.can_implement(config)
    assert not can_implement
    assert reason


@pytest.mark.parametrize(
    "weight_quant_key,activation_quant_key",
    [
        (kMxfp4Static, None),
        (kMxfp4Static, kMxfp4Dynamic),
        (kMxfp6E3M2Static, kMxfp6E3M2Dynamic),
    ],
)
def test_emulation_kernel_accepts_any_config(weight_quant_key, activation_quant_key):
    """EmulationOcpMxLinearKernel is the universal fallback: it must accept
    every weight/activation format combination."""
    config = MxFp4LinearLayerConfig(
        weight_quant_key=weight_quant_key, activation_quant_key=activation_quant_key
    )
    can_implement, reason = EmulationOcpMxLinearKernel.can_implement(config)
    assert can_implement, reason


def test_emulation_kernel_derives_dequant_funcs_from_config():
    """dequant_func/quant_dequant_func must be derived purely from the
    config's weight/activation QuantKeys, not set externally."""
    weight_only_config = MxFp4LinearLayerConfig(weight_quant_key=kMxfp4Static)
    kernel = EmulationOcpMxLinearKernel(weight_only_config)
    assert kernel.dequant_func is dequant_mxfp4
    x = torch.randn(4)
    assert torch.equal(kernel.quant_dequant_func(x), x)  # identity for weight-only

    w4a4_config = MxFp4LinearLayerConfig(
        weight_quant_key=kMxfp4Static, activation_quant_key=kMxfp4Dynamic
    )
    kernel = EmulationOcpMxLinearKernel(w4a4_config)
    assert kernel.dequant_func is dequant_mxfp4
    assert kernel.quant_dequant_func is quant_dequant_mxfp4


def test_aiter_kernel_is_supported_requires_native_mx_support():
    """AiterMxfp4LinearKernel must not be selected on platforms without
    native MX compute, even if AITER itself is importable."""
    with patch(
        "vllm.model_executor.kernels.linear.mxfp4.aiter.current_platform.supports_mx",
        return_value=False,
    ):
        is_supported, reason = AiterMxfp4LinearKernel.is_supported()
    assert not is_supported
    assert reason


class OOTMxFp4LinearKernel(MxFp4LinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        return True, None

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
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
def test_init_mxfp4_linear_kernel_dispatches_to_registered_kernel(platform_mock):
    """init_mxfp4_linear_kernel should select a registered kernel that
    reports itself as supported/able to implement the given config, and
    construct it with that exact config."""
    platform_mock._enum = PlatformEnum.OOT
    register_linear_kernel(OOTMxFp4LinearKernel, PlatformEnum.OOT, "mxfp4")

    kernel = init_mxfp4_linear_kernel(
        weight_quant_key=kMxfp4Static, activation_quant_key=kMxfp4Dynamic
    )

    assert isinstance(kernel, OOTMxFp4LinearKernel)
    assert kernel.config == MxFp4LinearLayerConfig(
        weight_quant_key=kMxfp4Static, activation_quant_key=kMxfp4Dynamic
    )


class UnsupportedMxFp4LinearKernel(MxFp4LinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        return False, "never supported"

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
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
def test_init_mxfp4_linear_kernel_raises_when_no_kernel_matches(platform_mock):
    platform_mock._enum = PlatformEnum.UNSPECIFIED
    register_linear_kernel(
        UnsupportedMxFp4LinearKernel, PlatformEnum.UNSPECIFIED, "mxfp4"
    )

    with pytest.raises(ValueError, match="Failed to find a kernel"):
        init_mxfp4_linear_kernel(weight_quant_key=kMxfp4Static)
