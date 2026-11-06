# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MXFP6 linear kernel selection logic (CPU-only).

Run `pytest tests/kernels/quantization/test_mxfp6_kernel_selection.py`.
"""

from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.kernels.linear import (
    EmulationMxfp6LinearKernel,
    HummingMxFp6LinearKernel,
    MxFp6LinearKernel,
    MxFp6LinearLayerConfig,
    init_mxfp6_linear_kernel,
    register_linear_kernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt8Dynamic32Sym,
    kMxfp4Dynamic,
    kMxfp4Static,
    kMxfp6E2M3Dynamic,
    kMxfp6E2M3Static,
    kMxfp6E3M2Dynamic,
    kMxfp6E3M2Static,
)
from vllm.platforms import PlatformEnum

pytestmark = pytest.mark.cpu_test

# The only implementation available at the moment is software emulation.
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
    [None, kMxfp4Dynamic, kMxfp6E3M2Dynamic, kMxfp6E2M3Dynamic],
)
def test_emulation_kernel_accepts_any_supported_config(
    weight_quant_key, activation_quant_key
):
    """EmulationMxfp6LinearKernel is the only backend today: it must accept
    every supported weight/activation format combination."""
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


@pytest.mark.parametrize("weight_quant_key", _WEIGHT_QUANT_KEYS)
def test_supports_quant_scheme_dispatches_on_weight_and_activation(weight_quant_key):
    """Base and emulation MXFP6 kernels gate on weight and activation keys."""
    for act_key in (None, kMxfp6E2M3Dynamic, kMxfp6E3M2Dynamic):
        assert MxFp6LinearKernel._supports_quant_scheme(weight_quant_key, act_key)
        assert EmulationMxfp6LinearKernel._supports_quant_scheme(
            weight_quant_key, act_key
        )
    assert not MxFp6LinearKernel._supports_quant_scheme(weight_quant_key, kMxfp4Dynamic)
    assert EmulationMxfp6LinearKernel._supports_quant_scheme(
        weight_quant_key, kMxfp4Dynamic
    )
    for cls in (MxFp6LinearKernel, EmulationMxfp6LinearKernel):
        assert not cls._supports_quant_scheme(kMxfp4Static, None)
        assert not cls._supports_quant_scheme(weight_quant_key, kInt8Dynamic32Sym)

    assert not HummingMxFp6LinearKernel._supports_quant_scheme(kMxfp4Static, None)
    with patch(
        "vllm.model_executor.kernels.linear.mxfp6.humming.quant_key_to_input_schema",
        side_effect=ValueError("unsupported"),
    ):
        assert not HummingMxFp6LinearKernel._supports_quant_scheme(
            weight_quant_key, kInt8Dynamic32Sym
        )


@patch("vllm.model_executor.kernels.linear.current_platform")
def test_init_mxfp6_linear_kernel_applies_override_and_scheme_gate(
    platform_mock, monkeypatch
):
    """init_mxfp6_linear_kernel resolves linear.activation overrides and skips
    kernels whose _supports_quant_scheme rejects the resolved scheme."""
    import vllm.model_executor.kernels.linear as linear_kernels

    platform_mock._enum = PlatformEnum.OOT
    monkeypatch.setitem(
        linear_kernels._POSSIBLE_MXFP6_KERNELS,
        PlatformEnum.OOT,
        [OOTMxFp6LinearKernel, EmulationMxfp6LinearKernel],
    )

    # OOTMxFp6LinearKernel inherits the base gate (MXFP6 or None activations),
    # so overriding the activation to MXFP4 skips it and selects emulation.
    monkeypatch.setattr(
        linear_kernels, "_user_linear_activation_override", lambda: kMxfp4Dynamic
    )
    kernel = init_mxfp6_linear_kernel(
        weight_quant_key=kMxfp6E3M2Static,
        activation_quant_key=kMxfp6E3M2Dynamic,
    )
    assert isinstance(kernel, EmulationMxfp6LinearKernel)
    assert kernel.config.activation_quant_key == kMxfp4Dynamic

    # An explicit `activation: null` override runs weight-only and selects OOT.
    monkeypatch.setattr(
        linear_kernels, "_user_linear_activation_override_is_set", lambda: True
    )
    monkeypatch.setattr(
        linear_kernels, "_user_linear_activation_override", lambda: None
    )
    kernel = init_mxfp6_linear_kernel(
        weight_quant_key=kMxfp6E3M2Static,
        activation_quant_key=kMxfp4Dynamic,
    )
    assert isinstance(kernel, OOTMxFp6LinearKernel)
    assert kernel.config.activation_quant_key is None
