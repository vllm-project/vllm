# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ScaledMM kernel selection logic (CPU-only)

Run `pytest tests/kernels/quantization/test_scaled_mm_kernel_selection.py`.
"""

import inspect
from abc import ABC
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.kernels.linear import (
    AiterInt8ScaledMMLinearKernel,
    CPUInt8ScaledMMLinearKernel,
    CutlassFp8BlockScaledMMKernel,
    DeepGemmFp8BlockScaledMMKernel,
    Fp8BlockScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
    MarlinFP8ScaledMMLinearKernel,
    ScaledMMLinearKernel,
    TritonFp8BlockScaledMMKernel,
    _apply_auto_kernel_preferences,
    _get_normalized_device_name,
    choose_scaled_mm_linear_kernel,
    init_int8_linear_kernel,
    register_linear_kernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import PlatformEnum

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def clear_normalized_device_name_cache():
    _get_normalized_device_name.cache_clear()
    yield
    _get_normalized_device_name.cache_clear()


def test_is_supported_is_abstract():
    """Test that is_supported() is properly defined as abstract."""
    assert issubclass(ScaledMMLinearKernel, ABC)
    assert hasattr(ScaledMMLinearKernel, "is_supported")


def test_cpu_kernel_implements_is_supported():
    """Test that CPUInt8ScaledMMLinearKernel implements is_supported() method."""
    assert hasattr(CPUInt8ScaledMMLinearKernel, "is_supported"), (
        "CPUInt8ScaledMMLinearKernel missing is_supported() method"
    )
    # Verify it's a classmethod by checking if it can be called with the class
    # and by checking the method type
    assert inspect.ismethod(
        CPUInt8ScaledMMLinearKernel.is_supported
    ) or inspect.isfunction(CPUInt8ScaledMMLinearKernel.is_supported), (
        "CPUInt8ScaledMMLinearKernel.is_supported() should be a classmethod"
    )
    # Verify it can be called as a classmethod
    result, reason = CPUInt8ScaledMMLinearKernel.is_supported()
    assert isinstance(result, bool), "is_supported() should return a bool"
    assert reason is None or isinstance(reason, str), "reason should be str or None"


def test_aiter_kernel_implements_is_supported():
    """Test that AiterInt8ScaledMMLinearKernel implements is_supported() method."""
    assert hasattr(AiterInt8ScaledMMLinearKernel, "is_supported"), (
        "AiterInt8ScaledMMLinearKernel missing is_supported() method"
    )
    # Verify it's a classmethod by checking if it can be called with the class
    # and by checking the method type
    assert inspect.ismethod(
        AiterInt8ScaledMMLinearKernel.is_supported
    ) or inspect.isfunction(AiterInt8ScaledMMLinearKernel.is_supported), (
        "AiterInt8ScaledMMLinearKernel.is_supported() should be a classmethod"
    )
    # Verify it can be called as a classmethod
    # (will return False on CPU, which is expected)
    result, reason = AiterInt8ScaledMMLinearKernel.is_supported()
    assert isinstance(result, bool), "is_supported() should return a bool"
    assert reason is None or isinstance(reason, str), "reason should be str or None"
    # On CPU, it should return False with a reason about requiring ROCm
    # This validates the method works correctly even on non-ROCm platforms


def test_cpu_kernel_accepts_all_configs():
    """Test that CPUInt8ScaledMMLinearKernel accepts all config combinations."""
    configs = [
        Int8ScaledMMLinearLayerConfig(
            is_channelwise=False,
            is_static_input_scheme=True,
            input_symmetric=True,
        ),
        Int8ScaledMMLinearLayerConfig(
            is_channelwise=True,
            is_static_input_scheme=False,
            input_symmetric=False,
        ),
    ]

    for config in configs:
        can_impl, reason = CPUInt8ScaledMMLinearKernel.can_implement(config)
        assert can_impl, (
            f"CPUInt8ScaledMMLinearKernel should accept config {config}: {reason}"
        )


class OOTInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        return True, None

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
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
def test_register_oot_linear_kernel(platform_mock):
    """Test that the linear kernel registration works correctly."""
    platform_mock._enum = PlatformEnum.OOT
    register_linear_kernel(OOTInt8ScaledMMLinearKernel, PlatformEnum.OOT, "int8")

    kernel = init_int8_linear_kernel(True, True, True, "module")

    assert isinstance(kernel, OOTInt8ScaledMMLinearKernel), (
        "init_int8_linear_kernel should return an instance of the registered kernel"
    )


def make_qwen_gdn_config(
    *,
    weight_shape: tuple[int, int] = (32, 1024),
    input_dtype: torch.dtype = torch.bfloat16,
    out_dtype: torch.dtype = torch.bfloat16,
) -> FP8ScaledMMLinearLayerConfig:
    return FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8Static128BlockSym,
        activation_quant_key=kFp8Dynamic128Sym,
        input_dtype=input_dtype,
        out_dtype=out_dtype,
        weight_shape=weight_shape,
    )


@pytest.mark.parametrize(
    ("device_name", "expected"),
    [
        (
            "NVIDIA H20",
            [TritonFp8BlockScaledMMKernel, CutlassFp8BlockScaledMMKernel],
        ),
        (
            "NVIDIA RTX PRO 5000 72GB Blackwell",
            [TritonFp8BlockScaledMMKernel],
        ),
    ],
)
@patch("vllm.model_executor.kernels.linear.current_platform")
def test_qwen_gdn_prefers_tuned_triton_in_auto(
    platform_mock,
    device_name: str,
    expected: list[type[Fp8BlockScaledMMLinearKernel]],
):
    platform_mock._enum = PlatformEnum.CUDA
    platform_mock.get_device_name.return_value = device_name
    kernels = [CutlassFp8BlockScaledMMKernel, TritonFp8BlockScaledMMKernel]

    config = make_qwen_gdn_config()
    result = _apply_auto_kernel_preferences(config, kernels)

    assert result == expected
    assert _apply_auto_kernel_preferences(config, result) == expected
    platform_mock.get_device_name.assert_called_once_with()


@patch("vllm.model_executor.kernels.linear.current_platform")
def test_qwen_gdn_preference_preserves_other_kernel_priorities(platform_mock):
    platform_mock._enum = PlatformEnum.CUDA
    platform_mock.get_device_name.return_value = "NVIDIA H20"
    kernels = [
        DeepGemmFp8BlockScaledMMKernel,
        CutlassFp8BlockScaledMMKernel,
        MarlinFP8ScaledMMLinearKernel,
        TritonFp8BlockScaledMMKernel,
    ]

    assert _apply_auto_kernel_preferences(make_qwen_gdn_config(), kernels) == [
        DeepGemmFp8BlockScaledMMKernel,
        TritonFp8BlockScaledMMKernel,
        CutlassFp8BlockScaledMMKernel,
        MarlinFP8ScaledMMLinearKernel,
    ]


@pytest.mark.parametrize(
    ("device_name", "weight_shape", "input_dtype", "out_dtype"),
    [
        ("NVIDIA H100", (32, 1024), torch.bfloat16, torch.bfloat16),
        ("NVIDIA H20", (64, 1024), torch.bfloat16, torch.bfloat16),
        ("NVIDIA H20", (32, 1024), torch.float16, torch.bfloat16),
        ("NVIDIA H20", (32, 1024), torch.bfloat16, torch.float16),
    ],
)
@patch("vllm.model_executor.kernels.linear.current_platform")
def test_qwen_gdn_preference_is_exact(
    platform_mock,
    device_name: str,
    weight_shape: tuple[int, int],
    input_dtype: torch.dtype,
    out_dtype: torch.dtype,
):
    platform_mock._enum = PlatformEnum.CUDA
    platform_mock.get_device_name.return_value = device_name
    kernels = [CutlassFp8BlockScaledMMKernel, TritonFp8BlockScaledMMKernel]
    config = make_qwen_gdn_config(
        weight_shape=weight_shape,
        input_dtype=input_dtype,
        out_dtype=out_dtype,
    )

    assert _apply_auto_kernel_preferences(config, kernels) == kernels
    if (
        weight_shape != (32, 1024)
        or input_dtype != torch.bfloat16
        or out_dtype != torch.bfloat16
    ):
        platform_mock.get_device_name.assert_not_called()


@pytest.mark.parametrize(
    ("field", "quant_key"),
    [
        ("weight_quant_key", kFp8StaticTensorSym),
        ("activation_quant_key", kFp8DynamicTensorSym),
    ],
)
@patch("vllm.model_executor.kernels.linear.current_platform")
def test_qwen_gdn_preference_requires_exact_quantization(
    platform_mock,
    field: str,
    quant_key: QuantKey,
):
    platform_mock._enum = PlatformEnum.CUDA
    platform_mock.get_device_name.return_value = "NVIDIA H20"
    kernels = [CutlassFp8BlockScaledMMKernel, TritonFp8BlockScaledMMKernel]
    config = make_qwen_gdn_config()
    setattr(config, field, quant_key)

    assert _apply_auto_kernel_preferences(config, kernels) == kernels
    platform_mock.get_device_name.assert_not_called()


@patch("vllm.model_executor.kernels.linear.is_supported_and_can_implement_kernel")
@patch("vllm.model_executor.kernels.linear.current_platform")
def test_qwen_gdn_auto_falls_back_when_triton_is_unavailable(
    platform_mock,
    support_mock,
):
    platform_mock._enum = PlatformEnum.CUDA
    platform_mock.get_device_name.return_value = "NVIDIA H20"
    support_mock.side_effect = lambda kernel, *_: (
        kernel is CutlassFp8BlockScaledMMKernel,
        "unavailable",
    )
    kernels = {
        PlatformEnum.CUDA: [
            CutlassFp8BlockScaledMMKernel,
            TritonFp8BlockScaledMMKernel,
        ]
    }

    selected = choose_scaled_mm_linear_kernel(make_qwen_gdn_config(), kernels)

    assert selected is CutlassFp8BlockScaledMMKernel


@patch("vllm.model_executor.kernels.linear.is_supported_and_can_implement_kernel")
@patch("vllm.model_executor.kernels.linear.current_platform")
def test_qwen_gdn_pro5000_does_not_fall_through_to_invalid_cutlass(
    platform_mock,
    support_mock,
):
    platform_mock._enum = PlatformEnum.CUDA
    platform_mock.get_device_name.return_value = "NVIDIA RTX PRO 5000 72GB Blackwell"
    support_mock.side_effect = lambda kernel, *_: (
        kernel is CutlassFp8BlockScaledMMKernel,
        "unavailable",
    )
    kernels = {
        PlatformEnum.CUDA: [
            CutlassFp8BlockScaledMMKernel,
            TritonFp8BlockScaledMMKernel,
        ]
    }

    with pytest.raises(ValueError, match="Failed to find a kernel"):
        choose_scaled_mm_linear_kernel(make_qwen_gdn_config(), kernels)

    support_mock.assert_called_once_with(
        TritonFp8BlockScaledMMKernel, make_qwen_gdn_config(), None
    )


@pytest.mark.parametrize(
    "force_kernel",
    [CutlassFp8BlockScaledMMKernel, TritonFp8BlockScaledMMKernel],
)
@patch("vllm.model_executor.kernels.linear.is_supported_and_can_implement_kernel")
@patch("vllm.model_executor.kernels.linear.current_platform")
def test_qwen_gdn_force_kernel_is_respected(
    platform_mock,
    support_mock,
    force_kernel: type[Fp8BlockScaledMMLinearKernel],
):
    platform_mock._enum = PlatformEnum.CUDA
    platform_mock.get_device_name.return_value = "NVIDIA H20"
    support_mock.return_value = (True, "")
    kernels = {
        PlatformEnum.CUDA: [
            CutlassFp8BlockScaledMMKernel,
            TritonFp8BlockScaledMMKernel,
        ]
    }

    assert (
        choose_scaled_mm_linear_kernel(
            make_qwen_gdn_config(), kernels, force_kernel=force_kernel
        )
        is force_kernel
    )
    support_mock.assert_called_once_with(force_kernel, make_qwen_gdn_config(), None)
    platform_mock.get_device_name.assert_not_called()
