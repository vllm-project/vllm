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
    MxFp4LinearKernel,
    MxFp4LinearLayerConfig,
    init_mxfp4_linear_kernel,
    register_linear_kernel,
)
from vllm.platforms import PlatformEnum

pytestmark = pytest.mark.cpu_test


def test_can_implement_is_abstract():
    """Test that can_implement()/is_supported() are properly defined."""
    assert hasattr(MxFp4LinearKernel, "can_implement")
    assert hasattr(MxFp4LinearKernel, "is_supported")


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
    reports itself as supported, and construct it with a fresh config."""
    platform_mock._enum = PlatformEnum.OOT
    register_linear_kernel(OOTMxFp4LinearKernel, PlatformEnum.OOT, "mxfp4")

    kernel = init_mxfp4_linear_kernel()

    assert isinstance(kernel, OOTMxFp4LinearKernel)
    assert kernel.config == MxFp4LinearLayerConfig()


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
        init_mxfp4_linear_kernel()


@patch("vllm.model_executor.kernels.linear.mxfp4.aiter.is_aiter_found_and_supported")
@patch("vllm.model_executor.kernels.linear.mxfp4.aiter.current_platform")
@patch("vllm.model_executor.kernels.linear.current_platform")
def test_init_mxfp4_linear_kernel_raises_on_rocm_without_aiter(
    linear_platform_mock, aiter_platform_mock, is_aiter_found_and_supported_mock
):
    """On ROCm, the only registered MXFP4 linear kernel is AITER-based.
    If AITER is not found/supported, no kernel should be selected."""
    linear_platform_mock._enum = PlatformEnum.ROCM
    aiter_platform_mock.supports_mx.return_value = True
    is_aiter_found_and_supported_mock.return_value = False

    with pytest.raises(
        ValueError,
        match="(?s)Failed to find a kernel.*"
        "AITER not found or not supported on the current platform",
    ):
        init_mxfp4_linear_kernel()
