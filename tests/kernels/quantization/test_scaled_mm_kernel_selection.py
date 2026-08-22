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
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
    ScaledMMLinearKernel,
    init_int8_linear_kernel,
    register_linear_kernel,
)
from vllm.platforms import PlatformEnum

pytestmark = pytest.mark.cpu_test


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


@pytest.mark.parametrize(
    "compute_capability,expected_supported",
    [(90, False), (100, True), (103, True), (120, True), (121, True)],
)
def test_flashinfer_fp8_kernel_is_supported_per_arch(
    compute_capability: int, expected_supported: bool
):
    """FlashInfer per-tensor FP8 kernel stays enabled on Blackwell,
    including the sm_12x family — the cuDNN hot-path stall there is
    handled by pinning the bmm_fp8 backend, not by gating the kernel
    (see bmm_fp8_backend in vllm.utils.flashinfer)."""
    from vllm.model_executor.kernels.linear import (
        FlashInferFP8ScaledMMLinearKernel,
    )

    with (
        patch(
            "vllm.model_executor.kernels.linear.scaled_mm.flashinfer"
            ".current_platform"
        ) as platform_mock,
        patch(
            "vllm.model_executor.kernels.linear.scaled_mm.flashinfer"
            ".has_flashinfer",
            return_value=True,
        ),
    ):
        platform_mock.is_cuda.return_value = True
        supported, reason = FlashInferFP8ScaledMMLinearKernel.is_supported(
            compute_capability
        )
    assert supported is expected_supported, reason


@pytest.mark.parametrize(
    "capability_family_120,expected_backend",
    [(True, "cublas"), (False, "auto")],
)
def test_bmm_fp8_backend_pins_cublas_on_sm_12x(
    capability_family_120: bool, expected_backend: str
):
    """On sm_12x, bmm_fp8 must not use backend="auto": with the cudnn
    python module importable, FlashInfer's auto selection lazily builds
    cuDNN GEMM graphs per novel (batch, seqlen) shape inside the serving
    hot path (multi-second stalls) and can reject plans at execute time
    (flashinfer-ai/flashinfer#3566)."""
    from vllm.utils import flashinfer as fi_utils

    def fake_family(capability: int) -> bool:
        return capability_family_120 if capability == 120 else False

    saved = fi_utils._BMM_FP8_BACKEND
    fi_utils._BMM_FP8_BACKEND = None
    try:
        with patch.object(
            fi_utils.current_platform,
            "is_device_capability_family",
            side_effect=fake_family,
        ):
            assert fi_utils.bmm_fp8_backend() == expected_backend
    finally:
        fi_utils._BMM_FP8_BACKEND = saved
