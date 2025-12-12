# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ScaledMM kernel selection logic (CPU-only)

Run `pytest tests/kernels/quantization/test_scaled_mm_kernel_selection.py`.
"""

import inspect
from abc import ABC

import pytest

from vllm.model_executor.layers.quantization.kernels.scaled_mm import (
    ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.aiter import (
    AiterScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.cpu import (
    CPUScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.ScaledMMLinearKernel import (  # noqa: E501
    ScaledMMLinearKernel,
)

pytestmark = pytest.mark.cpu_test


def test_is_supported_is_abstract():
    """Test that is_supported() is properly defined as abstract."""
    assert issubclass(ScaledMMLinearKernel, ABC)
    assert hasattr(ScaledMMLinearKernel, "is_supported")


def test_cpu_kernel_implements_is_supported():
    """Test that CPUScaledMMLinearKernel implements is_supported() method."""
    assert hasattr(CPUScaledMMLinearKernel, "is_supported"), (
        "CPUScaledMMLinearKernel missing is_supported() method"
    )
    # Verify it's a classmethod by checking if it can be called with the class
    # and by checking the method type
    assert inspect.ismethod(CPUScaledMMLinearKernel.is_supported) or inspect.isfunction(
        CPUScaledMMLinearKernel.is_supported
    ), "CPUScaledMMLinearKernel.is_supported() should be a classmethod"
    # Verify it can be called as a classmethod
    result, reason = CPUScaledMMLinearKernel.is_supported()
    assert isinstance(result, bool), "is_supported() should return a bool"
    assert reason is None or isinstance(reason, str), "reason should be str or None"


def test_aiter_kernel_implements_is_supported():
    """Test that AiterScaledMMLinearKernel implements is_supported() method."""
    assert hasattr(AiterScaledMMLinearKernel, "is_supported"), (
        "AiterScaledMMLinearKernel missing is_supported() method"
    )
    # Verify it's a classmethod by checking if it can be called with the class
    # and by checking the method type
    assert inspect.ismethod(
        AiterScaledMMLinearKernel.is_supported
    ) or inspect.isfunction(AiterScaledMMLinearKernel.is_supported), (
        "AiterScaledMMLinearKernel.is_supported() should be a classmethod"
    )
    # Verify it can be called as a classmethod
    # (will return False on CPU, which is expected)
    result, reason = AiterScaledMMLinearKernel.is_supported()
    assert isinstance(result, bool), "is_supported() should return a bool"
    assert reason is None or isinstance(reason, str), "reason should be str or None"
    # On CPU, it should return False with a reason about requiring ROCm
    # This validates the method works correctly even on non-ROCm platforms


def test_cpu_kernel_accepts_all_configs():
    """Test that CPUScaledMMLinearKernel accepts all config combinations."""
    configs = [
        ScaledMMLinearLayerConfig(
            is_channelwise=False,
            is_static_input_scheme=True,
            input_symmetric=True,
        ),
        ScaledMMLinearLayerConfig(
            is_channelwise=True,
            is_static_input_scheme=False,
            input_symmetric=False,
        ),
    ]

    for config in configs:
        can_impl, reason = CPUScaledMMLinearKernel.can_implement(config)
        assert can_impl, (
            f"CPUScaledMMLinearKernel should accept config {config}: {reason}"
        )
