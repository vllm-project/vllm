# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ScaledMM kernel selection logic

Run `pytest tests/kernels/quantization/test_scaled_mm_kernel_selection.py`.
"""

from abc import ABC

from vllm.model_executor.layers.quantization.kernels.scaled_mm import (
    ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.cpu import (
    CPUScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.cutlass import (
    CutlassScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.ScaledMMLinearKernel import (  # noqa: E501
    ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.triton import (
    TritonScaledMMLinearKernel,
)


def test_is_supported_is_abstract():
    """Test that is_supported() is properly defined as abstract."""
    assert issubclass(ScaledMMLinearKernel, ABC)
    assert hasattr(ScaledMMLinearKernel, "is_supported")


def test_all_kernels_implement_is_supported():
    """Test that all kernel classes implement is_supported() method."""
    kernels = [
        CPUScaledMMLinearKernel,
        CutlassScaledMMLinearKernel,
        TritonScaledMMLinearKernel,
    ]

    for kernel in kernels:
        assert hasattr(kernel, "is_supported"), (
            f"{kernel.__name__} missing is_supported() method"
        )
        # Verify it's a classmethod
        assert isinstance(kernel.is_supported, classmethod), (
            f"{kernel.__name__}.is_supported() should be a classmethod"
        )


def test_triton_kernel_rejects_asymmetric():
    """Test that TritonScaledMMLinearKernel rejects asymmetric quantization."""
    config = ScaledMMLinearLayerConfig(
        is_channelwise=False,
        is_static_input_scheme=True,
        input_symmetric=False,  # Asymmetric
    )

    can_impl, reason = TritonScaledMMLinearKernel.can_implement(config)
    assert not can_impl, "TritonScaledMMLinearKernel should reject asymmetric config"
    assert "symmetric" in reason.lower(), f"Unexpected rejection reason: {reason}"


def test_triton_kernel_accepts_symmetric():
    """Test that TritonScaledMMLinearKernel accepts symmetric quantization."""
    config = ScaledMMLinearLayerConfig(
        is_channelwise=False,
        is_static_input_scheme=True,
        input_symmetric=True,  # Symmetric
    )

    can_impl, reason = TritonScaledMMLinearKernel.can_implement(config)
    assert can_impl, (
        f"TritonScaledMMLinearKernel should accept symmetric config: {reason}"
    )


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
