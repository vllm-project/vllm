# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ScaledMM kernel selection logic

Run `pytest tests/kernels/quantization/test_scaled_mm_kernel_selection.py`.
"""

import pytest

from vllm.model_executor.layers.quantization.kernels.scaled_mm import (
    ScaledMMLinearLayerConfig,
    choose_scaled_mm_linear_kernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.triton import (
    TritonScaledMMLinearKernel,
)
from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-specific test")
def test_triton_kernel_selected_on_rocm():
    """Test that TritonScaledMMLinearKernel is selected on ROCm
    when Aiter is not available."""
    config = ScaledMMLinearLayerConfig(
        is_channelwise=False,
        is_static_input_scheme=True,
        input_symmetric=True,
    )

    kernel = choose_scaled_mm_linear_kernel(config, compute_capability=None)

    assert kernel == TritonScaledMMLinearKernel, (
        f"Expected TritonScaledMMLinearKernel on ROCm, got {kernel.__name__}"
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-specific test")
def test_triton_kernel_available_on_cuda():
    """Test that TritonScaledMMLinearKernel can be selected on CUDA."""
    config = ScaledMMLinearLayerConfig(
        is_channelwise=False,
        is_static_input_scheme=True,
        input_symmetric=True,
    )

    # Triton should be supported on CUDA
    supported, reason = TritonScaledMMLinearKernel.is_supported()
    assert supported, (
        f"TritonScaledMMLinearKernel should be supported on CUDA: {reason}"
    )

    # It should be able to implement symmetric configs
    can_impl, reason = TritonScaledMMLinearKernel.can_implement(config)
    assert can_impl, (
        f"TritonScaledMMLinearKernel should implement symmetric config: {reason}"
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
