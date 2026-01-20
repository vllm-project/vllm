# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ScaledMM kernel selection logic (CPU-only)

Run `pytest tests/kernels/quantization/test_scaled_mm_kernel_selection.py`.
"""

import inspect
from abc import ABC

import pytest
import torch

from vllm.model_executor.layers.quantization.kernels.scaled_mm import (
    Int8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.aiter import (
    AiterInt8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.cpu import (
    CPUInt8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.ScaledMMLinearKernel import (  # noqa: E501
    ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    ScaleDesc,
)

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
    # Helper to build Int8 config
    def make_int8_config(is_channelwise: bool, is_static: bool, symmetric: bool):
        weight_group = GroupShape.PER_TOKEN if is_channelwise else GroupShape.PER_TENSOR
        act_group = GroupShape.PER_TENSOR if is_static else GroupShape.PER_TOKEN
        return Int8ScaledMMLinearLayerConfig(
            weight_quant_key=QuantKey(
                dtype=torch.int8,
                scale=ScaleDesc(torch.float32, True, weight_group),
                symmetric=True,
            ),
            activation_quant_key=QuantKey(
                dtype=torch.int8,
                scale=ScaleDesc(torch.float32, is_static, act_group),
                symmetric=symmetric,
            ),
        )

    configs = [
        make_int8_config(is_channelwise=False, is_static=True, symmetric=True),
        make_int8_config(is_channelwise=True, is_static=False, symmetric=False),
    ]

    for config in configs:
        can_impl, reason = CPUInt8ScaledMMLinearKernel.can_implement(config)
        assert can_impl, (
            f"CPUInt8ScaledMMLinearKernel should accept config {config}: {reason}"
        )
