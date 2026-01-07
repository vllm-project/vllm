# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MixedPrecision kernel selection logic (CPU-only)

Run `pytest tests/kernels/quantization/test_mixed_precision_kernel_selection.py`.
"""

import inspect
from abc import ABC

import pytest

from vllm.model_executor.layers.quantization.kernels.mixed_precision import (
    _POSSIBLE_KERNELS,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.allspark import (  # noqa: E501
    AllSparkLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.bitblas import (  # noqa: E501
    BitBLASLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.conch import (  # noqa: E501
    ConchLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.MPLinearKernel import (  # noqa: E501
    MPLinearKernel,
)

pytestmark = pytest.mark.cpu_test


def test_is_supported_is_abstract():
    """Test that is_supported() is properly defined as abstract."""
    assert issubclass(MPLinearKernel, ABC)
    assert hasattr(MPLinearKernel, "is_supported")


def test_all_kernels_implement_is_supported_implemented():
    """Test that all kernels implement is_supported() method."""
    for kernel in _POSSIBLE_KERNELS:
        assert hasattr(kernel, "is_supported"), (
            f"{kernel.__name__} missing is_supported() method"
        )
        # Verify it's a classmethod by checking if it can be called with the class
        # and by checking the method type
        assert inspect.ismethod(kernel.is_supported) or inspect.isfunction(
            kernel.is_supported
        ), f"{kernel.__name__}.is_supported() should be a classmethod"
        # Verify it can be called as a classmethod
        result, reason = kernel.is_supported()
        assert isinstance(result, bool), "is_supported() should return a bool"
        assert reason is None or isinstance(reason, str), "reason should be str or None"


def test_compute_capability_check():
    """Test that is_supported() correctly checks compute capability."""
    for kernel in [
        AllSparkLinearKernel,
        BitBLASLinearKernel,
        ConchLinearKernel,
    ]:
        result, reason = kernel.is_supported(compute_capability=0)
        assert result is False
        assert reason is not None and "capability" in reason.lower()

        result, reason = kernel.is_supported(compute_capability=100)
        assert result is True
        assert reason is None
