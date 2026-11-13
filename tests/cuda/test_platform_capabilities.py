# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test platform capability detection for DeepGEMM support."""

from unittest.mock import patch

from vllm.platforms.cuda import CudaPlatform


def test_gh200_excluded_from_deep_gemm():
    """GH200 (integrated GPU, capability 90) should not support DeepGEMM."""
    with (
        patch.object(
            CudaPlatform,
            "is_integrated_gpu",
            classmethod(lambda cls, device_id=0: True),
        ),
        patch.object(
            CudaPlatform,
            "is_device_capability",
            classmethod(lambda cls, cap, device_id=0: cap == 90),
        ),
        patch.object(
            CudaPlatform,
            "is_device_capability_family",
            classmethod(lambda cls, cap, device_id=0: False),
        ),
    ):
        assert CudaPlatform.support_deep_gemm() is False


def test_hopper_discrete_supports_deep_gemm():
    """Discrete Hopper (non-integrated, capability 90) supports DeepGEMM."""
    with (
        patch.object(
            CudaPlatform,
            "is_integrated_gpu",
            classmethod(lambda cls, device_id=0: False),
        ),
        patch.object(
            CudaPlatform,
            "is_device_capability",
            classmethod(lambda cls, cap, device_id=0: cap == 90),
        ),
        patch.object(
            CudaPlatform,
            "is_device_capability_family",
            classmethod(lambda cls, cap, device_id=0: False),
        ),
    ):
        assert CudaPlatform.support_deep_gemm() is True


def test_dgx_spark_blackwell_uma_supports_deep_gemm():
    """DGX Spark (integrated GPU, SM 100 family) should support DeepGEMM."""
    with (
        patch.object(
            CudaPlatform,
            "is_integrated_gpu",
            classmethod(lambda cls, device_id=0: True),
        ),
        patch.object(
            CudaPlatform,
            "is_device_capability",
            classmethod(lambda cls, cap, device_id=0: False),
        ),
        patch.object(
            CudaPlatform,
            "is_device_capability_family",
            classmethod(lambda cls, cap, device_id=0: cap == 100),
        ),
    ):
        assert CudaPlatform.support_deep_gemm() is True
