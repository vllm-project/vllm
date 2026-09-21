# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MXFP8 linear kernel selection logic (CPU-only)

Run `pytest tests/kernels/quantization/test_mxfp8_kernel_selection.py`.
"""

from unittest.mock import patch

import pytest

from vllm.model_executor.kernels.linear import (
    FlashInferCutedslMxfp8LinearKernel,
    FlashInferCutlassMxfp8LinearKernel,
    MarlinMxfp8LinearKernel,
    Mxfp8LinearLayerConfig,
    init_mxfp8_linear_kernel,
)
from vllm.platforms import PlatformEnum

pytestmark = pytest.mark.cpu_test

# Kernels backed by FlashInfer mm_mxfp8, which requires N, K >= 128 and
# K % 32 == 0.
_MM_MXFP8_KERNELS = [
    FlashInferCutedslMxfp8LinearKernel,
    FlashInferCutlassMxfp8LinearKernel,
]

_SUPPORTED_SHAPE = (4096, 4096)
# (N, K): N < 128, K < 128, K % 32 != 0.
_UNSUPPORTED_SHAPES = [(64, 4096), (4096, 64), (4096, 4112)]


@pytest.mark.parametrize("kernel_cls", _MM_MXFP8_KERNELS)
def test_mm_mxfp8_kernels_accept_supported_shape(kernel_cls):
    config = Mxfp8LinearLayerConfig(weight_shape=_SUPPORTED_SHAPE)
    can_implement, reason = kernel_cls.can_implement(config)
    assert can_implement, reason


@pytest.mark.parametrize("kernel_cls", _MM_MXFP8_KERNELS)
@pytest.mark.parametrize("weight_shape", _UNSUPPORTED_SHAPES)
def test_mm_mxfp8_kernels_reject_unsupported_shape(kernel_cls, weight_shape):
    config = Mxfp8LinearLayerConfig(weight_shape=weight_shape)
    can_implement, reason = kernel_cls.can_implement(config)
    assert not can_implement
    assert reason


@pytest.mark.parametrize(
    ("weight_shape", "expected_kernel_cls"),
    [
        (_SUPPORTED_SHAPE, FlashInferCutedslMxfp8LinearKernel),
        ((64, 4096), MarlinMxfp8LinearKernel),
    ],
)
def test_init_mxfp8_linear_kernel_falls_back_on_unsupported_shape(
    weight_shape, expected_kernel_cls
):
    """A layer that mm_mxfp8 cannot handle must fall through to the next
    kernel in the CUDA priority list instead of being selected and failing
    in apply_weights."""
    with (
        patch("vllm.model_executor.kernels.linear.current_platform") as platform,
        patch.object(
            FlashInferCutedslMxfp8LinearKernel,
            "is_supported",
            return_value=(True, None),
        ),
        patch.object(
            FlashInferCutlassMxfp8LinearKernel,
            "is_supported",
            return_value=(True, None),
        ),
        patch.object(
            MarlinMxfp8LinearKernel, "is_supported", return_value=(True, None)
        ),
    ):
        platform._enum = PlatformEnum.CUDA
        kernel = init_mxfp8_linear_kernel(weight_shape=weight_shape)

    assert isinstance(kernel, expected_kernel_cls)
    assert kernel.config == Mxfp8LinearLayerConfig(weight_shape=weight_shape)
