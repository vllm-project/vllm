# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MXFP8 linear kernel selection logic (CPU-only)

Run `pytest tests/kernels/quantization/test_mxfp8_kernel_selection.py`.
"""

from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.kernels.linear import init_mxfp8_linear_kernel
from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
    FlashInferCutlassMxfp8LinearKernel,
)
from vllm.model_executor.kernels.linear.mxfp8.marlin import MarlinMxfp8LinearKernel
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    _mxfp8_e4m3_quantize_impl,
)
from vllm.platforms import PlatformEnum

pytestmark = pytest.mark.cpu_test


def test_flashinfer_cutlass_kernel_requires_flashinfer():
    """The kernel dispatches to FlashInfer, so it must not report itself as
    supported when FlashInfer is absent."""
    with patch(
        "vllm.model_executor.kernels.linear.mxfp8.flashinfer.has_flashinfer",
        return_value=False,
    ):
        is_supported, reason = FlashInferCutlassMxfp8LinearKernel.is_supported()

    assert not is_supported
    assert reason


def test_flashinfer_cutlass_kernel_supported_on_sm100_and_above():
    """The device gate stays `>= sm_100`: FlashInfer ships a CUTLASS MXFP8
    GEMM for consumer Blackwell (sm_120) as well, so narrowing this to the
    sm_100 family would drop a working path."""
    import vllm.model_executor.kernels.linear.mxfp8.flashinfer as fi_mod

    with (
        patch.object(fi_mod, "has_flashinfer", return_value=True),
        patch.object(fi_mod.current_platform, "is_cuda", return_value=True),
        patch.object(
            fi_mod.current_platform, "has_device_capability", return_value=True
        ),
    ):
        is_supported, reason = FlashInferCutlassMxfp8LinearKernel.is_supported()

    assert is_supported
    assert reason is None


@patch("vllm.model_executor.kernels.linear.current_platform")
def test_selection_skips_flashinfer_when_unavailable(platform_mock):
    """Without FlashInfer the selector must fall through to a kernel that can
    actually run, rather than picking one whose dependency is missing."""
    platform_mock._enum = PlatformEnum.CUDA

    with (
        patch(
            "vllm.model_executor.kernels.linear.mxfp8.flashinfer.has_flashinfer",
            return_value=False,
        ),
        patch(
            "vllm.model_executor.kernels.linear.mxfp8.flashinfer."
            "has_flashinfer_cutedsl",
            return_value=False,
        ),
        patch.object(
            MarlinMxfp8LinearKernel,
            "is_supported",
            classmethod(lambda cls: (True, None)),
        ),
    ):
        kernel = init_mxfp8_linear_kernel()

    assert not isinstance(kernel, FlashInferCutlassMxfp8LinearKernel)


def test_quantizer_falls_back_to_torch_without_flashinfer():
    """The activation quantizer has a torch implementation; a Blackwell device
    without FlashInfer must reach it instead of raising ImportError."""
    x = torch.randn(4, 64, dtype=torch.bfloat16)

    with (
        patch(
            "vllm.platforms.current_platform.has_device_capability", return_value=True
        ),
        patch("vllm.utils.flashinfer.has_flashinfer", return_value=False),
    ):
        x_q, x_scales = _mxfp8_e4m3_quantize_impl(x)

    assert x_q.shape == x.shape
    assert x_scales.numel() == x.numel() // 32
