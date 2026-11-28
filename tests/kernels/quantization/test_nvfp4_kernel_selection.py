# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for NVFP4 linear kernel selection order (CPU-only)."""

import pytest
import torch

from vllm.model_executor.kernels.linear import (
    _POSSIBLE_NVFP4_KERNELS,
    CutlassNvFp4LinearKernel,
    FlashInferB12xNvFp4LinearKernel,
    FlashInferCuteDslNvFp4LinearKernel,
    FlashInferCuteDslNvFp4W4A16LinearKernel,
    FlashInferCutlassNvFp4LinearKernel,
)
from vllm.model_executor.kernels.linear.nvfp4 import flashinfer as fi_nvfp4
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability, PlatformEnum
from vllm.utils import flashinfer as fi_utils

# W4A4 kernels that run on SM120/121, where the head of the list is gated to
# sm_10x and selection falls through to whatever follows.
W4A4_KERNELS_ON_SM12X = (
    FlashInferCutlassNvFp4LinearKernel,
    FlashInferB12xNvFp4LinearKernel,
    CutlassNvFp4LinearKernel,
)


@pytest.mark.parametrize("w4a4_kernel", W4A4_KERNELS_ON_SM12X)
def test_w4a16_kernel_does_not_precede_w4a4_kernels(w4a4_kernel):
    candidates = _POSSIBLE_NVFP4_KERNELS[PlatformEnum.CUDA]
    w4a16_index = candidates.index(FlashInferCuteDslNvFp4W4A16LinearKernel)
    assert candidates.index(w4a4_kernel) < w4a16_index, (
        f"{w4a4_kernel.__name__} must be preferred over "
        f"{FlashInferCuteDslNvFp4W4A16LinearKernel.__name__}"
    )


@pytest.mark.parametrize(
    "capability,has_sm12x_runner,expected",
    [
        ((10, 0), False, True),
        ((10, 3), False, True),
        ((12, 0), True, True),
        ((12, 0), False, False),
        ((12, 1), True, True),
        ((9, 0), True, False),
    ],
)
def test_cutedsl_w4a4_supported_on_sm10x_and_sm120(
    monkeypatch, capability, has_sm12x_runner, expected
):
    monkeypatch.setattr(
        type(current_platform),
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(*capability)),
    )
    monkeypatch.setattr(fi_nvfp4, "has_flashinfer", lambda: True)
    monkeypatch.setattr(
        fi_nvfp4,
        "has_flashinfer_sm12x_cutedsl_fp4_gemm",
        lambda: has_sm12x_runner,
    )
    supported, _ = FlashInferCuteDslNvFp4LinearKernel.is_supported()
    assert supported is expected


@pytest.mark.parametrize(
    "n,k,expected",
    [
        (512, 1024, True),
        (248320, 5120, True),
        (160, 1024, False),
        (512, 96, False),
        (262144, 8192, False),
    ],
)
def test_sm120_cutedsl_shape_support(n, k, expected):
    assert fi_nvfp4._sm120_cutedsl_supports_shape(n, k) is expected


@pytest.mark.parametrize(
    "cuda_version,expected", [("13.0", True), ("12.9", False), (None, False)]
)
def test_sm12x_cutedsl_runner_requires_cuda13(monkeypatch, cuda_version, expected):
    monkeypatch.setattr(torch.version, "cuda", cuda_version)
    monkeypatch.setattr(fi_utils, "has_flashinfer_cutedsl", lambda: True)
    monkeypatch.setattr(fi_utils.importlib.util, "find_spec", lambda name: object())
    fi_utils.has_flashinfer_sm12x_cutedsl_fp4_gemm.cache_clear()
    try:
        assert fi_utils.has_flashinfer_sm12x_cutedsl_fp4_gemm() is expected
    finally:
        fi_utils.has_flashinfer_sm12x_cutedsl_fp4_gemm.cache_clear()
