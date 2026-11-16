# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for NVFP4 linear kernel selection order (CPU-only)."""

import pytest

from vllm.model_executor.kernels.linear import (
    _POSSIBLE_NVFP4_KERNELS,
    CutlassNvFp4LinearKernel,
    FlashInferB12xNvFp4LinearKernel,
    FlashInferCuteDslNvFp4W4A16LinearKernel,
    FlashInferCutlassNvFp4LinearKernel,
    _prioritize_b12x_nvfp4,
)
from vllm.platforms.interface import PlatformEnum

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
    ("compute_capability", "b12x_first"),
    [(120, True), (121, False), (100, False), (90, False)],
)
def test_b12x_precedes_flashinfer_cutlass_on_sm120_only(compute_capability, b12x_first):
    candidates = _POSSIBLE_NVFP4_KERNELS[PlatformEnum.CUDA]
    ordered = _prioritize_b12x_nvfp4(candidates, compute_capability)
    assert (
        ordered.index(FlashInferB12xNvFp4LinearKernel)
        < ordered.index(FlashInferCutlassNvFp4LinearKernel)
    ) == b12x_first
    assert sorted(map(id, ordered)) == sorted(map(id, candidates))
