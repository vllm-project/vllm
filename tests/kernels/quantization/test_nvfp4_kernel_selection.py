# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for NVFP4 linear kernel selection order (CPU-only)."""

import pytest

from vllm.model_executor.kernels.linear import (
    _POSSIBLE_NVFP4_KERNELS,
    CutlassNvFp4LinearKernel,
    FlashInferAutoNvFp4LinearKernel,
    FlashInferB12xNvFp4LinearKernel,
    FlashInferCuteDslNvFp4LinearKernel,
    FlashInferCuteDslNvFp4W4A16LinearKernel,
    FlashInferCutlassNvFp4LinearKernel,
)
from vllm.model_executor.layers.quantization.compressed_tensors.transform.schemes.linear_qutlass_nvfp4 import (  # noqa: E501
    QutlassNvFP4LinearMethod,
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
    ("kernel_cls", "backend"),
    [
        (FlashInferCutlassNvFp4LinearKernel, "cutlass"),
        (FlashInferCuteDslNvFp4LinearKernel, "cute-dsl"),
        (FlashInferAutoNvFp4LinearKernel, "auto"),
    ],
)
def test_qutlass_nvfp4_uses_the_kernel_flashinfer_backend(kernel_cls, backend):
    """The Qutlass NVFP4 transform runs mm_fp4 with the backend of the layer's
    kernel; --linear-backend=flashinfer's kernel must map to "auto" instead of
    failing model loading."""
    kernel = kernel_cls.__new__(kernel_cls)
    assert QutlassNvFP4LinearMethod._get_flashinfer_gemm_backend(kernel) == backend
