# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coverage for the FlashInfer BF16 GEMM dispatch path.

Targets the two fake-op failure modes that previously slipped through review:
  1. Fake-op signature drift versus the registered impl (PyTorch raises
     TypeError during torch.compile / fake dispatch).
  2. Wrong output shape from the fake impl (silently propagates bogus
     shape metadata into downstream compile passes).
"""

import inspect

import pytest
import torch

from vllm.model_executor.layers.utils import (
    cuda_flashinfer_bf16_gemm,
    cuda_flashinfer_bf16_gemm_fake,
    cuda_flashinfer_bf16_gemm_impl,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import flashinfer_bf16_mm

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="FlashInfer BF16 GEMM is CUDA-only",
)


def test_cuda_flashinfer_bf16_gemm_fake_matches_impl_signature():
    impl_params = list(inspect.signature(cuda_flashinfer_bf16_gemm_impl).parameters)
    fake_params = list(inspect.signature(cuda_flashinfer_bf16_gemm_fake).parameters)
    assert impl_params == fake_params, (
        f"fake/impl signature mismatch: impl={impl_params} fake={fake_params}"
    )


def test_flashinfer_bf16_gemm_keeps_pdl_disabled_by_default():
    functions = (
        cuda_flashinfer_bf16_gemm_impl,
        cuda_flashinfer_bf16_gemm_fake,
        cuda_flashinfer_bf16_gemm,
        flashinfer_bf16_mm,
    )
    assert all(
        inspect.signature(function).parameters["pdl"].default is False
        for function in functions
    )


@pytest.mark.parametrize(
    "x_shape, n",
    [
        ((4, 16), 8),
        ((2, 3, 16), 8),
        ((1, 1, 1, 32), 64),
    ],
)
def test_cuda_flashinfer_bf16_gemm_fake_preserves_leading_dims(x_shape, n):
    k = x_shape[-1]
    x = torch.empty(x_shape, dtype=torch.bfloat16, device="meta")
    weight = torch.empty(n, k, dtype=torch.bfloat16, device="meta")
    out = cuda_flashinfer_bf16_gemm_fake(x, weight, None)
    assert out.shape == (*x_shape[:-1], n)
    assert out.dtype == torch.bfloat16
