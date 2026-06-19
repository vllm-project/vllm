# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coverage for the FlashInfer BF16 GEMM dispatch path.

Targets the two failure modes that previously slipped through review:
  1. Fake-op signature drift versus the registered impl (PyTorch raises
     TypeError during torch.compile / fake dispatch).
  2. Wrong output shape from the fake impl (silently propagates bogus
     shape metadata into downstream compile passes).
"""

import inspect

import pytest
import torch

from vllm.model_executor.layers.utils import (
    cuda_flashinfer_bf16_gemm_fake,
    cuda_flashinfer_bf16_gemm_impl,
)
from vllm.platforms import current_platform

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


def test_flashinfer_mm_bf16_fake_output_shape():
    """Exercise the registered fake for vllm::flashinfer_mm_bf16 via meta
    tensors. Catches the [M, K] vs [M, N] shape bug in the fake impl.

    The op is only registered when FlashInfer is importable, so skip
    otherwise.
    """
    from vllm.utils.flashinfer import has_flashinfer

    if not has_flashinfer():
        pytest.skip("FlashInfer not installed; op is not registered.")

    # Importing the module triggers op registration (gated on has_flashinfer).
    import vllm.utils.flashinfer  # noqa: F401

    A = torch.empty(4, 16, dtype=torch.bfloat16, device="meta")
    # B is the post-transpose tensor the wrapper passes: [K, N]
    B = torch.empty(16, 8, dtype=torch.bfloat16, device="meta")
    out = torch.ops.vllm.flashinfer_mm_bf16(A, B, None)
    assert out.shape == (4, 8)
    assert out.dtype == torch.bfloat16
