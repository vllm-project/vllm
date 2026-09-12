# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test RMSNorm precision boundary consistency for speculative decoding."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="rms_norm requires a CUDA/ROCm device",
)


def _unfused_rms_norm_reference(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Compute the composite RMSNorm path with scalar_t intermediate rounding."""
    input_fp32 = input_tensor.float()
    variance = input_fp32.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps).to(input_tensor.dtype)
    return inv_rms * input_tensor * weight


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("hidden_size", [512, 2048, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_rmsnorm_precision_intermediate_rounding(
    batch_size: int,
    hidden_size: int,
    dtype: torch.dtype,
):
    """Verify the kernel matches the unfused scalar_t rounding boundary exactly."""
    torch.manual_seed(42)
    torch.set_default_device("cuda")
    eps = 1e-6

    input_tensor = torch.randn(batch_size, hidden_size, dtype=dtype)
    weight = torch.randn(hidden_size, dtype=dtype)

    out = torch.empty_like(input_tensor)
    ops.rms_norm(out, input_tensor, weight, eps)
    reference = _unfused_rms_norm_reference(input_tensor, weight, eps)

    # The CUDA/ROCm kernel must round the normalized value through scalar_t
    # before multiplying by the scalar_t weight, matching the composite path.
    assert torch.equal(out, reference)
