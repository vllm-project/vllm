# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

pytest.importorskip("triton")

from vllm.model_executor.models.eagle3_aux_rmsnorm import fused_dual_rmsnorm_cat
from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_rocm() or not torch.cuda.is_available(),
    reason="requires ROCm",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("leading_shape", [(7,), (2, 7)])
def test_fused_dual_rmsnorm_cat_matches_reference(
    dtype: torch.dtype, leading_shape: tuple[int, ...]
):
    torch.manual_seed(0)
    hidden_size = 96
    a = torch.randn(*leading_shape, hidden_size, device="cuda", dtype=dtype)
    b = torch.randn(*leading_shape, hidden_size, device="cuda", dtype=dtype)
    w_a = torch.randn(hidden_size, device="cuda", dtype=dtype)
    w_b = torch.randn(hidden_size, device="cuda", dtype=dtype)
    eps = 1e-6

    actual = fused_dual_rmsnorm_cat(a, b, w_a, w_b, eps)

    def rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        out = x.to(torch.float32) * torch.rsqrt(variance + eps) * weight
        return out.to(dtype)

    expected = torch.cat([rmsnorm(a, w_a), rmsnorm(b, w_b)], dim=-1)
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
