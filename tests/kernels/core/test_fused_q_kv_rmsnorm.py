# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness + large-token-count launch tests for fused_q_kv_rmsnorm.

Before the grid-dim fix the kernel used grid ``(2, num_tokens)``, which hit
CUDA's 65535 grid-y cap for ``num_tokens >= 65536`` and failed with
``Triton Error [CUDA]: invalid argument`` at every large chunked-prefill
profile run. These tests pin the new grid layout.
"""

from __future__ import annotations

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.deepseek_v4_ops import fused_q_kv_rmsnorm

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fused_q_kv_rmsnorm requires a CUDA/ROCm device",
)


def _ref_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x_f32 = x.to(torch.float32)
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    y = x_f32 * torch.rsqrt(variance + eps) * w.to(torch.float32)
    return y.to(x.dtype)


@pytest.mark.parametrize("num_tokens", [1, 17, 1024, 8192])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_q_kv_rmsnorm_correctness(num_tokens: int, dtype: torch.dtype):
    torch.manual_seed(0)
    device = "cuda"
    q_size, kv_size = 192, 576
    qr = torch.randn(num_tokens, q_size, dtype=dtype, device=device)
    kv = torch.randn(num_tokens, kv_size, dtype=dtype, device=device)
    qw = torch.randn(q_size, dtype=dtype, device=device)
    kvw = torch.randn(kv_size, dtype=dtype, device=device)
    eps = 1e-6

    qr_out, kv_out = fused_q_kv_rmsnorm(qr, kv, qw, kvw, eps)

    qr_ref = _ref_rmsnorm(qr, qw, eps)
    kv_ref = _ref_rmsnorm(kv, kvw, eps)

    tol = dict(rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(qr_out, qr_ref, **tol)
    torch.testing.assert_close(kv_out, kv_ref, **tol)


@pytest.mark.parametrize("num_tokens", [65535, 65536, 131072])
def test_fused_q_kv_rmsnorm_launches_past_grid_y_cap(num_tokens: int):
    """Regression guard: grid used to be (2, num_tokens), hitting CUDA's
    65535 grid-y cap at num_tokens >= 65536. The new grid (num_tokens, 2)
    lifts that bound to 2**31-1."""
    device = "cuda"
    dtype = torch.bfloat16
    q_size, kv_size = 192, 576
    qr = torch.randn(num_tokens, q_size, dtype=dtype, device=device)
    kv = torch.randn(num_tokens, kv_size, dtype=dtype, device=device)
    qw = torch.randn(q_size, dtype=dtype, device=device)
    kvw = torch.randn(kv_size, dtype=dtype, device=device)

    qr_out, kv_out = fused_q_kv_rmsnorm(qr, kv, qw, kvw, 1e-6)
    # spot-check a couple of rows against the torch reference
    for row in (0, num_tokens // 2, num_tokens - 1):
        torch.testing.assert_close(
            qr_out[row],
            _ref_rmsnorm(qr[row : row + 1], qw, 1e-6)[0],
            rtol=1e-2,
            atol=1e-2,
        )
        torch.testing.assert_close(
            kv_out[row],
            _ref_rmsnorm(kv[row : row + 1], kvw, 1e-6)[0],
            rtol=1e-2,
            atol=1e-2,
        )
