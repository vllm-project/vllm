# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness test for the tiny-dot Triton fast path.

The shared_expert_gate = Linear(hidden, 1) in Qwen2/3/3.5 MoE flows
through `rocm_unquantized_gemm_impl` with m == n == 1.  When the
preconditions (K<=4096, bias is None, bf16/fp16 input) match, the
implementation routes through a small Triton kernel
(_tiny_dot_triton in vllm/model_executor/layers/utils.py) instead of
the eager `(x*w).sum(dtype=x.dtype)` chain.

This test asserts the Triton path produces results that match the
eager reference within bf16/fp16 noise across the K values the
shared_expert_gate actually uses (1024, 2048, 4096) for both dtypes.
"""

from __future__ import annotations

import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="Triton kernel requires CUDA/ROCm")
@pytest.mark.parametrize("K", [32, 1024, 2048, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_tiny_dot_matches_eager(K: int, dtype: torch.dtype):
    """_tiny_dot_triton(x, w) ~= (x.flatten() * w.flatten()).sum(dtype)."""
    from vllm.model_executor.layers.utils import _tiny_dot_triton

    torch.manual_seed(0)
    x = (torch.randn(K, dtype=dtype, device="cuda") * 0.05).contiguous()
    w = (torch.randn(K, dtype=dtype, device="cuda") * 0.05).contiguous()

    ref = (x * w).sum(dtype=x.dtype)
    got = _tiny_dot_triton(x, w)

    # bf16/fp16 tolerance: dot of K terms has ~sqrt(K) accumulated noise
    # relative to the ulp.  Allow 5e-3 absolute and 1e-2 relative -- both
    # paths use fp32 accumulation internally so the bound is generous.
    assert torch.allclose(got, ref, atol=5e-3, rtol=1e-2), (
        f"K={K} {dtype}: got {got.item():.4e}, ref {ref.item():.4e}, "
        f"abs diff {(got.float() - ref.float()).abs().item():.2e}"
    )
