# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness test for the triple-fused GDN WY-chain kernel.

Two paths produce w, u from (k, v, beta, g_cumsum):
  1. unfused: chunk_scaled_dot_kkt_fwd -> solve_tril -> recompute_w_u_fwd
  2. triple fused: fused_kkt_solve_tril_recompute_w_u (skips A entirely)

The two should produce numerically close (w, u) within bf16 noise.

Restricted to BT=64 because that's where the fused kernel is valid
(other BT falls through to the unfused chain in chunk_gated_delta_rule_fwd).
"""

from __future__ import annotations

import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.fixture
def gdn_inputs() -> dict:
    """Representative GDN inputs at BT=64."""
    torch.manual_seed(0)
    B, T, Hg, K, H, V, BT = 1, 64 * 4, 2, 128, 16, 128, 64
    device, dtype = "cuda", torch.bfloat16

    k = torch.randn(B, T, Hg, K, dtype=dtype, device=device) * 0.05
    v = torch.randn(B, T, H, V, dtype=dtype, device=device) * 0.05
    beta = torch.rand(B, T, H, dtype=dtype, device=device) * 0.1
    # g_cumsum: per-token cumulative log gate values, fp32
    g_cumsum = torch.randn(B, T, H, dtype=torch.float32, device=device) * 0.01
    return dict(k=k, v=v, beta=beta, g_cumsum=g_cumsum, BT=BT)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="GDN kernels require CUDA/ROCm")
def test_triple_fused_wy_matches_unfused(gdn_inputs):
    """Triple-fused chain (skips A intermediate) matches unfused."""
    from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import (
        chunk_scaled_dot_kkt_fwd,
    )
    from vllm.model_executor.layers.fla.ops.solve_tril import solve_tril
    from vllm.model_executor.layers.fla.ops.wy_fast import recompute_w_u_fwd
    from vllm.model_executor.layers.fla.ops.wy_fast_doubly_fused import (
        fused_kkt_solve_tril_recompute_w_u_fwd,
    )

    k, v, beta, g = (
        gdn_inputs["k"],
        gdn_inputs["v"],
        gdn_inputs["beta"],
        gdn_inputs["g_cumsum"],
    )

    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g=g,
        cu_seqlens=None,
        chunk_indices=None,
        output_dtype=torch.float32,
    )
    Ai = solve_tril(
        A=A,
        cu_seqlens=None,
        chunk_indices=None,
        output_dtype=k.dtype,
    )
    w_ref, u_ref = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=Ai,
        g_cumsum=g,
        cu_seqlens=None,
        chunk_indices=None,
    )

    w_tri, u_tri = fused_kkt_solve_tril_recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        g_cumsum=g,
        cu_seqlens=None,
        chunk_indices=None,
    )

    # bf16 tolerance: max abs diff at this test shape (T=256) is ~1e-4;
    # allow 5e-4 for headroom across different inputs.  Still well below
    # bf16 ulp at unit scale (~8e-3).
    assert torch.allclose(w_tri, w_ref, atol=5e-4, rtol=1e-3), (
        f"w max abs diff: {(w_tri.float() - w_ref.float()).abs().max():.2e}"
    )
    assert torch.allclose(u_tri, u_ref, atol=5e-4, rtol=1e-3), (
        f"u max abs diff: {(u_tri.float() - u_ref.float()).abs().max():.2e}"
    )
