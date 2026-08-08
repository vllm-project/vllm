# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for recompute_w_u_fwd_kernel (FLA/GDN forward recomputation).

Tests the Triton kernel that computes:
  w = A @ (k * beta * exp(g))
  u = A @ (v * beta)

where A is the precomputed block-triangular inverse from solve_tril(),
beta is a per-token scaling factor, and g is a cumulative log-decay.
Used in the GDN (Gated Delta Network) chunked attention forward pass.
"""

import pytest
import torch

from vllm.model_executor.layers.fla.ops.wy_fast import recompute_w_u_fwd
from vllm.platforms import current_platform

DEVICE = current_platform.device_type


def make_strictly_lower_tri(
    B: int, T: int, H: int, BT: int, dtype: torch.dtype
) -> torch.Tensor:
    """Create strictly lower triangular A in chunked layout [B, T, H, BT]."""
    A = torch.randn(B, T, H, BT, dtype=dtype, device=DEVICE) * 0.05
    NT = T // BT
    with torch.no_grad():
        for b in range(B):
            for t_idx in range(NT):
                ts = t_idx * BT
                for h in range(H):
                    for i in range(BT):
                        A[b, ts + i, h, i:] = 0.0
    return A


def recompute_w_u_fwd_ref(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch reference for recompute_w_u_fwd_kernel.

    Args:
        k: [B, T, Hg, K] — keys
        v: [B, T, H, V] — values
        beta: [B, T, H] — per-token scaling
        g_cumsum: [B, T, H, K] — cumulative gate decay
        A: [B, T, H, BT] — block-triangular inverse (chunked)

    Returns:
        w: [B, T, H, K]
        u: [B, T, H, V]
    """
    B, T, Hg, K = k.shape
    H, V = v.shape[2], v.shape[3]
    BT = A.shape[-1]
    NT = T // BT

    w = torch.zeros(B, T, H, K, device=k.device, dtype=k.dtype)
    u = torch.zeros(B, T, H, V, device=v.device, dtype=v.dtype)

    for b in range(B):
        for t_idx in range(NT):
            ts = t_idx * BT
            te = min(ts + BT, T)
            cl = te - ts

            for h in range(H):
                hg = h // (H // Hg) if Hg != H else h

                # Build the BT×BT block from chunked storage
                A_block = torch.zeros(BT, BT, device=A.device, dtype=torch.float32)
                for i in range(cl):
                    A_block[i, :BT] = A[b, ts + i, h, :BT].float()

                b_beta = beta[b, ts:te, h].float()
                b_g = g_cumsum[b, ts:te, h].float()

                # u = A @ (v * beta)
                b_v = v[b, ts:te, h].float()
                b_vb = b_v * b_beta[:, None]
                b_u = A_block[:cl, :cl] @ b_vb
                u[b, ts:te, h] = b_u.to(u.dtype)

                # w = A @ (k * beta * exp(g))
                b_k = k[b, ts:te, hg].float()
                b_kb = b_k * b_beta[:, None] * b_g[:, None].exp()
                b_w = A_block[:cl, :cl] @ b_kb
                w[b, ts:te, h] = b_w.to(w.dtype)

    return w, u


@pytest.mark.parametrize(
    "B,T,H,Hg,K,V,BT,dtype",
    [
        (1, 64, 4, 4, 64, 64, 64, torch.float32),
        (1, 64, 4, 2, 64, 64, 64, torch.float32),
        (2, 64, 4, 4, 64, 64, 64, torch.float32),
        (1, 128, 4, 4, 64, 64, 64, torch.float32),
        (1, 64, 8, 4, 64, 64, 64, torch.float32),
        (1, 64, 4, 4, 128, 64, 64, torch.float32),
        (1, 64, 4, 4, 64, 64, 64, torch.bfloat16),
        (2, 128, 4, 4, 64, 64, 64, torch.bfloat16),
    ],
)
def test_recompute_w_u_fwd(
    B: int, T: int, H: int, Hg: int, K: int, V: int, BT: int, dtype: torch.dtype
) -> None:
    """Verify recompute_w_u_fwd_kernel against PyTorch reference."""
    torch.manual_seed(42)

    k = torch.randn(B, T, Hg, K, device=DEVICE, dtype=dtype)
    v = torch.randn(B, T, H, V, device=DEVICE, dtype=dtype)
    beta = torch.randn(B, T, H, device=DEVICE, dtype=torch.float32).sigmoid()
    g_cumsum = (
        torch.randn(B, T, H, device=DEVICE, dtype=torch.float32).cumsum(dim=1) * 0.1
    )
    A = make_strictly_lower_tri(B, T, H, BT, dtype)

    w, u = recompute_w_u_fwd(k, v, beta, g_cumsum, A, cu_seqlens=None)

    w_ref, u_ref = recompute_w_u_fwd_ref(k, v, beta, g_cumsum, A)

    atol = 1e-2 if dtype == torch.bfloat16 else 1e-3
    rtol = 5e-2 if dtype == torch.bfloat16 else 1e-2

    torch.testing.assert_close(
        w.float().cpu(), w_ref.float().cpu(), atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        u.float().cpu(), u_ref.float().cpu(), atol=atol, rtol=rtol
    )
