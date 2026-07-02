# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision tests for vllm's chunk_scaled_dot_kkt Triton operator.

Compares chunk_scaled_dot_kkt_fwd against a naive PyTorch reference
(float32). The kernel computes A[i, j] = (beta[i] * k[i]) . k[j] with
optional scalar gating exp(g[i] - g[j]), masked to strict lower-triangular
(i > j) within each chunk. Output shape: [B, T, H, BT].
"""

import pytest
import torch

from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from vllm.platforms import current_platform

DEVICE = "xpu" if current_platform.is_xpu() else "cuda"


def chunk_scaled_dot_kkt_ref(k, g=None, beta=None, chunk_size=64):
    """Naive PyTorch reference for chunk_scaled_dot_kkt (float32).

    Args:
        k: [B, T, H, K] — key tensor.
        beta: [B, T, H] — beta scaling.
        g: [B, T, H] — optional cumulative scalar gate.
        chunk_size: BT (default 64).

    Returns:
        A: [B, T, H, BT] — strictly lower-triangular per chunk.
    """
    B, T, H, K = k.shape
    BT = chunk_size
    NT = (T + BT - 1) // BT

    A = torch.zeros(B, T, H, BT, device=k.device, dtype=torch.float32)

    for b in range(B):
        for t_idx in range(NT):
            ts = t_idx * BT
            te = min(ts + BT, T)
            cl = te - ts

            for h in range(H):
                k_c = k[b, ts:te, h].float()  # [cl, K]
                b_c = beta[b, ts:te, h].float()  # [cl]

                a = (k_c * b_c[:, None]) @ k_c.t()  # [cl, cl]

                if g is not None:
                    g_c = g[b, ts:te, h].float()
                    a = a * torch.exp(g_c[:, None] - g_c[None, :])

                # Strict lower-triangular mask (i > j).
                idx = torch.arange(cl, device=k.device)
                a = a * (idx[:, None] > idx[None, :]).float()

                A[b, ts:te, h, :cl] = a

    return A


def _make_inputs(B, T, H, K, dtype=torch.float32, use_g=True):
    k = torch.randn(B, T, H, K, device=DEVICE, dtype=dtype) * 0.1
    beta = torch.randn(B, T, H, device=DEVICE, dtype=torch.float32).sigmoid()
    g = (
        torch.randn(B, T, H, device=DEVICE, dtype=torch.float32) * 0.1
        if use_g
        else None
    )
    return k, beta, g


# (B, T, H, K, use_g) — T is always a multiple of 64.
CONFIGS = [
    (1, 64, 2, 64, True),
    (1, 64, 2, 64, False),  # no gating
    (1, 128, 2, 64, True),  # two chunks
    (2, 64, 2, 64, True),  # batch > 1
    (1, 64, 4, 64, True),  # more heads
    (1, 64, 2, 128, True),  # K = 128
    (1, 64, 4, 32, True),  # K = 32
    (1, 192, 2, 64, True),  # three chunks
]


@pytest.mark.parametrize(
    "B,T,H,K,use_g",
    CONFIGS,
    ids=[f"B{b}_T{t}_H{h}_K{kk}_g{int(ug)}" for b, t, h, kk, ug in CONFIGS],
)
@torch.inference_mode()
def test_chunk_scaled_dot_kkt(B, T, H, K, use_g):
    """chunk_scaled_dot_kkt_fwd must match the naive reference (fp32)."""
    torch.manual_seed(0)
    k, beta, g = _make_inputs(B, T, H, K, use_g=use_g)

    A = chunk_scaled_dot_kkt_fwd(k, g=g, beta=beta, chunk_size=64)
    A_ref = chunk_scaled_dot_kkt_ref(k, g=g, beta=beta, chunk_size=64)

    assert A.shape == A_ref.shape
    assert not torch.isnan(A).any()
    torch.testing.assert_close(A.float(), A_ref, rtol=1e-3, atol=1e-3)


@torch.inference_mode()
def test_chunk_scaled_dot_kkt_output_structure():
    """A must be strictly lower-triangular within each chunk."""
    torch.manual_seed(0)
    B, T, H, K = 1, 64, 2, 64
    k, beta, g = _make_inputs(B, T, H, K)

    A = chunk_scaled_dot_kkt_fwd(k, g=g, beta=beta, chunk_size=64)

    assert A.shape == (B, T, H, 64)
    assert A.dtype == torch.float32

    # Within each chunk, A[i, j] must be ~0 for j >= i (strict causality).
    # Row i sits at within-chunk position (i % 64); every column j at or past
    # that position must vanish.
    col = torch.arange(64, device=A.device)
    row_pos = (torch.arange(T, device=A.device) % 64).unsqueeze(-1)  # [T, 1]
    upper = col.unsqueeze(0) >= row_pos  # [T, 64] positions that must be ~0
    masked = A[0, :, :, :].abs() * upper.unsqueeze(1)  # broadcast over heads
    assert masked.max().item() < 1e-5, "A is not strictly lower-triangular"
