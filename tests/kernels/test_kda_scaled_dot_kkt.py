# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision tests for vllm's chunk_kda_scaled_dot_kkt Triton operator.

Compares chunk_kda_scaled_dot_kkt_fwd against a naive PyTorch reference
(float32).
"""

import pytest
import torch

from vllm.model_executor.layers.fla.ops.kda import (
    chunk_kda_scaled_dot_kkt_fwd,
)
from vllm.platforms import current_platform

DEVICE = "xpu" if current_platform.is_xpu() else "cuda"


def chunk_kda_scaled_dot_kkt_ref(q, k, gk, beta, scale, chunk_size=64):
    """Naive PyTorch reference for chunk_kda_scaled_dot_kkt (float32, exp2 gate)."""
    B, T, H, K = k.shape
    BT = chunk_size
    NT = (T + BT - 1) // BT

    A = torch.zeros(B, T, H, BT, device=k.device, dtype=torch.float32)
    Aqk = torch.zeros(B, T, H, BT, device=k.device, dtype=torch.float32)

    for b in range(B):
        for t in range(NT):
            t_start = t * BT
            t_end = min(t_start + BT, T)
            chunk_len = t_end - t_start

            for h in range(H):
                q_chunk = q[b, t_start:t_end, h, :].float()
                k_chunk = k[b, t_start:t_end, h, :].float()
                g_chunk = gk[b, t_start:t_end, h, :].float()
                b_chunk = beta[b, t_start:t_end, h].float()

                for i in range(chunk_len):
                    for j in range(chunk_len):
                        gate = torch.exp2(g_chunk[i] - g_chunk[j])
                        if i > j:
                            A[b, t_start + i, h, j] = (
                                b_chunk[i] * (k_chunk[i] * gate).dot(k_chunk[j])
                            )
                        if i >= j:
                            Aqk[b, t_start + i, h, j] = (
                                scale * (q_chunk[i] * gate).dot(k_chunk[j])
                            )

    return A, Aqk


def _make_inputs(B, T, H, K, dtype=torch.bfloat16):
    q = torch.randn(B, T, H, K, device=DEVICE, dtype=dtype) * 0.1
    k = torch.randn(B, T, H, K, device=DEVICE, dtype=dtype) * 0.1
    gk = torch.randn(B, T, H, K, device=DEVICE, dtype=dtype) * 0.5
    gk = gk.cumsum(dim=1)
    # Gate is computed in fp32; cast gk to float32.
    gk = gk.float()
    beta = torch.randn(B, T, H, device=DEVICE, dtype=dtype) * 0.1
    scale = K**-0.5
    return q, k, gk, beta, scale


@pytest.mark.parametrize(
    ("B", "T", "H", "K", "dtype"),
    [
        (1, 64, 2, 32, torch.float32),
        (1, 64, 2, 64, torch.float32),
        (2, 64, 2, 32, torch.float32),
        (1, 64, 4, 32, torch.float32),
        (1, 128, 2, 32, torch.float32),
        (1, 64, 2, 16, torch.float32),
        (1, 64, 4, 32, torch.bfloat16),
    ],
)
@torch.inference_mode()
def test_chunk_kda_scaled_dot_kkt(B, T, H, K, dtype):
    """chunk_kda_scaled_dot_kkt_fwd must match the naive reference."""
    torch.manual_seed(0)
    q, k, gk, beta, scale = _make_inputs(B, T, H, K, dtype=dtype)

    A_ref, Aqk_ref = chunk_kda_scaled_dot_kkt_ref(q, k, gk, beta, scale)
    A, Aqk = chunk_kda_scaled_dot_kkt_fwd(
        q, k, gk=gk, beta=beta, scale=scale, chunk_size=64
    )

    torch.testing.assert_close(A.float(), A_ref, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(Aqk.float(), Aqk_ref, rtol=5e-2, atol=5e-2)


@torch.inference_mode()
def test_chunk_kda_scaled_dot_kkt_output_structure():
    """A must be strictly lower-triangular within each chunk."""
    torch.manual_seed(0)
    B, T, H, K = 1, 64, 2, 32
    q, k, gk, beta, scale = _make_inputs(B, T, H, K, dtype=torch.float32)

    A, Aqk = chunk_kda_scaled_dot_kkt_fwd(
        q, k, gk=gk, beta=beta, scale=scale, chunk_size=64
    )

    assert A.shape == (B, T, H, 64)
    assert Aqk.shape == (B, T, H, 64)
    assert A.dtype == torch.float32
    assert Aqk.dtype == torch.float32

    # Within each chunk, A[i, j] must be ~0 for j >= i (strict causality).
    for i in range(T):
        chunk_offset = i % 64
        for j in range(chunk_offset, 64):
            assert A[0, i, 0, j].abs().item() < 1e-5, (
                f"A[0,{i},0,{j}]={A[0, i, 0, j].item()} should be ~0"
            )
