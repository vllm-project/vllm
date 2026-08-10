# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the chunked scaled-dot K@K^T Triton kernels: the scalar-gate
variant, which also serves grouped K heads, and the KDA per-channel-gate
variant, which additionally returns the non-strictly causal Aqk.
"""

from itertools import accumulate

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from vllm.third_party.flash_linear_attention.ops.cumsum import chunk_local_cumsum
from vllm.third_party.flash_linear_attention.ops.kda import chunk_kda_scaled_dot_kkt_fwd
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
from vllm.utils.math_utils import RCP_LN2
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="Chunked scaled-dot KKT Triton kernels require CUDA/ROCm or XPU.",
)

DEVICE = current_platform.device_type
BT = FLA_CHUNK_SIZE
TOL = 1e-3


def layout(seqlens, packed):
    if packed:
        cu_seqlens = torch.tensor(
            [0, *accumulate(seqlens)], dtype=torch.int32, device=DEVICE
        )
        return 1, sum(seqlens), cu_seqlens
    assert len(set(seqlens)) == 1, "a padded batch needs equal lengths"
    return len(seqlens), seqlens[0], None


def l2_randn(*shape, dtype):
    return F.normalize(torch.randn(*shape, device=DEVICE), dim=-1).to(dtype)


def make_gate(shape, cu_seqlens, base2=False):
    # Undamped, two thirds of the causal block underflows on both sides.
    raw = -F.softplus(torch.randn(*shape, device=DEVICE)) * 0.25
    gate = chunk_local_cumsum(raw, BT, cu_seqlens=cu_seqlens)
    return gate * RCP_LN2 if base2 else gate


def chunk_spans(B, T, cu_seqlens):
    """(batch, start, end) per chunk."""
    if cu_seqlens is None:
        seqs = [(b, 0, T) for b in range(B)]
    else:
        bounds = cu_seqlens.tolist()
        seqs = [(0, bos, eos) for bos, eos in zip(bounds[:-1], bounds[1:])]
    return [
        (b, s, min(s + BT, eos)) for b, bos, eos in seqs for s in range(bos, eos, BT)
    ]


def block_indices(length, device):
    """Row/column indices broadcasting against an [i, head, j] block."""
    idx = torch.arange(length, device=device)
    return idx[:, None, None], idx


def ref_scaled_dot_kkt(k, g, beta, cu_seqlens):
    B, T, Hg, _ = k.shape
    H = beta.shape[-1]
    k = k.repeat_interleave(H // Hg, dim=2)
    A = torch.zeros(B, T, H, BT, device=k.device, dtype=torch.float32)
    for b, start, end in chunk_spans(B, T, cu_seqlens):
        k_c = k[b, start:end].float()
        beta_c = beta[b, start:end].float().unsqueeze(-1)
        rows, cols = block_indices(end - start, k.device)

        block = torch.einsum("ihd,jhd->ihj", k_c, k_c) * beta_c
        if g is not None:
            g_c = g[b, start:end].float()
            block = block * torch.exp(g_c.unsqueeze(-1) - g_c.t().unsqueeze(0))
        A[b, start:end, :, : end - start] = torch.where(rows > cols, block, 0.0)
    return A


def ref_kda_scaled_dot_kkt(q, k, gk, beta, scale, cu_seqlens):
    B, T, H, _ = k.shape
    A = torch.zeros(B, T, H, BT, device=k.device, dtype=torch.float32)
    Aqk = torch.zeros_like(A)
    for b, start, end in chunk_spans(B, T, cu_seqlens):
        q_c, k_c, g_c = (x[b, start:end].float() for x in (q, k, gk))
        beta_c = beta[b, start:end].float().unsqueeze(-1)
        rows, cols = block_indices(end - start, k.device)

        gate = torch.exp2(g_c.unsqueeze(1) - g_c.unsqueeze(0))
        kkt = torch.einsum("ijhd,ihd,jhd->ihj", gate, k_c, k_c)
        qkt = torch.einsum("ijhd,ihd,jhd->ihj", gate, q_c, k_c)
        A[b, start:end, :, : end - start] = torch.where(rows > cols, kkt * beta_c, 0.0)
        Aqk[b, start:end, :, : end - start] = torch.where(
            rows >= cols, qkt * scale, 0.0
        )
    return A, Aqk


@pytest.mark.parametrize(
    "seqlens, Hg, H, K, use_g, packed",
    [
        ([130], 2, 2, 64, False, False),
        ([130, 130], 2, 2, 64, True, False),
        ([192, 192], 2, 4, 128, True, False),
        ([30, 64, 162], 4, 4, 64, True, True),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_chunk_scaled_dot_kkt(seqlens, Hg, H, K, use_g, packed, dtype):
    set_random_seed(0)
    B, T, cu_seqlens = layout(seqlens, packed)
    k = l2_randn(B, T, Hg, K, dtype=dtype)
    beta = torch.randn(B, T, H, device=DEVICE).sigmoid()
    g = make_gate((B, T, H), cu_seqlens) if use_g else None

    A_ref = ref_scaled_dot_kkt(k, g, beta, cu_seqlens)
    A = chunk_scaled_dot_kkt_fwd(
        k, g=g, beta=beta, cu_seqlens=cu_seqlens, chunk_size=BT
    )

    torch.testing.assert_close(A, A_ref, rtol=TOL, atol=TOL)


@pytest.mark.parametrize(
    "seqlens, H, K, packed",
    [
        ([64], 2, 64, False),
        ([130, 130], 4, 128, False),
        ([256], 1, 32, False),
        ([30, 64, 162], 4, 64, True),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_chunk_kda_scaled_dot_kkt(seqlens, H, K, packed, dtype):
    set_random_seed(0)
    B, T, cu_seqlens = layout(seqlens, packed)
    q = l2_randn(B, T, H, K, dtype=dtype)
    k = l2_randn(B, T, H, K, dtype=dtype)
    beta = torch.randn(B, T, H, device=DEVICE).sigmoid()
    gk = make_gate((B, T, H, K), cu_seqlens, base2=True)
    scale = K**-0.5

    A_ref, Aqk_ref = ref_kda_scaled_dot_kkt(q, k, gk, beta, scale, cu_seqlens)
    A, Aqk = chunk_kda_scaled_dot_kkt_fwd(
        q, k, gk=gk, beta=beta, scale=scale, cu_seqlens=cu_seqlens, chunk_size=BT
    )

    torch.testing.assert_close(A, A_ref, rtol=TOL, atol=TOL)
    torch.testing.assert_close(Aqk, Aqk_ref, rtol=TOL, atol=TOL)
