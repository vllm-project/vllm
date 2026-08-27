# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical tests for the Triton sparse-MLA attention kernel.

Validates ``triton_mla_sparse_attention`` against a naive PyTorch sparse-MLA
reference for both supported geometries:

* NoPE MLA  (``dim_qk == 512``): glm5_next / GLM-5.3-Flash, ``qk_rope_head_dim=0``
* RoPE MLA  (``dim_qk == 576``): DeepSeek-V3.2 / GLM-5 (512 latent + 64 rope)

and across the split-KV counts the dispatcher may choose. The V projection is
always the first 512 (``kv_lora_rank``) lanes of each cached row.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
    triton_mla_sparse_attention,
)

_DV = 512  # kv_lora_rank == V width, for both geometries


def _reference(q, kv, indices, sm_scale):
    """Naive fp32 sparse MLA. q [T,H,D], kv [S,1,D], indices [T,1,topk]."""
    num_tokens, num_heads = q.shape[0], q.shape[1]
    out = torch.zeros(num_tokens, num_heads, _DV, device=q.device, dtype=torch.float32)
    kvf = kv[:, 0, :].float()  # [S, D]
    seq_kv = kvf.shape[0]
    for t in range(num_tokens):
        idx = indices[t, 0]
        valid = idx[(idx >= 0) & (idx < seq_kv)].long()
        if valid.numel() == 0:
            continue
        k = kvf[valid]  # [n, D]
        scores = (q[t].float() @ k.t()) * sm_scale  # [H, n], over full dim_qk
        p = torch.softmax(scores, dim=-1)
        out[t] = p @ k[:, :_DV]  # V == first 512 lanes
    return out


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("dim_qk", [512, 576])
@pytest.mark.parametrize("num_tokens", [1, 8])
@pytest.mark.parametrize("num_kv_splits", [1, 2, 4, None])
def test_triton_mla_sparse_matches_reference(dim_qk, num_tokens, num_kv_splits):
    torch.manual_seed(0)
    device = "cuda"
    num_heads = 128
    seq_kv = 64
    topk = 2048
    n_valid = 40
    sm_scale = 1.0 / (dim_qk**0.5)

    q = torch.randn(num_tokens, num_heads, dim_qk, device=device, dtype=torch.bfloat16)
    kv = torch.randn(seq_kv, 1, dim_qk, device=device, dtype=torch.bfloat16)
    indices = torch.full((num_tokens, 1, topk), -1, device=device, dtype=torch.int32)
    for t in range(num_tokens):
        perm = torch.randperm(seq_kv, device=device)[:n_valid].to(torch.int32)
        indices[t, 0, :n_valid] = perm

    got = triton_mla_sparse_attention(
        q, kv, indices, sm_scale=sm_scale, num_kv_splits=num_kv_splits
    ).float()
    expected = _reference(q, kv, indices, sm_scale)

    torch.testing.assert_close(got, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("dim_qk", [512, 576])
def test_triton_mla_sparse_empty_query_is_zero(dim_qk):
    """A token with no valid KV slots must return zeros, not NaN."""
    torch.manual_seed(0)
    device = "cuda"
    num_heads = 16
    topk = 64
    sm_scale = 1.0 / (dim_qk**0.5)

    q = torch.randn(1, num_heads, dim_qk, device=device, dtype=torch.bfloat16)
    kv = torch.randn(8, 1, dim_qk, device=device, dtype=torch.bfloat16)
    indices = torch.full((1, 1, topk), -1, device=device, dtype=torch.int32)  # all pad

    out = triton_mla_sparse_attention(q, kv, indices, sm_scale=sm_scale).float()
    assert not out.isnan().any()
    assert torch.count_nonzero(out) == 0
