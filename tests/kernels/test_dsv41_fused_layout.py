# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Index permutations between the standard [h, d] layout and the layouts the
FlashMLA fused sparse-attention kernel reads (Q) and writes (O)."""

import torch

from vllm.models.deepseek_v4_1.common.ops.fused_layout import (
    inverse_permutation,
    o_fused_chunk_permutation,
    o_fused_permutation,
    permute_q_from_fused,
    permute_q_to_fused,
    permute_wo_a_,
    permute_wq_b_,
    q_fused_permutation,
)


def test_q_fused_permutation_matches_kernel_formula():
    num_heads, head_dim = 4, 64
    perm = q_fused_permutation(num_heads, head_dim)
    for h in range(num_heads):
        for d in range(head_dim):
            fused = (d // 16) * (num_heads * 16) + h * 16 + d % 16
            assert perm[fused].item() == h * head_dim + d


def test_o_fused_permutation_matches_kernel_formula():
    perm = o_fused_permutation(8, 512)
    for h in range(8):
        for c in range(16):
            for j in (0, 17, 31):
                fused = (c * 8 + h) * 32 + j
                assert perm[fused].item() == h * 512 + c * 32 + j
    chunk_perm = o_fused_chunk_permutation(8, 512)
    assert torch.equal(chunk_perm, perm[::32] // 32)


def test_inverse_permutation_round_trip():
    perm = q_fused_permutation(16, 512)
    inv = inverse_permutation(perm)
    assert torch.equal(perm[inv], torch.arange(perm.numel()))
    q = torch.randn(3, 16, 512)
    assert torch.equal(permute_q_from_fused(permute_q_to_fused(q)), q)


def test_permuted_wq_b_produces_fused_q():
    torch.manual_seed(0)
    n, heads, k = 5, 16, 1280
    x = torch.randn(n, k, dtype=torch.float64)
    w = torch.randn(heads * 512, k, dtype=torch.float64)
    scale = torch.arange(heads * 512, dtype=torch.int32).view(-1, 1)
    scale = scale.expand(-1, k // 32).to(torch.uint8)
    w_perm, scale_perm = w.clone(), scale.clone()
    permute_wq_b_(w_perm, scale_perm, heads)
    q_std = (x @ w.T).view(n, heads, 512)
    torch.testing.assert_close(
        (x @ w_perm.T).view(n, heads, 512), permute_q_to_fused(q_std)
    )
    perm = q_fused_permutation(heads, 512)
    assert torch.equal(scale_perm, scale[perm])


def test_permuted_wo_a_consumes_fused_o():
    torch.manual_seed(0)
    n, rank, group_in = 5, 1024, 8 * 512
    o_std = torch.randn(n, group_in, dtype=torch.float64)
    w = torch.randn(rank, group_in, dtype=torch.float64)
    scale = torch.arange(group_in // 32, dtype=torch.int32).view(1, -1)
    scale = scale.expand(rank, -1).to(torch.uint8)
    w_perm, scale_perm = w.clone(), scale.clone()
    permute_wo_a_(w_perm, scale_perm, heads_per_group=8)
    o_fused = o_std[:, o_fused_permutation(8, 512)]
    torch.testing.assert_close(o_fused @ w_perm.T, o_std @ w.T)
    assert torch.equal(scale_perm, scale[:, o_fused_chunk_permutation(8, 512)])


def test_permute_helpers_accept_fp8_storage():
    w = torch.randn(16 * 512, 64).to(torch.float8_e4m3fn)
    s = torch.zeros(16 * 512, 2, dtype=torch.uint8)
    permute_wq_b_(w, s, 16)
    w2 = torch.randn(1024, 8 * 512).to(torch.float8_e4m3fn)
    s2 = torch.zeros(1024, 128, dtype=torch.uint8)
    permute_wo_a_(w2, s2)
