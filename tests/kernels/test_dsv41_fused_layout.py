# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Index permutations between the standard [h, d] layout and the layouts the
FlashMLA fused sparse-attention kernel reads (Q) and writes (O)."""

import pytest
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


def test_fused_attention_finalize_permutes_only_loaded_layers():
    from types import SimpleNamespace

    from vllm.models.deepseek_v4_1.nvidia.flashmla_fused import (
        DeepseekV4FlashMLAFusedAttention,
    )

    def fake_attn(prefix):
        w_q = torch.randn(16 * 512, 64).to(torch.float8_e4m3fn)
        s_q = torch.randint(0, 255, (16 * 512, 2), dtype=torch.uint8)
        w_o = torch.randn(2 * 1024, 8 * 512).to(torch.float8_e4m3fn)
        s_o = torch.randint(0, 255, (2 * 1024, 128), dtype=torch.uint8)
        return SimpleNamespace(
            prefix=prefix,
            n_local_heads=16,
            n_local_groups=2,
            wq_b=SimpleNamespace(weight=w_q, weight_scale=s_q),
            wo_a=SimpleNamespace(weight=w_o, weight_scale=s_o),
        )

    attn = fake_attn("model.layers.3.attn")
    before = (attn.wq_b.weight.clone(), attn.wo_a.weight.clone())
    finalize = DeepseekV4FlashMLAFusedAttention.finalize_loaded_weights
    finalize(attn, {"model.layers.4.attn.wq_b.weight"})
    assert torch.equal(attn.wq_b.weight.view(torch.uint8), before[0].view(torch.uint8))
    finalize(
        attn, {"model.layers.3.attn.wq_b.weight", "model.layers.3.attn.wo_a.weight"}
    )
    perm = q_fused_permutation(16, 512)
    assert torch.equal(
        attn.wq_b.weight.view(torch.uint8), before[0].view(torch.uint8)[perm]
    )
    assert torch.equal(
        attn.wo_a.weight.view(torch.uint8),
        before[1].view(torch.uint8)[:, o_fused_permutation(8, 512)],
    )
    # None means "everything was (re)loaded": permutes again (dummy weights).
    finalize(attn, None)
    assert torch.equal(
        attn.wq_b.weight.view(torch.uint8), before[0].view(torch.uint8)[perm][perm]
    )


@pytest.mark.parametrize("num_tokens", [1, 5, 129])
def test_inv_rope_quant_permuted_output_matches_standard(num_tokens):
    """The split-KV fallback's O quant must emit the fused kernel's layout."""
    from tests.kernels.attention.test_flashmla_fused_sparse import (
        make_cos_sin_cache,
    )
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
        fused_inv_rope_fp8_quant,
    )

    device = torch.device("cuda")
    n_groups = 2
    o = torch.randn(num_tokens, n_groups * 8, 512, device=device, dtype=torch.bfloat16)
    positions = torch.randint(0, 4096, (num_tokens,), device=device)
    cos_sin = make_cos_sin_cache(4096, device)
    kwargs = dict(
        n_groups=n_groups,
        heads_per_group=8,
        quant_group_size=32,
        tma_aligned_scales=True,
    )
    std_fp8, std_sf = fused_inv_rope_fp8_quant(o, positions, cos_sin, **kwargs)
    perm_fp8, perm_sf = fused_inv_rope_fp8_quant(
        o, positions, cos_sin, permuted_output=True, **kwargs
    )
    perm = o_fused_permutation(8, 512).to(device)
    assert torch.equal(perm_fp8.view(torch.uint8), std_fp8.view(torch.uint8)[..., perm])
    chunk_perm = o_fused_chunk_permutation(8, 512).to(device)
    std_bytes = std_sf.contiguous().view(torch.uint8).view(num_tokens, n_groups, 128)
    perm_bytes = perm_sf.contiguous().view(torch.uint8).view(num_tokens, n_groups, 128)
    assert torch.equal(perm_bytes, std_bytes[..., chunk_perm])
    assert perm_sf.stride() == std_sf.stride()
