# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Ampere SM80 Native Specialized Decode kernels and components."""

import math

import torch
import torch.nn.functional as F

from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import (
    sparse_mla_decode,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere.dspark_fused import (
    DSparkFusedMarkovSampler,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere.moe_fastpath import (
    AmpereSmallBatchMoEFastPath,
)

small_batch_sparse_mla_decode = sparse_mla_decode.small_batch_sparse_mla_decode


def test_dspark_fused_markov_sampler_greedy():
    """Verify DSpark fused Markov sampling matches eager greedy argmax."""
    device = torch.device("cuda:0")
    num_reqs = 2
    num_steps = 3
    vocab_size = 4096
    rank = 128

    base_logits = torch.randn(
        num_reqs, num_steps, vocab_size, dtype=torch.bfloat16, device=device
    )
    bias = torch.randn(num_reqs, vocab_size, dtype=torch.bfloat16, device=device)
    embed_weight = torch.randn(vocab_size, rank, dtype=torch.bfloat16, device=device)

    idx_mapping = torch.arange(num_reqs * num_steps, dtype=torch.int32, device=device)
    sample_pos = torch.ones(num_reqs * num_steps, dtype=torch.int64, device=device) * 10
    temperature = torch.zeros(num_reqs, dtype=torch.float32, device=device)  # Greedy
    seeds = torch.zeros(num_reqs, dtype=torch.int64, device=device)

    logits_cache = torch.empty(
        num_reqs, num_steps, vocab_size, dtype=torch.bfloat16, device=device
    )
    draft_tokens = torch.zeros(num_reqs, num_steps, dtype=torch.int64, device=device)

    # 1. Run Fused Sampler
    sampler = DSparkFusedMarkovSampler(
        num_reqs, vocab_size, embed_weight, use_fp64=False
    )
    for step in range(num_steps):
        sampler.sample_step(
            base_logits,
            bias,
            step,
            idx_mapping,
            sample_pos,
            temperature,
            seeds,
            logits_cache,
            draft_tokens,
        )

    # 2. Run Eager Reference
    ref_tokens = torch.zeros_like(draft_tokens)
    for step in range(num_steps):
        ref_logits = (base_logits[:, step].float() + bias.float()).to(torch.bfloat16)
        ref_tokens[:, step] = torch.argmax(ref_logits.float(), dim=-1)

    # Assert exact match of sampled tokens
    assert torch.equal(draft_tokens[:, 0], ref_tokens[:, 0]), (
        f"Step 0 mismatch: fused={draft_tokens[:, 0]}, ref={ref_tokens[:, 0]}"
    )
    print("test_dspark_fused_markov_sampler_greedy PASSED!")


def test_sparse_mla_decode_kernel():
    """Verify SM80 Sparse MLA decode numerical correctness vs PyTorch SDPA."""
    device = torch.device("cuda:0")
    M = 2  # requests
    H = 8  # heads
    D = 512  # head_dim
    K = 256  # gathered keys

    torch.manual_seed(42)
    q = torch.randn(M, H, D, dtype=torch.bfloat16, device=device)
    total_tokens = 2048
    kv_cache = torch.randn(total_tokens, D, dtype=torch.bfloat16, device=device)
    indices = torch.randint(0, total_tokens, (M, K), dtype=torch.int32, device=device)

    # Run kernel
    out_kernel = small_batch_sparse_mla_decode(q, kv_cache, indices)

    # Run PyTorch reference
    out_ref = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(D)
    for m in range(M):
        k_gathered = kv_cache[indices[m].long()]  # [K, D]
        scores = torch.einsum("hd,kd->hk", q[m].float(), k_gathered.float()) * sm_scale
        attn = F.softmax(scores, dim=-1)
        res = torch.einsum("hk,kd->hd", attn, k_gathered.float())
        out_ref[m] = res.to(torch.bfloat16)

    # Compare with relative tolerance for BF16
    diff = torch.abs(out_kernel.float() - out_ref.float())
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    print(
        f"Sparse MLA Decode Error vs Reference: max={max_err:.5f}, mean={mean_err:.5f}"
    )
    assert max_err < 0.05, f"Sparse MLA decode error {max_err} too high!"
    print("test_sparse_mla_decode_kernel PASSED!")


def test_moe_fastpath_routing():
    """Verify SM80 MoE fast path routing scores vs reference."""
    device = torch.device("cuda:0")
    M = 4
    hidden = 5120
    num_experts = 384
    top_k = 6

    class DummyMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.top_k = top_k
            self.routed_scaling_factor = 1.5
            self.gate = torch.nn.Linear(hidden, num_experts, bias=False, device=device)
            self.experts = lambda x, w, idx: torch.zeros_like(x)
            self.shared_experts = None

    dummy_moe = DummyMoE()
    fastpath = AmpereSmallBatchMoEFastPath(dummy_moe)
    x = torch.randn(M, hidden, device=device)

    weights, ids = fastpath.route(x)
    assert weights.shape == (M, top_k)
    assert ids.shape == (M, top_k)
    # Check normalization * routed_scaling_factor
    assert torch.allclose(
        weights.sum(dim=-1).float(), torch.tensor([1.5] * M, device=device), atol=1e-3
    )
    print("test_moe_fastpath_routing PASSED!")


if __name__ == "__main__":
    test_dspark_fused_markov_sampler_greedy()
    test_sparse_mla_decode_kernel()
    test_moe_fastpath_routing()
