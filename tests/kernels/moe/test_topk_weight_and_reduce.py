# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TopKWeightAndReduceNaiveBatched.apply (PR #53440).

PR #53440 replaced a non-deterministic ``output.index_add_`` (CUDA atomic
floating-point adds, whose accumulation order varies run-to-run when several
experts route to the same token) with a scatter-into-unique-slots followed by a
``moe_sum`` reduction, giving bitwise-identical output across repeated calls.
The PR shipped no tests. ``moe_sum`` is a compiled CUDA kernel, so these run on
GPU.

Covered:
  * numerical equivalence against an independent float64 reference when
    multiple experts contribute to the same token (the case index_add_'s
    ordering affected), for topk in {1, 2, 3};
  * bitwise determinism across repeated calls (fails on the old index_add_
    path, passes on the scatter-then-sum path);
  * the bf16 dtype path;
  * apply_router_weight_on_input=True skips the topk re-weighting.
"""
import pytest
import torch

from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNaiveBatched,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="moe_sum requires CUDA"
)

DEVICE = "cuda"


def _build_batched_inputs(num_tokens, num_experts, K, topk, dtype, seed):
    """Construct (fused_expert_output, topk_weights, topk_ids) in the layout
    apply() expects, plus a float64 reference on CPU.

    fused_expert_output has shape (num_experts, batch_size, K); for expert e,
    row ``slot`` holds e's contribution for the slot-th token routed to e
    (slot = running count of tokens routed to e, in token order).
    """
    g = torch.Generator().manual_seed(seed)

    topk_ids = torch.empty((num_tokens, topk), dtype=torch.int64)
    for t in range(num_tokens):
        topk_ids[t] = torch.randperm(num_experts, generator=g)[:topk]
    topk_weights = torch.rand((num_tokens, topk), generator=g).to(dtype)

    batch_size = num_tokens  # upper bound on tokens routed to any one expert
    fused = torch.zeros((num_experts, batch_size, K), dtype=dtype)
    ref = torch.zeros((num_tokens, K), dtype=torch.float64)

    slot_counter = [0] * num_experts
    for t in range(num_tokens):
        for j in range(topk):
            e = int(topk_ids[t, j].item())
            slot = slot_counter[e]
            slot_counter[e] += 1
            vals = torch.rand((K,), generator=g).to(dtype)
            fused[e, slot, :] = vals
            ref[t] += vals.to(torch.float64) * float(topk_weights[t, j].item())

    return (
        fused.to(DEVICE),
        topk_weights.to(DEVICE),
        topk_ids.to(DEVICE),
        ref,
    )


@pytest.mark.parametrize("topk", [1, 2, 3])
def test_matches_float64_reference_multi_expert_per_token(topk):
    num_tokens, num_experts, K = 6, 4, 8
    fused, topk_weights, topk_ids, ref = _build_batched_inputs(
        num_tokens, num_experts, K, topk, torch.float32, seed=1234
    )
    red = TopKWeightAndReduceNaiveBatched(rank=0)
    out = red.apply(
        output=None,
        fused_expert_output=fused,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        apply_router_weight_on_input=False,
    )
    assert out.shape == (num_tokens, K)
    torch.testing.assert_close(
        out.cpu().to(torch.float64), ref, rtol=1e-5, atol=1e-5
    )


def test_bitwise_deterministic_across_calls():
    num_tokens, num_experts, K, topk = 8, 6, 16, 3
    fused, topk_weights, topk_ids, _ = _build_batched_inputs(
        num_tokens, num_experts, K, topk, torch.bfloat16, seed=99
    )
    red = TopKWeightAndReduceNaiveBatched(rank=0)

    def run():
        return red.apply(
            output=None,
            fused_expert_output=fused.clone(),
            topk_weights=topk_weights.clone(),
            topk_ids=topk_ids.clone(),
            apply_router_weight_on_input=False,
        )

    a = run()
    for _ in range(10):
        assert torch.equal(a, run()), "reduction not bitwise-deterministic"


def test_bf16_path_matches_reference():
    num_tokens, num_experts, K, topk = 5, 4, 8, 2
    fused, topk_weights, topk_ids, ref = _build_batched_inputs(
        num_tokens, num_experts, K, topk, torch.bfloat16, seed=7
    )
    red = TopKWeightAndReduceNaiveBatched(rank=0)
    out = red.apply(
        output=None,
        fused_expert_output=fused,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        apply_router_weight_on_input=False,
    )
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(
        out.cpu().to(torch.float64), ref, rtol=3e-2, atol=3e-2
    )


def test_apply_router_weight_on_input_skips_reweight():
    num_tokens, num_experts, K, topk = 4, 3, 8, 2
    fused, topk_weights, topk_ids, _ = _build_batched_inputs(
        num_tokens, num_experts, K, topk, torch.float32, seed=42
    )
    # Independent unweighted reference (recompute slot map on CPU copy).
    fused_cpu = fused.cpu()
    ids_cpu = topk_ids.cpu()
    ref = torch.zeros((num_tokens, K), dtype=torch.float64)
    slot_counter = [0] * num_experts
    for t in range(num_tokens):
        for j in range(topk):
            e = int(ids_cpu[t, j].item())
            slot = slot_counter[e]
            slot_counter[e] += 1
            ref[t] += fused_cpu[e, slot, :].to(torch.float64)

    red = TopKWeightAndReduceNaiveBatched(rank=0)
    out = red.apply(
        output=None,
        fused_expert_output=fused,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        apply_router_weight_on_input=True,
    )
    torch.testing.assert_close(
        out.cpu().to(torch.float64), ref, rtol=1e-5, atol=1e-5
    )
