# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for EMA load-penalized MoE routing.

Covers adjust_logits_ema and build_expert_to_rank without requiring
a GPU or distributed setup.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.router.balance_routing_ema import (
    adjust_logits_ema,
    build_expert_to_rank,
)

# ---------------------------------------------------------------------------
# build_expert_to_rank
# ---------------------------------------------------------------------------

def test_build_expert_to_rank_linear_exact_multiple():
    """8 experts, EP4 — 2 experts per rank, no remainder."""
    e2r = build_expert_to_rank(8, 4, "linear", torch.device("cpu"))
    expected = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    assert torch.equal(e2r, expected)


def test_build_expert_to_rank_linear_non_multiple():
    """60 experts at EP8 — remainder 4, ranks 0-3 get 8 experts, 4-7 get 7."""
    e2r = build_expert_to_rank(60, 8, "linear", torch.device("cpu"))
    # Rank 0: experts 0-7
    assert e2r[0].item() == 0
    assert e2r[7].item() == 0
    # Rank 1: experts 8-15
    assert e2r[8].item() == 1
    assert e2r[15].item() == 1
    # Rank 3: experts 24-31
    assert e2r[24].item() == 3
    assert e2r[31].item() == 3
    # Rank 4: experts 32-38
    assert e2r[32].item() == 4
    assert e2r[38].item() == 4
    # Rank 7: experts 53-59
    assert e2r[53].item() == 7
    assert e2r[59].item() == 7
    # Every expert is assigned to a valid rank
    assert e2r.min().item() == 0
    assert e2r.max().item() == 7
    assert e2r.shape == (60,)


def test_build_expert_to_rank_round_robin():
    """Round-robin: expert e → rank e % ep_size."""
    e2r = build_expert_to_rank(60, 8, "round_robin", torch.device("cpu"))
    expected = torch.arange(60) % 8
    assert torch.equal(e2r, expected)


# ---------------------------------------------------------------------------
# adjust_logits_ema — balanced load: logits returned unchanged
# ---------------------------------------------------------------------------

def test_balanced_ema_no_penalty():
    """When all ranks carry equal load the penalty is zero and logits are
    returned unchanged."""
    T, E, ep = 4, 8, 2
    device = torch.device("cpu")
    expert_to_rank = build_expert_to_rank(E, ep, "linear", device)
    # Equal EMA load on every rank → excess = 0 → penalty = 0
    ema_load = torch.full((ep,), 10.0)
    logits = torch.randn(T, E)

    out = adjust_logits_ema(
        router_logits=logits,
        ema_load=ema_load,
        prev_topk_ids=None,
        expert_to_rank=expert_to_rank,
        num_experts=E,
    )
    assert torch.allclose(out, logits), "balanced EMA must not change logits"


# ---------------------------------------------------------------------------
# adjust_logits_ema — logits wider than num_experts (shared / redundant cols)
# ---------------------------------------------------------------------------

def test_wider_router_logits_extra_columns_unchanged():
    """When router_logits has extra columns (e.g. AITER shared-expert gate),
    the penalty must be zero-padded so those columns are not modified."""
    T, E, extra, ep = 4, 8, 2, 2
    device = torch.device("cpu")
    expert_to_rank = build_expert_to_rank(E, ep, "linear", device)

    # Make rank 0 hot so there IS a non-zero penalty on experts 0-3
    ema_load = torch.tensor([20.0, 5.0])
    logits = torch.randn(T, E + extra)

    out = adjust_logits_ema(
        router_logits=logits,
        ema_load=ema_load,
        prev_topk_ids=None,
        expert_to_rank=expert_to_rank,
        num_experts=E,
    )

    # Extra columns must be exactly unchanged
    assert torch.allclose(out[:, E:], logits[:, E:]), (
        "extra columns beyond num_experts must not be penalised"
    )
    # Routed expert columns on the hot rank must be reduced
    assert (out[:, :E // 2] < logits[:, :E // 2]).all(), (
        "hot-rank expert logits must be reduced"
    )


# ---------------------------------------------------------------------------
# adjust_logits_ema — prev_topk_ids with -1 and shared-expert IDs
# ---------------------------------------------------------------------------

def test_invalid_topk_ids_not_counted():
    """-1 and IDs >= num_experts in prev_topk_ids must not be counted in
    rank_counts, so they do not bias the EMA."""
    T, E, ep = 4, 8, 2
    device = torch.device("cpu")
    expert_to_rank = build_expert_to_rank(E, ep, "linear", device)
    ema_load = torch.zeros(ep)
    logits = torch.randn(T, E)

    # Mix of valid IDs, -1 (padding), and shared-expert IDs (>= E)
    prev_topk_ids = torch.tensor([
        [0, 1, -1, E + 1],   # 2 valid routed, 1 invalid, 1 shared
        [2, 3, -1, E],       # 2 valid routed, 1 invalid, 1 shared
        [-1, -1, -1, -1],    # all invalid
        [4, 5,  E, -1],      # 2 valid routed, 1 shared, 1 invalid
    ], dtype=torch.int32)

    _ = adjust_logits_ema(
        router_logits=logits,
        ema_load=ema_load,
        prev_topk_ids=prev_topk_ids,
        expert_to_rank=expert_to_rank,
        num_experts=E,
    )

    # Only the 6 valid routed slots should be counted across both ranks
    total_counted = ema_load.sum().item()
    # ema_load = alpha * rank_counts; default alpha=0.1, 6 valid slots split
    # 4 on rank 0 (experts 0-3) and 2 on rank 1 (experts 4-5)
    assert pytest.approx(total_counted, abs=1e-5) == 0.1 * 6
    assert ema_load[0].item() > ema_load[1].item(), (
        "rank 0 should have more load than rank 1"
    )
