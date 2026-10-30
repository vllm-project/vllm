# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe.prepare_finalize.alltoall_batched import (
    AllToAllBatchedPrepareAndFinalize,
)


def make_pf(max_num_tokens: int, num_local_experts: int, num_dispatchers: int):
    return AllToAllBatchedPrepareAndFinalize(
        max_num_tokens=max_num_tokens,
        num_local_experts=num_local_experts,
        num_dispatchers=num_dispatchers,
        rank=0,
    )


def worst_case_fill(topk_ids: torch.Tensor, num_local_experts: int, world: int):
    """Replays prepare()'s packing, returning peak (token, slot, expert) usage."""
    num_tokens, _ = topk_ids.shape
    flat_expert = topk_ids.reshape(-1).to(torch.int64)
    flat_token = torch.arange(num_tokens).repeat_interleave(topk_ids.size(1))
    dest = torch.div(flat_expert, num_local_experts, rounding_mode="floor")

    tok_fill, slot_fill = 0, 0
    for r in range(world):
        m = dest == r
        toks, counts = torch.unique(flat_token[m], return_counts=True)
        tok_fill = max(tok_fill, int(toks.numel()))
        if counts.numel():
            slot_fill = max(slot_fill, int(counts.max()))

    local = flat_expert[dest == 0]
    rows = torch.zeros(num_local_experts, dtype=torch.int64)
    rows.index_add_(0, local, torch.ones(local.numel(), dtype=torch.int64))
    # Every source rank may send an identical batch.
    return tok_fill, slot_fill, int(rows.max()) * world


@pytest.mark.parametrize(
    "world,num_experts,topk",
    [(2, 40, 8), (2, 40, 1), (4, 64, 6), (8, 256, 8), (8, 16, 8)],
)
def test_capacity_covers_random_routing(world, num_experts, topk):
    max_num_tokens, e_local = 256, num_experts // world
    pf = make_pf(max_num_tokens, e_local, world)
    tok_cap, slots, expert_cap = pf.capacities(topk)

    torch.manual_seed(0)
    topk_ids = torch.stack(
        [torch.randperm(num_experts)[:topk] for _ in range(max_num_tokens)]
    )
    tok_fill, slot_fill, expert_fill = worst_case_fill(topk_ids, e_local, world)

    assert tok_fill <= tok_cap
    assert slot_fill <= slots
    assert expert_fill <= expert_cap


def test_token_capacity_is_tight_when_all_tokens_target_one_rank():
    """Dedup keeps the worst case token-sized rather than pair-sized."""
    max_num_tokens, world, e_local, topk = 256, 2, 20, 8
    pf = make_pf(max_num_tokens, e_local, world)
    tok_cap, slots, _ = pf.capacities(topk)

    topk_ids = torch.arange(topk).repeat(max_num_tokens, 1)
    tok_fill, slot_fill, _ = worst_case_fill(topk_ids, e_local, world)

    assert tok_fill == tok_cap
    assert slot_fill == slots


def test_expert_capacity_is_tight_when_all_tokens_pick_one_expert():
    """One hot expert receives one row per token from every source rank."""
    max_num_tokens, world, e_local = 256, 2, 20
    pf = make_pf(max_num_tokens, e_local, world)
    _, _, expert_cap = pf.capacities(topk=1)

    topk_ids = torch.zeros((max_num_tokens, 1), dtype=torch.int64)
    _, _, expert_fill = worst_case_fill(topk_ids, e_local, world)

    assert expert_fill == expert_cap


def test_activation_buffer_does_not_scale_with_topk():
    """Only the int/float metadata may scale with topk, not the payload."""
    pf = make_pf(max_num_tokens=256, num_local_experts=20, num_dispatchers=2)
    tok_cap, slots, _ = pf.capacities(topk=8)
    assert tok_cap == 256
    assert slots == 8


def test_slots_clamped_by_local_expert_count():
    """A token cannot need more slots on a rank than that rank has experts."""
    pf = make_pf(max_num_tokens=256, num_local_experts=2, num_dispatchers=8)
    _, slots, _ = pf.capacities(topk=8)
    assert slots == 2


def test_expert_capacity_matches_batched_experts_workspace():
    """b_a1's token dim must match BatchedTritonExperts.workspace_shapes()."""
    pf = make_pf(max_num_tokens=256, num_local_experts=20, num_dispatchers=4)
    _, _, expert_cap = pf.capacities(topk=8)
    assert expert_cap == pf.max_num_tokens_per_rank() * pf.num_dispatchers()
