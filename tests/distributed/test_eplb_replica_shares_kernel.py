# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The EPLB mapping kernel's replica-share path.

A load balancer decides how much of a logical expert's traffic each physical
replica should take. Applying that split stays in this kernel, which also
records expert load -- so the split must be honoured *without* disturbing the
mapping's correctness or the recording.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.router.base_router import (
    _eplb_map_and_record_triton,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="the mapping kernel is CUDA-only"
)

NUM_LOGICAL = 8
NUM_PHYSICAL = 12  # experts 0..3 get a second replica
NUM_TOKENS = 512
TOPK = 4


def _placement(device):
    """physical_to_logical: 0..7 then a second replica of 0..3."""
    phy2log = torch.tensor(
        list(range(NUM_LOGICAL)) + list(range(NUM_PHYSICAL - NUM_LOGICAL)),
        dtype=torch.int64,
        device=device,
    )
    # logical -> [primary, secondary or -1]
    log2phy = torch.full((NUM_LOGICAL, 2), -1, dtype=torch.int32, device=device)
    counts = torch.ones(NUM_LOGICAL, dtype=torch.int32, device=device)
    for logical in range(NUM_LOGICAL):
        log2phy[logical, 0] = logical
    for logical in range(NUM_PHYSICAL - NUM_LOGICAL):
        log2phy[logical, 1] = NUM_LOGICAL + logical
        counts[logical] = 2
    return phy2log, log2phy, counts


def _run(log2phy, counts, topk_ids, prob, device):
    load = torch.zeros(NUM_PHYSICAL, dtype=torch.int64, device=device)
    out = _eplb_map_and_record_triton(
        topk_ids=topk_ids,
        logical_to_physical_map=log2phy,
        logical_replica_count=counts,
        expert_load_view=load,
        record_enabled=torch.tensor(1, dtype=torch.int32, device=device),
        num_unpadded_tokens=None,
        replica_prob=prob,
    )
    return out, load


@pytest.mark.parametrize("target", [0, 1])
def test_a_degenerate_share_sends_every_token_to_that_replica(target):
    """The sharpest check that the table is actually consulted: put all of an
    expert's share on one replica and every one of its tokens must land there."""
    device = torch.device("cuda")
    phy2log, log2phy, counts = _placement(device)
    # Only experts with two replicas can demonstrate a choice.
    topk_ids = torch.randint(
        0,
        NUM_PHYSICAL - NUM_LOGICAL,
        (NUM_TOKENS, TOPK),
        dtype=torch.int32,
        device=device,
    )
    prob = torch.zeros(NUM_LOGICAL, 2, dtype=torch.float32, device=device)
    prob[:, target] = 1.0

    out, _ = _run(log2phy, counts, topk_ids, prob, device)

    expected = log2phy[topk_ids.long(), target]
    assert torch.equal(out.to(torch.int32), expected), (
        f"tokens did not follow a share of 1.0 on replica {target}"
    )


def test_split_shares_are_respected_in_proportion():
    device = torch.device("cuda")
    phy2log, log2phy, counts = _placement(device)
    # One expert, so the sample is the whole batch.
    topk_ids = torch.zeros((NUM_TOKENS, TOPK), dtype=torch.int32, device=device)
    prob = torch.zeros(NUM_LOGICAL, 2, dtype=torch.float32, device=device)
    prob[0, 0] = 0.25
    prob[0, 1] = 0.75

    out, _ = _run(log2phy, counts, topk_ids, prob, device)

    secondary = int((out == NUM_LOGICAL).sum())
    fraction = secondary / out.numel()
    # The draw is a hash of the token index, so this is deterministic, not
    # statistical; allow slack only for the finite sample.
    assert 0.65 < fraction < 0.85, (
        f"expected ~0.75 on the second replica, got {fraction:.3f}"
    )


def test_shares_never_break_the_logical_choice():
    """The accuracy-preserving invariant: whatever replica is chosen, mapping
    it back must return the logical expert the router asked for."""
    device = torch.device("cuda")
    phy2log, log2phy, counts = _placement(device)
    torch.manual_seed(0)
    topk_ids = torch.randint(
        0, NUM_LOGICAL, (NUM_TOKENS, TOPK), dtype=torch.int32, device=device
    )
    prob = torch.rand(NUM_LOGICAL, 2, dtype=torch.float32, device=device)
    prob[counts == 1, 1] = 0.0  # padded column must stay unreachable

    out, _ = _run(log2phy, counts, topk_ids, prob, device)

    assert torch.equal(phy2log[out.long()].to(torch.int32), topk_ids)


def test_an_all_zero_row_falls_back_to_the_builtin_choice():
    """'No preference' must not collapse every token onto replica 0."""
    device = torch.device("cuda")
    phy2log, log2phy, counts = _placement(device)
    topk_ids = torch.zeros((NUM_TOKENS, TOPK), dtype=torch.int32, device=device)
    prob = torch.zeros(NUM_LOGICAL, 2, dtype=torch.float32, device=device)

    with_table, _ = _run(log2phy, counts, topk_ids, prob, device)
    without_table, _ = _run(log2phy, counts, topk_ids, None, device)

    assert torch.equal(with_table, without_table)


def test_recording_still_happens_on_the_share_path():
    """Recording lives in this kernel; supplying shares must not disturb it."""
    device = torch.device("cuda")
    phy2log, log2phy, counts = _placement(device)
    topk_ids = torch.randint(
        0, NUM_LOGICAL, (NUM_TOKENS, TOPK), dtype=torch.int32, device=device
    )
    prob = torch.rand(NUM_LOGICAL, 2, dtype=torch.float32, device=device)
    prob[counts == 1, 1] = 0.0

    out, load = _run(log2phy, counts, topk_ids, prob, device)

    assert int(load.sum()) == topk_ids.numel()
    assert torch.equal(
        load, torch.bincount(out.reshape(-1).long(), minlength=NUM_PHYSICAL)
    )


def test_matching_the_collective_does_not_publish_zero_counts():
    """A dummy forward must not wipe the counts a real one just published.

    `finalize_step_counts` copies the per-layer local counts into the global
    buffer the LP reads, then clears the local one. Running it a second time
    within a step -- which is what a DP rank on a dummy batch does beside a peer
    on a real batch -- copies the freshly cleared buffer over the published
    counts. The solve then sees an all-zero load, produces an exactly uniform
    split, and LPLB silently becomes `dynamic` while still paying for the solve.

    That is not hypothetical: it is what shipped, and every LPLB measurement
    taken before this fix was of a policy solving against zeros.

    So the dummy path reduces a scratch tensor instead. This checks the counts
    survive it.
    """
    import torch

    from vllm.distributed.eplb.mlb_runtime import MlbRoutingRuntime

    rt = MlbRoutingRuntime.__new__(MlbRoutingRuntime)
    rt._logical_count_local = torch.zeros(2, 8)
    rt._logical_count_global = torch.zeros(2, 8)
    rt._logical_count_ready = False
    rt._count_scratch = None

    published = torch.tensor(
        [[3.0, 1.0, 0, 0, 0, 0, 0, 0], [0, 2.0, 5.0, 0, 0, 0, 0, 0]]
    )
    rt._logical_count_local.copy_(published)

    class _Group:
        def all_reduce(self, t):  # single rank: reduction is identity
            return t

    import vllm.distributed as dist_mod

    orig = dist_mod.get_ep_group
    dist_mod.get_ep_group = lambda: _Group()
    try:
        rt.finalize_step_counts()
        assert torch.equal(rt._logical_count_global, published)
        # The dummy path runs next, in the same step.
        rt.match_step_counts_collective()
        assert torch.equal(rt._logical_count_global, published), (
            "the dummy path overwrote the published counts; the solve would "
            "see zeros and split uniformly"
        )
        assert rt._count_scratch is not None, "no collective was issued"
    finally:
        dist_mod.get_ep_group = orig
