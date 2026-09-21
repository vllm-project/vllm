# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.worker.cp_utils import should_skip_dcp_context_attention


def test_skip_gate_only_for_zero_context():
    assert should_skip_dcp_context_attention(torch.zeros(3, dtype=torch.int32))
    assert not should_skip_dcp_context_attention(
        torch.tensor([0, 5, 0], dtype=torch.int32)
    )


@pytest.mark.parametrize(
    "dcp_world_size,interleave_size,context_len",
    [(2, 16, 10), (4, 16, 10), (8, 16, 10), (4, 1, 2)],
)
def test_skip_gate_rank_invariant_with_divergent_local_context(
    dcp_world_size: int, interleave_size: int, context_len: int
):
    """Contexts shorter than a full interleave round land entirely on a
    subset of DCP ranks, so the per-rank local context lengths diverge:
    some ranks hold zero local context while others hold all of it. Ranks
    with zero local context must still take the collective (non-skip) path,
    otherwise the query all-gather in _forward_with_dcp deadlocks across
    ranks. The skip gate must therefore depend only on the rank-invariant
    global context lengths, never on get_dcp_local_seq_lens output.
    """
    context_kv_lens = torch.tensor([context_len], dtype=torch.int32)
    local_maxes = [
        int(
            get_dcp_local_seq_lens(
                context_kv_lens, dcp_world_size, rank, interleave_size
            ).max()
        )
        for rank in range(dcp_world_size)
    ]
    # Precondition: the local view diverges across ranks.
    assert 0 in local_maxes
    assert max(local_maxes) > 0
    # The batch still has context globally, so no rank may skip.
    assert not should_skip_dcp_context_attention(context_kv_lens)


class _StubDCPGroup:
    world_size = 8
    rank_in_group = 3


def test_block_table_per_group_dcp_world_size(monkeypatch):
    """Replicated groups (dcp_world_size=1) keep plain dcp=1 slot-mapping
    geometry on every rank; sharded groups (None) read the process DCP
    group. A group world that contradicts the process group must fail."""
    import vllm.v1.worker.block_table as bt

    monkeypatch.setattr(bt, "get_dcp_group", lambda: _StubDCPGroup())

    common = dict(
        block_size=16,
        max_num_reqs=2,
        max_num_blocks_per_req=4,
        max_num_batched_tokens=16,
        pin_memory=False,
        device=torch.device("cpu"),
        kernel_block_size=16,
        cp_kv_cache_interleave_size=1,
    )
    sharded = bt.BlockTable(**common)
    assert (sharded.dcp_world_size, sharded.dcp_rank) == (8, 3)

    replicated = bt.BlockTable(**common, dcp_world_size=1)
    assert (replicated.dcp_world_size, replicated.dcp_rank) == (1, 0)

    matching = bt.BlockTable(**common, dcp_world_size=8)
    assert (matching.dcp_world_size, matching.dcp_rank) == (8, 3)

    with pytest.raises(AssertionError, match="disagrees"):
        bt.BlockTable(**common, dcp_world_size=4)


def test_multi_group_block_table_threads_dcp_world_sizes(monkeypatch):
    import vllm.v1.worker.block_table as bt

    monkeypatch.setattr(bt, "get_dcp_group", lambda: _StubDCPGroup())

    multi = bt.MultiGroupBlockTable(
        max_num_reqs=2,
        max_num_batched_tokens=16,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[16, 16],
        kernel_block_sizes=[16, 16],
        max_num_blocks=[4, 4],
        cp_kv_cache_interleave_size=1,
        dcp_world_sizes=[None, 1],
    )
    sharded, replicated = multi.block_tables
    assert (sharded.dcp_world_size, sharded.dcp_rank) == (8, 3)
    assert (replicated.dcp_world_size, replicated.dcp_rank) == (1, 0)
