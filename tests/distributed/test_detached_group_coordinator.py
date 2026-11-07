# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Placeholder groups of an elastic EP rank that is parked before it joins."""

import pytest
import torch

from vllm.distributed.stateless_coordinator import DetachedGroupCoordinator


def _world(global_rank: int) -> DetachedGroupCoordinator:
    return DetachedGroupCoordinator(
        [[0, 1, 2, 3]], local_rank=1, group_name="world", global_rank=global_rank
    )


def test_detached_group_reports_rank_without_peers():
    group = _world(2)
    assert (group.rank, group.rank_in_group, group.world_size) == (2, 2, 4)
    assert (group.cpu_group.rank(), group.cpu_group.size()) == (2, 4)


def test_detached_group_collectives_fail_instead_of_waiting():
    """A collective on a parked rank must fail, not wait for peers that are
    not there."""
    group = _world(2)
    with pytest.raises(RuntimeError, match="peers of a detached group"):
        group.barrier()
    with pytest.raises(RuntimeError, match="peers of a detached group"):
        group.broadcast(torch.zeros(1))


def test_detached_group_rebind_moves_rank_within_group():
    group = _world(1)
    group.rebind(3)
    assert (group.rank, group.rank_in_group, group.cpu_group.rank()) == (3, 3, 3)
    with pytest.raises(AssertionError, match="outside the group"):
        group.rebind(7)
