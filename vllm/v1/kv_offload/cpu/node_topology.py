# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Placement of an engine's offloading workers across nodes.

The tiering scheduler and its secondary tiers run in the engine core, which
shares a node with group rank 0 (the head node), and address only that node's
shared CPU region. Worker creation uses this placement to tell whether an
engine's workers span several nodes.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator


@dataclass(frozen=True)
class MirrorTopology:
    """Placement of this worker relative to the head node (group rank 0)."""

    group: "GroupCoordinator"
    rank: int
    on_head: bool
    # Group ranks whose workers are not on the head node.
    remote_ranks: frozenset[int]


def _optional_group(get_group: Callable[[], Any]) -> Any:
    try:
        return get_group()
    except AssertionError:
        return None


def multi_node_topology(
    world_size: int,
) -> tuple[MirrorTopology | None, bool | None]:
    """Locate this worker relative to its engine's head node.

    Collective over every worker of every data parallel replica.

    Args:
        world_size: Number of workers in this engine (one data parallel
            replica).

    Returns:
        This worker's placement, or None when the engine's workers share a
        node or their group cannot be identified; and whether the engine's
        workers span several nodes, or None when their group cannot be
        identified.

    """
    from torch.distributed import ProcessGroup

    from vllm.distributed.parallel_state import (
        get_inner_dp_world_group,
        get_world_group,
        in_the_same_node_as,
    )

    # The inner group exists only when a data parallel replica spans nodes
    # under the multiprocessing backend.
    group = _optional_group(get_inner_dp_world_group) or _optional_group(
        get_world_group
    )
    if group is not None and group.world_size == world_size == 1:
        return None, False
    if (
        group is None
        or not isinstance(group.cpu_group, ProcessGroup)
        or group.world_size % world_size
    ):
        return None, None
    placement = replica_placement(
        group.rank_in_group,
        world_size,
        group.world_size,
        lambda source_rank: in_the_same_node_as(
            group.cpu_group, source_rank=source_rank
        ),
    )
    if placement is None:
        return None, False
    if group.world_size > world_size:
        # The world group holds every replica; only the inner group of a
        # replica can serve as its topology.
        return None, True
    on_head, remote_ranks = placement
    return MirrorTopology(group, group.rank_in_group, on_head, remote_ranks), True


def replica_placement(
    rank: int,
    world_size: int,
    group_world_size: int,
    same_node_as: Callable[[int], list[bool]],
) -> tuple[bool, frozenset[int]] | None:
    """Place a worker within its replica, given a group of whole replicas.

    The group holds rank-ordered replicas of ``world_size`` workers each.

    Args:
        rank: The worker's rank in the group.
        world_size: Number of workers in one replica.
        group_world_size: Number of workers in the group.
        same_node_as: Collective returning, for a source rank, whether each
            group rank shares its node. Every rank calls it with the first
            rank of each replica, in order.

    Returns:
        None when the worker's replica shares one node; otherwise whether the
        worker shares its replica's first node, and the replica-local ranks
        that do not.

    """
    replica = rank // world_size
    placement = None
    for index in range(group_world_size // world_size):
        first = index * world_size
        same_node = same_node_as(first)[first : first + world_size]
        if index == replica and not all(same_node):
            placement = (
                same_node[rank - first],
                frozenset(r for r, local in enumerate(same_node) if not local),
            )
    return placement
