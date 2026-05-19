# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP mapping data structures for NIXL KV cache transfers."""

from __future__ import annotations

from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
)
from vllm.v1.kv_cache_interface import TPTransferSlice

# ======================================================================
# Data structures
# ======================================================================


@dataclass(frozen=True)
class ReadSpec:
    """Specification for a single remote block read operation."""

    remote_rank: int
    local_block_ids: BlockIds
    remote_block_ids: BlockIds


@dataclass(frozen=True)
class TPMapping:
    """Per-group TP transfer mapping for one remote engine.

    Generated once per remote engine during handshake. Each group carries
    its own list of TPTransferSlice describing the transfer plan.
    """

    # Per-group transfer slices. slices_per_group[g] = tuple of
    # TPTransferSlice for this local rank's reads from remotes for group g.
    slices_per_group: tuple[tuple[TPTransferSlice, ...], ...]

    # Derived: union of all remote ranks across all groups (sorted).
    all_source_ranks: tuple[int, ...] = field(init=False)

    # Derived: per-group set of source ranks for O(1) membership tests.
    _source_ranks_sets: tuple[frozenset[int], ...] = field(init=False)

    def __post_init__(self):
        all_ranks: set[int] = set()
        rank_sets: list[frozenset[int]] = []
        for group_slices in self.slices_per_group:
            group_ranks = frozenset(s.remote_rank for s in group_slices)
            rank_sets.append(group_ranks)
            all_ranks.update(group_ranks)
        object.__setattr__(self, "all_source_ranks", tuple(sorted(all_ranks)))
        object.__setattr__(self, "_source_ranks_sets", tuple(rank_sets))

    def source_ranks_for_group(self, group_idx: int) -> frozenset[int]:
        """Remote ranks involved in transfers for a given group."""
        return self._source_ranks_sets[group_idx]

    def has_rank_in_group(self, group_idx: int, rank: int) -> bool:
        """Check if a remote rank participates in a group's transfers."""
        return rank in self._source_ranks_sets[group_idx]

    def slice_for_rank(self, group_idx: int, rank: int) -> TPTransferSlice | None:
        """Find the transfer slice for a specific remote rank in a group."""
        for s in self.slices_per_group[group_idx]:
            if s.remote_rank == rank:
                return s
        return None

    def num_slices_for_group(self, group_idx: int) -> int:
        """Number of transfer slices (remote reads) for a group."""
        return len(self.slices_per_group[group_idx])
