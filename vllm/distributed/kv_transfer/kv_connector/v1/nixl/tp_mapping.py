# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP mapping computation for NIXL KV cache transfers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    TransferTopology,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec, MambaSpec

# ======================================================================
# Data structures
# ======================================================================


class PageView(NamedTuple):
    """``heads`` KV heads from ``first`` of every page of a region holding
    ``tokens`` tokens per head, in ``units`` equal token intervals, each one
    descriptor or, with ``runs`` == ``heads``, one per head."""

    region: int
    first: int
    heads: int
    tokens: int
    units: int
    runs: int
    token_bytes: int


class GroupPairs(NamedTuple):
    """Layers split by their (local, remote) KV cache group and transfer
    geometry, for peers whose layers the model-wide mapping cannot pair:
    different groups, page shapes or KV head ownership. Each pair runs with its
    groups' block ids on each side, from its own remote ranks.
    """

    # Indices into each side's KV cache transfer groups.
    local_groups: list[int]
    remote_groups: list[int]
    spec_types: list[type[KVCacheSpec]]
    # Remote ranks each pair transfers with.
    source_ranks: list[tuple[int, ...]]
    # Local and remote transfer units per page, and descriptors per unit.
    units: list[tuple[int, int, int]]
    # Attention: each layer's first descriptor in its remote rank's lists.
    # SSM: state sub-regions. Aligned by layer.
    local_regions: list[np.ndarray]
    remote_regions: list[np.ndarray]
    # Per remote rank: attention descriptor views of the local and remote list.
    local_views: dict[int, tuple[PageView, ...]]
    remote_views: dict[int, tuple[PageView, ...]]
    # Remote regions holding SSM state, in remote descriptor order.
    remote_ssm_regions: list[int]


@dataclass(frozen=True)
class ReadSpec:
    """Specification for a single remote block read operation."""

    remote_rank: int
    local_block_ids: BlockIds
    remote_block_ids: BlockIds
    block_ids_by_region: bool = False
    # Set when the block ids are per group pair.
    group_pairs: GroupPairs | None = None


def kv_head_slices(
    heads: int,
    tp: int,
    rank: int,
    remote_heads: int,
    remote_tp: int,
    remote_ranks: list[int],
    read: bool,
) -> dict[int, tuple[int, int, int]]:
    """(first local head, first remote head, count) each remote rank shares
    with this rank, from both sides' per-rank KV heads.

    A rank owns a contiguous range of global KV heads; with more ranks than
    heads, QKVParallelLinear replicates each head on consecutive ranks. A read
    takes each local head from the first remote rank owning it; a write sends
    every local head a remote rank owns.
    """
    # Total KV heads; if both sides hold one head, any total maps alike.
    total = next(
        (h * t for h, t in ((heads, tp), (remote_heads, remote_tp)) if h > 1),
        min(tp, remote_tp),
    )

    def owned(n: int, t: int, r: int) -> range:
        first = r * n if n > 1 else r * total // t
        return range(first, first + n)

    local = owned(heads, tp, rank)
    start = local.start
    slices = {}
    for remote_rank in remote_ranks:
        remote = owned(remote_heads, remote_tp, remote_rank)
        first, last = max(start, remote.start), min(local.stop, remote.stop)
        if first < last:
            slices[remote_rank] = (
                first - local.start,
                first - remote.start,
                last - first,
            )
            if read:
                start = last
    return slices


def _is_attention_spec(spec_type: type[KVCacheSpec]) -> bool:
    return issubclass(spec_type, AttentionSpec)


def _is_ssm_spec(spec_type: type[KVCacheSpec]) -> bool:
    return issubclass(spec_type, MambaSpec)


@dataclass(frozen=True)
class TPMapping:
    """Complete local-to-remote TP mapping for one remote engine.

    Generated once per remote engine during handshake.
    """

    # Remote TP ranks that this local rank reads from, per group.
    # Position = local piece index.
    source_ranks_per_group: tuple[tuple[int, ...], ...]

    # Superset of all source ranks (union of all groups).
    all_source_ranks: tuple[int, ...]

    # Maps each source rank to its FA head slot index.
    rank_to_attention_slot: dict[int, int]

    # FA head offset factor for hetero-TP (D_TP > P_TP).
    rank_offset_factor: int

    # Local ranks (in aggregate) that read from a given source rank. The producer frees
    # a request's blocks only once that many notifications have come in.
    local_consumers: int = 1

    group_pairs: GroupPairs | None = None


# ======================================================================
# TP mapping computation
# ======================================================================


def compute_tp_mapping(
    transfer_topology: TransferTopology,
    remote_tp_size: int,
    group_spec_types: tuple[type[KVCacheSpec], ...],
    remote_dcp_size: int = 1,
    group_pairs: GroupPairs | None = None,
) -> TPMapping:
    """Build the complete local-to-remote TP mapping.

    Computes source ranks, head slot assignments, and the rank offset
    factor in a single pass.

    DCP support is scoped to MLA only, with a side is either fully replicated or fully
    sharded. DCP-branch reuses the same rank set used at handshake selection.
    """
    tp_rank = transfer_topology.tp_rank
    tp_size = transfer_topology.tp_size
    total_num_kv_heads = transfer_topology.total_num_kv_heads
    # --- Attention source ranks ---
    if transfer_topology.is_mla or tp_size >= remote_tp_size:
        if transfer_topology.is_mla and remote_dcp_size > 1:
            attn_ranks = transfer_topology.dcp_source_ranks(
                remote_tp_size, remote_dcp_size
            )
        else:
            # D (local TP) > P (remote TP): multiple local ranks read different chunks
            # from *one* remote rank, corresponding to different kv heads.
            # For MLA, we only need one remote since cache is duplicated. When
            # P TP=k*TP k, this will spread mla ranks to read from remote k*tp_rank.
            attn_ranks = [tp_rank * remote_tp_size // tp_size]
    else:
        # P (remote TP) > D (local TP): one local rank
        # reads from multiple remote ranks.
        # GQA dedup: when K < remote_tp_size, several remote ranks
        # hold the same KV head.  np.unique keeps only the first
        # rank per unique head so we don't issue redundant reads.
        abs_tp = remote_tp_size // tp_size
        start = tp_rank * abs_tp
        heads = np.arange(start, start + abs_tp) * total_num_kv_heads // remote_tp_size
        _, unique_idx = np.unique(heads, return_index=True)
        attn_ranks = (start + np.sort(unique_idx)).tolist()

    # --- SSM source ranks ---
    has_ssm = any(_is_ssm_spec(t) for t in group_spec_types)
    if has_ssm:
        if tp_size < remote_tp_size:
            abs_tp = remote_tp_size // tp_size
            ssm_ranks = list(range(tp_rank * abs_tp, (tp_rank + 1) * abs_tp))
        else:
            ssm_ranks = list(attn_ranks)
    else:
        ssm_ranks = []

    all_ranks = sorted(set(attn_ranks) | set(ssm_ranks))

    # --- Per-group ordered source ranks ---
    source_ranks_per_group = tuple(
        tuple(ssm_ranks) if _is_ssm_spec(t) else tuple(attn_ranks)
        for t in group_spec_types
    )

    # --- Attention head slots ---
    head_to_slot: dict[int, int] = {}
    for i, r in enumerate(attn_ranks):
        head_to_slot[r * total_num_kv_heads // remote_tp_size] = i
    rank_to_attention_slot = {
        r: head_to_slot.get(r * total_num_kv_heads // remote_tp_size, 0)
        for r in all_ranks
    }

    # --- Rank offset factor ---
    if transfer_topology.is_mla or tp_size <= remote_tp_size:
        # We don't index into remote for reading, no offset needed.
        rank_offset_factor = 0
    elif tp_size > total_num_kv_heads:
        local_head = tp_rank * total_num_kv_heads // tp_size
        p_start = attn_ranks[0] * total_num_kv_heads // remote_tp_size
        rank_offset_factor = local_head - p_start
    else:
        # D TP > P TP: we index into remote to read different heads depending on rank.
        rank_offset_factor = tp_rank % (tp_size // remote_tp_size)

    local_consumers = transfer_topology.dcp_consumer_count(
        remote_tp_size, remote_dcp_size
    )

    return TPMapping(
        source_ranks_per_group=source_ranks_per_group,
        all_source_ranks=tuple(all_ranks),
        rank_to_attention_slot=rank_to_attention_slot,
        rank_offset_factor=rank_offset_factor,
        local_consumers=local_consumers,
        group_pairs=group_pairs,
    )
