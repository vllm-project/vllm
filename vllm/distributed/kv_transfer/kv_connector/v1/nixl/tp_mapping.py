# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP mapping computation for NIXL KV cache transfers."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    TransferTopology,
    kv_head_transfer,
    mamba_tp_transfer_chunks,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec, MambaSpec

# ======================================================================
# Data structures
# ======================================================================


@dataclass(frozen=True)
class ReadSpec:
    """Specification for a single remote block read operation."""

    remote_rank: int
    local_block_ids: BlockIds
    remote_block_ids: BlockIds
    block_ids_by_region: bool = False


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

    # Maps each head-sharded attention source rank to its FA head slot index.
    rank_to_attention_slot: dict[int, int]

    # FA head offset factor for hetero-TP (D_TP > P_TP).
    rank_offset_factor: int

    # Local ranks (in aggregate) that read from a given source rank. The producer frees
    # a request's blocks only once that many notifications have come in.
    local_consumers: int = 1


# ======================================================================
# TP mapping computation
# ======================================================================


def compute_tp_mapping(
    transfer_topology: TransferTopology,
    remote_tp_size: int,
    group_spec_types: tuple[type[KVCacheSpec], ...],
    remote_dcp_size: int = 1,
    head_sharded_kv_heads: int | None = None,
    mamba_spec: MambaSpec | None = None,
) -> TPMapping:
    """Build the complete local-to-remote TP mapping.

    Computes source ranks, head slot assignments, and the rank offset
    factor in a single pass.

    DCP support is scoped to MLA only, with a side is either fully replicated or fully
    sharded. DCP-branch reuses the same rank set used at handshake selection.

    ``head_sharded_kv_heads`` maps head-sharded KV with that many total KV heads
    instead of the topology's, e.g. a GQA draft under an MLA target.
    ``mamba_spec`` is the spec of the sharded Mamba layers, required with SSM
    groups.
    """
    tp_rank = transfer_topology.tp_rank
    tp_size = transfer_topology.tp_size
    is_mla = transfer_topology.is_mla and head_sharded_kv_heads is None
    total_num_kv_heads = head_sharded_kv_heads or transfer_topology.total_num_kv_heads
    # --- Attention source ranks ---
    if is_mla and remote_dcp_size > 1:
        attn_ranks = transfer_topology.dcp_source_ranks(remote_tp_size, remote_dcp_size)
        reads = {}
    else:
        # MLA is one KV head replicated on every rank.
        num_heads = 1 if is_mla else total_num_kv_heads
        reads = {
            rank: heads
            for rank in range(remote_tp_size)
            if (
                heads := kv_head_transfer(
                    num_heads, tp_size, tp_rank, remote_tp_size, rank, receiving=True
                )
            )
        }
        attn_ranks = list(reads)

    # --- SSM source ranks ---
    ssm_ranks = []
    if any(_is_ssm_spec(t) for t in group_spec_types):
        assert mamba_spec is not None
        ssm_ranks = [
            rank
            for rank in range(remote_tp_size)
            if mamba_tp_transfer_chunks(
                mamba_spec, tp_size, tp_rank, remote_tp_size, rank, receiving=True
            )
        ]

    all_ranks = sorted(set(attn_ranks) | set(ssm_ranks))

    # --- Per-group ordered source ranks ---
    source_ranks_per_group = tuple(
        tuple(ssm_ranks) if _is_ssm_spec(t) else tuple(attn_ranks)
        for t in group_spec_types
    )

    # --- Attention head slots and offset within the remote block ---
    # Sources split this rank's heads evenly, in head order.
    rank_offset_factor = 0
    rank_to_attention_slot = {}
    for rank, (local_heads, remote_heads) in reads.items():
        num_read = local_heads.stop - local_heads.start
        rank_to_attention_slot[rank] = local_heads.start // num_read
        if len(reads) == 1:
            rank_offset_factor = remote_heads.start // num_read

    local_consumers = transfer_topology.dcp_consumer_count(
        remote_tp_size, remote_dcp_size
    )

    return TPMapping(
        source_ranks_per_group=source_ranks_per_group,
        all_source_ranks=tuple(all_ranks),
        rank_to_attention_slot=rank_to_attention_slot,
        rank_offset_factor=rank_offset_factor,
        local_consumers=local_consumers,
    )
