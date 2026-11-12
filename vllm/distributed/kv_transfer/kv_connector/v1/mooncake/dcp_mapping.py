# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pure DCP topology validation, rank mapping, and block ownership helpers."""


def _validate_dcp_group_block_sizes(
    block_sizes: tuple[int, ...],
    dcp_sharded: tuple[bool, ...],
    dcp_size: int,
    interleave_size: int,
) -> None:
    """Validate the local interleave for each DCP-sharded transfer group."""
    if len(block_sizes) != len(dcp_sharded):
        raise ValueError("Mooncake DCP group metadata lengths must match.")
    for group_index, (block_size, sharded) in enumerate(zip(block_sizes, dcp_sharded)):
        if sharded and dcp_size > 1 and block_size != interleave_size:
            raise ValueError(
                f"Mooncake DCP transfer group {group_index} requires "
                "spec.block_size == cp_kv_cache_interleave_size: "
                f"got {block_size} and {interleave_size}."
            )


def _get_prefill_ranks_for_decode(
    d_tp_rank: int,
    d_tp_size: int,
    d_dcp_size: int,
    p_tp_size: int,
    p_dcp_size: int,
) -> list[int]:
    """Return MLA producers owning this decoder's sequence shards, plus ACKs."""
    for tp_size, dcp_size, role in (
        (p_tp_size, p_dcp_size, "Producer"),
        (d_tp_size, d_dcp_size, "Consumer"),
    ):
        if tp_size < 1 or dcp_size not in (1, tp_size):
            raise ValueError(
                f"{role} MLA DCP size must be 1 or TP size: "
                f"dcp_size={dcp_size}, tp_size={tp_size}."
            )
    if not 0 <= d_tp_rank < d_tp_size:
        raise ValueError(f"Invalid consumer TP rank {d_tp_rank} for TP {d_tp_size}.")
    if max(p_dcp_size, d_dcp_size) % min(p_dcp_size, d_dcp_size):
        raise ValueError(
            "Mooncake DCP transfer requires mutually divisible P/D DCP sizes: "
            f"P={p_dcp_size}, D={d_dcp_size}."
        )
    if p_dcp_size == 1:
        # All replicas must acknowledge completion; only rank 0 sends payload.
        return list(range(p_tp_size))
    d_dcp_rank = d_tp_rank % d_dcp_size
    if p_dcp_size >= d_dcp_size:
        return list(range(d_dcp_rank, p_dcp_size, d_dcp_size))
    return [d_dcp_rank % p_dcp_size]


def _get_prefill_rank_groups(
    *,
    d_tp_rank: int,
    d_tp_size: int,
    d_dcp_size: int,
    p_tp_size: int,
    p_dcp_size: int,
    group_dcp_sharded: tuple[bool, ...],
    non_dcp_ranks: list[int],
) -> dict[int, tuple[int, ...]]:
    """Map participating P ranks to local transfer groups, including ACK peers.

    Non-sharded groups retain the connector's existing TP routing. Group
    membership describes participation, not permission to send every replica.
    """
    if not group_dcp_sharded:
        return {rank: () for rank in sorted(set(non_dcp_ranks))}
    rank_groups: dict[int, list[int]] = {}
    sharded_ranks: list[int] | None = None
    for group_index, sharded in enumerate(group_dcp_sharded):
        if sharded and (p_dcp_size > 1 or d_dcp_size > 1):
            if sharded_ranks is None:
                sharded_ranks = _get_prefill_ranks_for_decode(
                    d_tp_rank=d_tp_rank,
                    d_tp_size=d_tp_size,
                    d_dcp_size=d_dcp_size,
                    p_tp_size=p_tp_size,
                    p_dcp_size=p_dcp_size,
                )
            ranks = sharded_ranks
        else:
            ranks = non_dcp_ranks
        for rank in sorted(set(ranks)):
            rank_groups.setdefault(rank, []).append(group_index)
    return {rank: tuple(rank_groups[rank]) for rank in sorted(rank_groups)}


def _get_decode_ranks_for_prefill(
    p_tp_rank: int,
    p_tp_size: int,
    p_dcp_size: int,
    d_tp_size: int,
    d_dcp_size: int,
    *,
    group_dcp_sharded: tuple[bool, ...] | None = None,
    non_dcp_ranks: list[int] | None = None,
) -> list[int]:
    """Return D peers, optionally combining DCP and legacy per-group routes."""
    if group_dcp_sharded is not None:
        assert non_dcp_ranks is not None
        if (p_dcp_size == d_dcp_size == 1) or not any(group_dcp_sharded):
            return non_dcp_ranks
    d_tp_ranks: list[int] = []
    for d_tp_rank in range(d_tp_size):
        p_tp_ranks = _get_prefill_ranks_for_decode(
            d_tp_rank=d_tp_rank,
            d_tp_size=d_tp_size,
            d_dcp_size=d_dcp_size,
            p_tp_size=p_tp_size,
            p_dcp_size=p_dcp_size,
        )
        if p_tp_rank in p_tp_ranks:
            d_tp_ranks.append(d_tp_rank)
    if group_dcp_sharded is not None and not all(group_dcp_sharded):
        assert non_dcp_ranks is not None
        return sorted(set(d_tp_ranks) | set(non_dcp_ranks))
    return d_tp_ranks


def _filter_dcp_block_ids(
    p_block_ids: list[int],
    d_block_ids: list[int],
    p_dcp_rank: int,
    p_dcp_size: int,
    d_dcp_rank: int,
    d_dcp_size: int,
    num_computed_tokens: int,
    block_size: int,
) -> tuple[list[int], list[int]]:
    if num_computed_tokens < 0 or num_computed_tokens % block_size:
        raise ValueError("Mooncake DCP prefix must end on a physical block boundary.")
    d_start_sequence_block = num_computed_tokens // block_size
    d_first_sequence_block = d_start_sequence_block + (
        (d_dcp_rank - d_start_sequence_block) % d_dcp_size
    )
    d_sequence_block_to_cache_block = {
        d_first_sequence_block + i * d_dcp_size: block_id
        for i, block_id in enumerate(d_block_ids)
    }

    filtered_p_block_ids: list[int] = []
    filtered_d_block_ids: list[int] = []
    for i, p_block_id in enumerate(p_block_ids):
        p_sequence_block = i * p_dcp_size + p_dcp_rank
        d_block_id = d_sequence_block_to_cache_block.get(p_sequence_block)
        if d_block_id is not None:
            filtered_p_block_ids.append(p_block_id)
            filtered_d_block_ids.append(d_block_id)

    return filtered_p_block_ids, filtered_d_block_ids
