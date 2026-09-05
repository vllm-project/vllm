# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Schedule EPLB expert migrations into conflict-free batches."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class MigrationFlow:
    """Expert transfers between one directed rank pair."""

    src_rank: int
    dst_rank: int
    expert_ids: tuple[int, ...]


def schedule_migration_batches(
    num_local_experts: int,
    old_indices: np.ndarray,
    new_indices: np.ndarray,
) -> list[list[MigrationFlow]]:
    """Build and greedily group rank-pair-disjoint migration flows."""
    assert old_indices.shape == new_indices.shape
    recv_ranks_by_expert: dict[int, list[int]] = {}
    old_experts_by_rank: list[set[int]] = []
    old_by_rank = old_indices.reshape(-1, num_local_experts)
    new_by_rank = new_indices.reshape(-1, num_local_experts)
    for rank, (old_local, new_local) in enumerate(zip(old_by_rank, new_by_rank)):
        old_experts = set(old_local.tolist())
        old_experts.discard(-1)
        old_experts_by_rank.append(old_experts)

        new_experts = set(new_local.tolist())
        new_experts.discard(-1)
        for expert_id in new_experts - old_experts:
            recv_ranks_by_expert.setdefault(expert_id, []).append(rank)

    old_ranks_by_expert: dict[int, list[int]] = {}
    needed_experts = recv_ranks_by_expert.keys()
    for rank, old_experts in enumerate(old_experts_by_rank):
        for expert_id in old_experts & needed_experts:
            old_ranks_by_expert.setdefault(expert_id, []).append(rank)

    expert_ids_by_pair: dict[tuple[int, int], list[int]] = {}
    for expert_id in sorted(recv_ranks_by_expert):
        send_ranks = old_ranks_by_expert.get(expert_id, [])
        if not send_ranks:
            continue
        recv_ranks = recv_ranks_by_expert[expert_id]

        num_per_sender, remainder = divmod(len(recv_ranks), len(send_ranks))
        remainder_start = len(send_ranks) * num_per_sender
        for sender_idx, src_rank in enumerate(send_ranks):
            start = sender_idx * num_per_sender
            for dst_rank in recv_ranks[start : start + num_per_sender]:
                expert_ids_by_pair.setdefault((src_rank, dst_rank), []).append(
                    expert_id
                )
            if sender_idx < remainder:
                dst_rank = recv_ranks[remainder_start + sender_idx]
                expert_ids_by_pair.setdefault((src_rank, dst_rank), []).append(
                    expert_id
                )

    batches: list[list[MigrationFlow]] = []
    endpoints_used: list[int] = []
    for (src_rank, dst_rank), expert_ids in expert_ids_by_pair.items():
        flow = MigrationFlow(src_rank, dst_rank, tuple(expert_ids))
        endpoints = (1 << src_rank) | (1 << dst_rank)
        for batch_idx, (batch, used) in enumerate(zip(batches, endpoints_used)):
            if not endpoints & used:
                batch.append(flow)
                endpoints_used[batch_idx] |= endpoints
                break
        else:
            batches.append([flow])
            endpoints_used.append(endpoints)

    return batches


__all__ = [
    "MigrationFlow",
    "schedule_migration_batches",
]
