# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Schedule EPLB expert migrations into conflict-free batches."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MigrationInstruction:
    """One remote expert transfer."""

    src_rank: int
    dst_rank: int
    expert_id: int


def _ranks_with_expert(
    indices: np.ndarray, expert_id: int, num_local_experts: int
) -> list[int]:
    positions = np.flatnonzero(indices == expert_id)
    return sorted({int(pos // num_local_experts) for pos in positions})


def build_migration_instructions(
    num_local_experts: int,
    old_indices: np.ndarray,
    new_indices: np.ndarray,
) -> list[MigrationInstruction]:
    """Build the deterministic remote transfers for an expert remapping."""
    old_experts = old_indices[old_indices != -1]
    new_experts = new_indices[new_indices != -1]
    if old_experts.size == 0 and new_experts.size == 0:
        return []

    expert_ids = np.unique(np.concatenate((old_experts, new_experts)))
    instructions: list[MigrationInstruction] = []
    for expert_id in expert_ids.tolist():
        send_ranks = _ranks_with_expert(old_indices, int(expert_id), num_local_experts)
        recv_ranks = [
            rank
            for rank in _ranks_with_expert(
                new_indices, int(expert_id), num_local_experts
            )
            if rank not in send_ranks
        ]
        if not send_ranks or not recv_ranks:
            continue

        num_per_sender, remainder = divmod(len(recv_ranks), len(send_ranks))
        remainder_start = len(send_ranks) * num_per_sender
        for sender_idx, src_rank in enumerate(send_ranks):
            start = sender_idx * num_per_sender
            assigned_ranks = recv_ranks[start : start + num_per_sender]
            if sender_idx < remainder:
                assigned_ranks.append(recv_ranks[remainder_start + sender_idx])
            instructions.extend(
                MigrationInstruction(src_rank, dst_rank, int(expert_id))
                for dst_rank in assigned_ranks
            )

    return instructions


def schedule_migration_batches(
    instructions: list[MigrationInstruction],
) -> list[list[MigrationInstruction]]:
    """Greedily group transfers so each rank uses at most one peer per batch.

    Transfers with the same directed rank pair are kept in one flow. Flows are
    assigned to the first batch where neither endpoint is already in use.
    """
    flows_by_pair: dict[tuple[int, int], list[MigrationInstruction]] = {}
    for instruction in instructions:
        key = (instruction.src_rank, instruction.dst_rank)
        flows_by_pair.setdefault(key, []).append(instruction)

    batches: list[list[MigrationInstruction]] = []
    endpoints_used: list[set[int]] = []
    for flow in flows_by_pair.values():
        src_rank = flow[0].src_rank
        dst_rank = flow[0].dst_rank
        for batch, used in zip(batches, endpoints_used):
            if src_rank not in used and dst_rank not in used:
                batch.extend(flow)
                used.update((src_rank, dst_rank))
                break
        else:
            batches.append(list(flow))
            endpoints_used.append({src_rank, dst_rank})

    return batches


__all__ = [
    "MigrationInstruction",
    "build_migration_instructions",
    "schedule_migration_batches",
]
