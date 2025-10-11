# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
from collections.abc import MutableSequence, Sequence
from functools import partial

import torch


def idx_local_to_global(
    local_idx: int,
    local_cnt: int,
    ep_rank: int,
) -> int:
    """
    Convert a local expert index to a global expert index.
    """
    return ep_rank * local_cnt + local_idx


def idx_global_to_local(
    global_idx: int,
    local_cnt: int,
    ep_rank: int,
) -> int:
    """
    Convert a global expert index to a local expert index.
    """
    return global_idx - ep_rank * local_cnt


def global_idx_to_rank(
    global_idx: int,
    local_cnt: int,
) -> int:
    """
    Convert a global expert index to a rank index.
    """
    return global_idx // local_cnt


def get_ep_ranks_with_expert(
    idx: int,
    num_local_experts: int,
    old_indices: Sequence[int],
    new_indices: Sequence[int],
) -> tuple[MutableSequence[int], MutableSequence[int]]:
    """
    Get the ranks of the experts that need to be exchanged.

    Args:
        idx: The index of the expert.
        num_local_experts: The number of local experts.
        old_indices: The old indices of the experts.
        new_indices: The new indices of the experts.

    Returns:
        A tuple of two lists:
        - The ranks of the experts that need to be sent.
        - The ranks of the experts that need to be received.
    """
    global2rank = partial(
        global_idx_to_rank,
        local_cnt=num_local_experts,
    )

    ranks_to_send: list[int] = []
    ranks_to_recv: list[int] = []

    for i, e in enumerate(old_indices):
        if e == idx:
            rank = global2rank(i)
            if not ranks_to_send or ranks_to_send[-1] != rank:
                ranks_to_send.append(rank)

    for i, e in enumerate(new_indices):
        if e == idx:
            rank = global2rank(i)
            if not ranks_to_recv or ranks_to_recv[-1] != rank:
                ranks_to_recv.append(rank)

    # Remove those ranks that can get this expert locally.
    ranks_to_send_set = set(ranks_to_send)
    ranks_to_recv_actual = [
        rank for rank in ranks_to_recv if rank not in ranks_to_send_set
    ]

    return ranks_to_send, ranks_to_recv_actual


def generate_log2phy_map(expert_map):
    """
    Generates a logical-to-physical expert mapping for all ranks based on an
    initial expert distribution map. This map indicates which physical expert
    slot (on which rank) corresponds to a given logical expert. It handles
    cases where an expert might not be present on all ranks and fills in
    missing entries by replicating existing ones.

    Args:
        expert_map: A 2D tensor of shape [num_ranks, num_global_experts].
                    `expert_map[r, g]` contains the local physical ID of
                    global expert `g` on rank `r`, or -1 if global expert
                     `g` is not on rank `r`.

    Returns:
        A 2D tensor `log2phy_map` of shape [num_ranks, num_global_experts].
        `log2phy_map[r, g]` will contain the *global physical ID* of the
        expert that rank `r` should use for logical expert `g`.
        A global physical ID is
        `rank_id * num_local_experts + local_physical_expert_id`.
    """
    num_local_experts = expert_map.max() + 1
    log2phy_map = expert_map.clone()
    num_ranks, num_global_expert = log2phy_map.shape

    row_indices = (
        torch.arange(num_ranks).view(-1, 1).expand(num_ranks, num_global_expert)
        * num_local_experts
    )
    log2phy_map[log2phy_map != -1] += row_indices[log2phy_map != -1]

    for idx in range(num_global_expert):
        positive_rank_idx = torch.where(log2phy_map[:, idx] != -1)[0]
        negative_rank_idx = torch.where(log2phy_map[:, idx] == -1)[0]
        num_rank_holding_expert = positive_rank_idx.size(0)

        if num_rank_holding_expert == 1:
            log2phy_map[negative_rank_idx, idx] = torch.full(
                (num_ranks - 1,),
                log2phy_map[positive_rank_idx, idx].item(),
                dtype=log2phy_map.dtype,
            )
        else:
            holding_ranks_values = log2phy_map[positive_rank_idx, idx].tolist()
            random_list = [
                random.choice(holding_ranks_values)
                for _ in range(num_ranks - num_rank_holding_expert)
            ]
            log2phy_map[negative_rank_idx, idx] = torch.tensor(
                random_list, dtype=log2phy_map.dtype
            )

    return log2phy_map


def determine_default_log2phy_map(global_expert_num, world_size, rank_id):
    """
    Determines a default logical-to-physical expert mapping for a specific
    rank. This function sets up an initial, balanced distribution where
    experts are partitioned across ranks, and then uses
    `generate_log2phy_map` to create the final mapping, including
    replication for unassigned experts.

    Args:
        global_expert_num: The total number of logical experts in the system.
        world_size: The total number of ranks in the expert parallelism group.
        rank_id: The ID of the current rank.

    Returns:
        A 1D tensor representing the logical-to-physical mapping for the
        specified `rank_id`.
        Shape: [num_global_experts].
        Each element `log2phy_map[g]` is the global physical ID of the expert
        that `rank_id` should use for logical expert `g`.
    """
    local_num_experts = global_expert_num // world_size

    expert_map_all = torch.full((world_size, global_expert_num), -1, dtype=torch.int32)

    for r in range(world_size):
        if r < world_size - 1:
            start = r * local_num_experts
            end = (r + 1) * local_num_experts
            local_count = local_num_experts
        else:
            start = r * local_num_experts
            end = global_expert_num
            local_count = global_expert_num - r * local_num_experts

        local_ids = torch.arange(local_count, dtype=torch.int32)
        expert_map_all[r, start:end] = local_ids

    log2phy_map_all = generate_log2phy_map(expert_map_all)

    return log2phy_map_all[rank_id]
