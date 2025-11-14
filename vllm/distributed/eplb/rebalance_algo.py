# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm.

The rearrangement algorithm is adapted from
[DeepSeek EPLB](https://github.com/deepseek-ai/eplb).

Please find at [#12](https://github.com/deepseek-ai/EPLB/issues/12) an example
on how the EPLB algorithm works.
"""

import torch


def get_effective_ep_size(
    rank_mapping: dict[int, int],
) -> int:
    """
    Calculate the effective number of active ranks from rank_mapping.
    
    Args:
        rank_mapping: Mapping from physical rank to new rank, -1 for masked ranks
        
    Returns:
        Number of active (non-masked) ranks
    """
    return sum(1 for new_rank in rank_mapping.values() if new_rank != -1)


def rebalance_masked_experts(
    weight: torch.Tensor,
    full_num_replicas: int,
    num_groups: int,
    num_nodes: int,
    full_ep_size: int,
    rank_mapping: dict[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rebalance experts with masked ranks, maintaining full ep_size representation.
    
    For health-based masking where some ranks are masked but ep_size stays constant.
    Returns full-size tensors with -1 for masked rank slots.
    
    Args:
        weight: Load statistics [layers, num_logical_experts]
                NOT affected by masking - always same size (model constant)
        full_num_replicas: Full number of physical experts INCLUDING masked ranks
                          = full_ep_size * experts_per_rank
                          e.g., 8 (not 6) for 4 ranks with 1 masked
        num_groups: Number of expert groups (model architecture constant)
                   NOT affected by masking - e.g., 8 groups always
        num_nodes: Number of physical server nodes
                  May be affected if entire nodes are masked (calculated externally)
        full_ep_size: Full ep_group.size() INCLUDING masked ranks
                     e.g., 4 (not 3) even with 1 rank masked
        rank_mapping: Mapping from physical rank to new rank (active rank index)
                     INCLUDING masked ranks: {0:0, 1:1, 2:-1, 3:2}
                     Keys 0-3 (all physical ranks), value -1 means masked
        
    Returns:
        Tensors with full ep_size representation (masked slots filled with -1):
        - physical_to_logical_map: [layers, full_num_replicas] e.g., (layers, 8)
        - logical_to_physical_map: [layers, num_logical, num_redundant_experts + 1]
          where num_redundant_experts + 1 is the maximum possible replicas per logical expert
          (unchanged, e.g., (layers, 4, 3) for 2 redundant experts)
        - logical_replica_count: [layers, num_logical] (counts active replicas only)
    """
    # Calculate active counts (EXCLUDING masked ranks)
    effective_ep_size = get_effective_ep_size(rank_mapping)  # e.g., 3 (not 4)
    experts_per_rank = full_num_replicas // full_ep_size      # e.g., 8 // 4 = 2
    active_num_replicas = effective_ep_size * experts_per_rank # e.g., 3 * 2 = 6
    
    # Call standard rebalance with ACTIVE counts only (as if we only have 3 ranks)
    # This returns tensors of size 6 (for 3 active ranks * 2 experts/rank)
    (
        active_phy2log,   # (layers, active_num_replicas=6)
        active_log2phy,   # (layers, num_logical, max_replicas_per_expert)
                          # where max = num_redundant_experts + 1
        active_logcnt,    # (layers, num_logical) - actual replica count per expert
    ) = rebalance_experts(
        weight,                  # [layers, num_logical] - unchanged
        active_num_replicas,     # 6 (active only)
        num_groups,              # Model constant (NOT affected by masking)
        num_nodes,               # May be reduced if whole nodes masked
        effective_ep_size,       # 3 (active ranks only)
    )
    
    # Expand from active representation (size 6) to full ep_size representation (size 8)
    num_layers = active_phy2log.shape[0]
    
    # Create full-size physical_to_logical_map with -1 for masked slots
    # Output: (layers, full_num_replicas=8) filled with -1
    full_phy2log = torch.full(
        (num_layers, full_num_replicas),
        -1,
        dtype=active_phy2log.dtype,
        device=active_phy2log.device,
    )
    
    # Map active ranks to their physical positions
    # Example: rank_mapping = {0:0, 1:1, 2:-1, 3:2} with experts_per_rank=2
    # Physical rank 0 → new rank 0 → copy active[0:2] to full[0:2]
    # Physical rank 1 → new rank 1 → copy active[2:4] to full[2:4]
    # Physical rank 2 → masked (-1) → full[4:6] stays as -1
    # Physical rank 3 → new rank 2 → copy active[4:6] to full[6:8]
    for physical_rank in range(full_ep_size):  # 0, 1, 2, 3
        new_rank = rank_mapping[physical_rank]
        if new_rank != -1:
            # This physical rank is active - copy its expert assignments from active tensor
            phys_start = physical_rank * experts_per_rank  # Physical position in output
            phys_end = (physical_rank + 1) * experts_per_rank
            active_start = new_rank * experts_per_rank     # Active position in input
            active_end = (new_rank + 1) * experts_per_rank
            
            full_phy2log[:, phys_start:phys_end] = active_phy2log[:, active_start:active_end]
        # else: masked rank, leave as -1 (already initialized above)
    
    # For log2phy and logcnt, we can use the active versions directly
    # since they map logical experts to physical indices (which are the same)
    # These remain unchanged because logical expert IDs don't change due to masking
    return full_phy2log, active_log2phy, active_logcnt


def balanced_packing(
    weight: torch.Tensor, num_packs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly
    n/m objects and the weights of all packs are as balanced as possible.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(
            weight.size(-1), dtype=torch.int64, device=weight.device
        ).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    indices = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full_like(weight, fill_value=-1, dtype=torch.int64, device="cpu")
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)
    for i in range(num_layers):
        pack_weights = [0] * num_packs
        pack_items = [0] * num_packs
        for group in indices[i]:
            pack = min(
                (i for i in range(num_packs) if pack_items[i] < groups_per_pack),
                key=pack_weights.__getitem__,
            )
            assert pack_items[pack] < groups_per_pack
            pack_index[i, group] = pack
            rank_in_pack[i, group] = pack_items[pack]
            pack_weights[pack] += weight[i, group]
            pack_items[pack] += 1
    return pack_index, rank_in_pack


def replicate_experts(
    weight: torch.Tensor, num_phy: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas, such that the maximum
    load of all replicas is minimized.

    Parameters:
        weight: [X, num_log]
        num_phy: total number of experts after replication

    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        redundant_indices = (weight / logcnt).max(dim=-1).indices
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1
    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Parameters:
        weight: [num_moe_layers, num_logical_experts]
        num_physical_experts: number of physical experts after replication
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network
            (e.g., NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map (torch.Tensor):
            [num_moe_layers, num_physical_experts]
        logical_to_physical_map (torch.Tensor):
            [num_moe_layers, num_logical_experts, X]
        logical_count (torch.Tensor):
            [num_moe_layers, num_logical_experts]
    """
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(
            1,
            perm,
            torch.arange(perm.size(1), dtype=torch.int64, device=perm.device).expand(
                perm.shape
            ),
        )
        return inv

    # Step 1: pack groups to nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes)
    log2mlog = (
        (
            (group_pack_index * groups_per_node + group_rank_in_pack) * group_size
        ).unsqueeze(-1)
        + torch.arange(group_size, dtype=torch.int64, device=group_pack_index.device)
    ).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: construct redundant experts within nodes
    # [num_layers * num_nodes, num_logical_experts // num_nodes]
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes
    )
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes
    )

    # Step 3: pack physical_experts to GPUs
    # [num_layers * num_nodes, num_physical_experts // num_nodes]
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(
        -1, pphy2phy
    )  # [num_layers * num_nodes, num_log_per_nodes]
    pphy2mlog = (
        pphy2mlog.view(num_layers, num_nodes, -1)
        + torch.arange(
            0,
            num_logical_experts,
            num_logical_experts // num_nodes,
            device=group_pack_index.device,
        ).view(1, -1, 1)
    ).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
    return pphy2log, pphyrank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for expert-parallelism load balancer.

    Parameters:
        weight: [layers, num_logical_experts], the load statistics for all
            logical experts
        num_replicas: number of physical experts, must be a multiple of
            `num_gpus`
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network
            (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map:
            [layers, num_replicas], the expert index of each replica
        logical_to_physical_map:
            [layers, num_logical_experts, num_redundant_experts + 1],
            the replica indices for each expert, where num_redundant_experts + 1
            is the maximum possible replicas per logical expert (padded with -1)
        expert_count:
            [layers, num_logical_experts], number of actual physical
            replicas for each logical expert
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    if num_groups % num_nodes == 0:
        # use hierarchical load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus
        )
    else:
        # use global load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus
        )
    num_redundant_experts = num_replicas - num_logical_experts
    maxlogcnt = num_redundant_experts + 1
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(num_replicas, dtype=torch.int64, device=log2phy.device).expand(
            num_layers, -1
        ),
    )
    return phy2log, log2phy, logcnt


__all__ = ["rebalance_experts"]
