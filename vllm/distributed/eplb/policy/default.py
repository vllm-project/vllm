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

import numpy as np
import torch

from .abstract import AbstractEplbPolicy


class DefaultEplbPolicy(AbstractEplbPolicy):
    @classmethod
    def balanced_packing(
        cls, weight: torch.Tensor, num_packs: int
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

        device = weight.device

        if groups_per_pack == 1:
            pack_index = torch.arange(
                weight.size(-1), dtype=torch.int64, device=device
            ).expand(weight.shape)
            rank_in_pack = torch.zeros_like(weight, dtype=torch.int64, device=device)
            return pack_index, rank_in_pack

        weight_np = weight.cpu().numpy()

        # Sort and get indices in decending order
        indices_np = np.argsort(-weight_np, axis=-1)

        pack_index_np = np.full((num_layers, num_groups), -1, dtype=np.int64)
        rank_in_pack_np = np.full((num_layers, num_groups), -1, dtype=np.int64)

        # Run the packing algorithm
        for i in range(num_layers):
            pack_weights = [0.0] * num_packs
            pack_items = [0] * num_packs

            for group in indices_np[i]:
                # Find a pack with capacity that has the lowest weight
                pack = min(
                    (j for j in range(num_packs) if pack_items[j] < groups_per_pack),
                    key=pack_weights.__getitem__,
                )

                assert pack_items[pack] < groups_per_pack
                pack_index_np[i, group] = pack
                rank_in_pack_np[i, group] = pack_items[pack]
                pack_weights[pack] += weight_np[i, group]
                pack_items[pack] += 1

        pack_index = torch.from_numpy(pack_index_np).to(device)
        rank_in_pack = torch.from_numpy(rank_in_pack_np).to(device)

        return pack_index, rank_in_pack

    @classmethod
    def replicate_experts(
        cls, weight: torch.Tensor, num_phy: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Replicate `num_log` experts to `num_phy` replicas, such that the maximum
        load of all replicas is minimized.

        Parameters:
            weight: [X, num_log]
            num_phy: total number of experts after replication

        Returns:
            phy2log: [X, num_phy], logical expert id of each physical expert
            replica_idx: [X, num_phy], the index of the replica for each logical expert
            logcnt: [X, num_log], number of replicas for each logical expert
        """
        n, num_log = weight.shape
        num_redundant = num_phy - num_log
        assert num_redundant >= 0
        device = weight.device
        phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
        replica_idx = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
        logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
        arangen = torch.arange(n, dtype=torch.int64, device=device)
        for i in range(num_log, num_phy):
            redundant_indices = (weight / logcnt).max(dim=-1).indices
            phy2log[:, i] = redundant_indices
            replica_idx[:, i] = logcnt[arangen, redundant_indices]
            logcnt[arangen, redundant_indices] += 1
        return phy2log, replica_idx, logcnt

    @classmethod
    def rebalance_experts_hierarchical(
        cls,
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
                (e.g, NVLink) is faster
            num_gpus: number of GPUs, must be a multiple of `num_nodes`

        Returns:
            phy2log: [layers, num_replicas], the expert
                index of each replica
            pphy_replicas_idx: [layers, num_logical_experts, X],
                the replica indices for each expert
            logcnt: [layers, num_logical_experts], number of
                physical replicas for each logical expert
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
                torch.arange(
                    perm.size(1), dtype=torch.int64, device=perm.device
                ).expand(perm.shape),
            )
            return inv

        # Step 1: pack groups to nodes
        tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
        group_pack_index, group_rank_in_pack = cls.balanced_packing(
            tokens_per_group, num_nodes
        )
        log2mlog = (
            (
                (group_pack_index * groups_per_node + group_rank_in_pack) * group_size
            ).unsqueeze(-1)
            + torch.arange(
                group_size, dtype=torch.int64, device=group_pack_index.device
            )
        ).flatten(-2)
        mlog2log = inverse(log2mlog)

        # Step 2: construct redundant experts within nodes
        # [num_layers * num_nodes, num_logical_experts // num_nodes]
        tokens_per_mlog = weight.gather(-1, mlog2log).view(
            -1, num_logical_experts // num_nodes
        )
        phy2mlog, replicas_idx, mlogcnt = cls.replicate_experts(
            tokens_per_mlog, num_physical_experts // num_nodes
        )

        # Step 3: pack physical_experts to GPUs
        # [num_layers * num_nodes, num_physical_experts // num_nodes]
        tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
        pack_index, rank_in_pack = cls.balanced_packing(
            tokens_per_phy, num_gpus // num_nodes
        )
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
        pphy_replicas_idx = replicas_idx.gather(-1, pphy2phy).view(num_layers, -1)
        logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
        return pphy2log, pphy_replicas_idx, logcnt

    @classmethod
    def preserve_intragpu_slots(
        cls,
        phy2log: torch.Tensor,
        phy_replicas_idx: torch.Tensor,
        num_ranks: int,
        old_global_expert_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reorder the new mapping per GPU so that experts that remain on the same GPU
        keep their previous slot positions when possible. Incoming experts to that GPU
        fill any remaining available slots. This is applied only when the number of GPUs
        is unchanged and the slots per GPU remain the same between
        the old and new mappings.
        """
        device = phy2log.device
        num_phy_experts = phy2log.shape[1]
        if num_ranks <= 0 or num_phy_experts % num_ranks != 0:
            return phy2log, phy_replicas_idx

        # Move to CPU and convert to NumPy for processing
        new_phy2log_np = phy2log.cpu().numpy()
        replicas_idx_np = phy_replicas_idx.cpu().numpy()
        old_phy2log_np = old_global_expert_indices.cpu().numpy()

        slots_per_gpu = num_phy_experts // num_ranks
        num_layers = new_phy2log_np.shape[0]

        post_phy2log_np = new_phy2log_np.copy()
        post_phy_replicas_idx_np = replicas_idx_np.copy()

        for gpu_idx in range(num_ranks):
            start = gpu_idx * slots_per_gpu
            end = start + slots_per_gpu
            # Experts across all layers for this GPU
            old_local = old_phy2log_np[:, start:end]  # [layers, slots]
            new_local = new_phy2log_np[:, start:end]  # [layers, slots]
            new_ridx = replicas_idx_np[:, start:end]  # [layers, slots]

            used_new_indices = np.zeros((num_layers, slots_per_gpu), dtype=bool)
            preserved_positions = np.zeros((num_layers, slots_per_gpu), dtype=bool)

            # First pass: preserve same-logical experts in their previous slots
            for slot_idx in range(slots_per_gpu):
                # matches: [layers, slots], True where new local experts have
                # the same logical value as the old from 'slot_idx' and not checked yet
                matches = (new_local == old_local[:, slot_idx][:, None]) & (
                    ~used_new_indices
                )
                has_any = matches.any(axis=1)
                if np.any(has_any):
                    first_idx = np.argmax(matches, axis=1)
                    layer_indices = np.nonzero(has_any)[0]
                    matched_new_positions = first_idx[layer_indices]
                    post_phy2log_np[layer_indices, start + slot_idx] = new_local[
                        layer_indices, matched_new_positions
                    ]
                    post_phy_replicas_idx_np[layer_indices, start + slot_idx] = (
                        new_ridx[layer_indices, matched_new_positions]
                    )
                    used_new_indices[layer_indices, matched_new_positions] = True
                    preserved_positions[layer_indices, slot_idx] = True

            # Second pass: fill remaining slots with remaining new experts
            remaining_mask = ~used_new_indices  # [layers, slots]
            fill_mask = ~preserved_positions  # [layers, slots]
            if remaining_mask.any() and fill_mask.any():
                idx_base = np.tile(np.arange(slots_per_gpu), (num_layers, 1))
                # Sentinel value for unavailable positions.
                large = slots_per_gpu + 1
                # Priorities: keep original index for available spots, set sentinel
                # for unavailable; lower is earlier.
                remaining_priority = np.where(remaining_mask, idx_base, large)
                fill_priority = np.where(fill_mask, idx_base, large)
                # Sort to get ordered indices of available src/dst positions per layer.
                remaining_indices = np.argsort(remaining_priority, axis=1)
                fill_indices = np.argsort(fill_priority, axis=1)
                # Fill count per layer (cannot exceed either side).
                remaining_counts = remaining_mask.sum(axis=1)
                fill_counts = fill_mask.sum(axis=1)
                take_counts = np.minimum(remaining_counts, fill_counts)
                # Assign remaining new experts to remaining slots per layer.
                for layer_idx in range(num_layers):
                    k = int(take_counts[layer_idx])
                    if k <= 0:
                        continue
                    src_pos = remaining_indices[layer_idx, :k]
                    dst_pos = fill_indices[layer_idx, :k]
                    post_phy2log_np[layer_idx, start + dst_pos] = new_local[
                        layer_idx, src_pos
                    ]
                    post_phy_replicas_idx_np[layer_idx, start + dst_pos] = new_ridx[
                        layer_idx, src_pos
                    ]

        # Convert back to torch and move to original device
        post_phy2log = torch.from_numpy(post_phy2log_np).to(device)
        post_phy_replicas_idx = torch.from_numpy(post_phy_replicas_idx_np).to(device)
        return post_phy2log, post_phy_replicas_idx

    @classmethod
    def rebalance_experts(
        cls,
        weight: torch.Tensor,
        num_replicas: int,
        num_groups: int,
        num_nodes: int,
        num_ranks: int,
        old_global_expert_indices: torch.Tensor | None = None,
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
            num_ranks: number of ranks, must be a multiple of `num_nodes`
            old_global_expert_indices: [layers, num_logical_experts], the old global
                expert indices. Used to avoid unnecessary weight copying
                for experts moving within one rank.
        Returns:
            phy2log: [layers, num_replicas], the expert
                index of each replica
            log2phy: [layers, num_logical_experts, X],
                the replica indices for each expert
            logcnt: [layers, num_logical_experts], number of
                physical replicas for each logical expert
        """
        num_layers, num_logical_experts = weight.shape
        weight = weight.float()
        if num_groups % num_nodes == 0:
            # use hierarchical load-balance policy
            phy2log, phy_replicas_idx, logcnt = cls.rebalance_experts_hierarchical(
                weight, num_replicas, num_groups, num_nodes, num_ranks
            )
        else:
            # use global load-balance policy
            phy2log, phy_replicas_idx, logcnt = cls.rebalance_experts_hierarchical(
                weight, num_replicas, 1, 1, num_ranks
            )
        # Optional postprocessing to preserve slots for experts moving
        # within the same GPU
        # Only apply when the number of GPUs and slots per GPU remain unchanged.
        # Helps to avoid unnecessary weight copying when experts move
        # within the same GPU.
        if old_global_expert_indices is not None:
            phy2log, phy_replicas_idx = cls.preserve_intragpu_slots(
                phy2log, phy_replicas_idx, num_ranks, old_global_expert_indices
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
            phy2log * maxlogcnt + phy_replicas_idx,
            torch.arange(num_replicas, dtype=torch.int64, device=log2phy.device).expand(
                num_layers, -1
            ),
        )
        return phy2log, log2phy, logcnt
