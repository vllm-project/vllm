# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod


class EplbPolicy:

    @abstractmethod
    def rebalance_experts(self, weight, num_replicas, num_groups, num_nodes,
                          num_ranks, old_global_expert_indices):
        """
        Entry point for expert-parallelism load balancer.

        Parameters:
            old_global_expert_indices: [num_moe_layers, num_physical_experts],
            mapping from physical experts to logical experts.
            weight: [layers, num_logical_experts],the load statistics
            for all logical experts
            num_replicas: number of physical experts, must be a multiple of
                `num_ranks`
            num_groups: number of expert groups
            num_nodes: number of server nodes
            num_ranks: number of ranks, must be a multiple of `num_nodes`

        Returns:
            physical_to_logical_map: [layers, num_replicas], the expert
             index of each replica
            logical_to_physical_map: [layers, num_logical_experts, X],
             the replica indices for each expert
            expert_count: [layers, num_logical_experts], number of
            physical replicas for each logical expert
        """
        pass
