# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod

import torch


class EplbPolicy(ABC):

    @abstractmethod
    def rebalance_experts(self, old_global_expert_indices, weight, num_replicas, num_groups, num_nodes, num_ranks):
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

    @staticmethod
    def convert_format(current_expert_table, expert_workload, num_expert=256):
        num_layer = expert_workload.shape[0]
        converted_expert_workload = torch.zeros((num_layer, num_expert))
        for layer_id, layer in enumerate(current_expert_table):
            for rank_id, rank in enumerate(layer):
                for index, expert_id in enumerate(rank):
                    converted_expert_workload[layer_id][expert_id] += expert_workload[layer_id][rank_id][index]

        return converted_expert_workload
    @staticmethod
    def convert_table(current_expert_table, num_layer):
        return current_expert_table.reshape(num_layer, -1)