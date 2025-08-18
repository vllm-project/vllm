# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from abc import abstractmethod


class DynamicConfig:
    placement_policy = None

    max_transferred_expert_per_layer = 100  # Maximum number of experts that can be migrated per layer on a single host
    ep_worldsize = 64  # Total number of dies across the entire cluster where experts are distributed
    num_die_per_host = 8  # Number of dies on each host machine


class EplbPolicy:

    def __init__(self, config: DynamicConfig):
        self.config = config

    @abstractmethod
    def rebalance_experts(self, current_expert_table, expert_workload, num_replicas, num_groups, num_nodes, num_ranks):
        """
        Mapping from physical experts to logical experts.

        Shape: (num_moe_layers, num_physical_experts)

        # Example

        For a 2-layer MoE model with 6 physical experts and 4 logical experts on 3
        EP ranks, the mapping could look like this:

        ```
        [[0, 1, 2, 3, 0, 1],
         [0, 2, 0, 1, 0, 3]]
        ```
        weight: [layers, num_logical_experts], the load statistics for all logical experts
        """
        pass
