from abc import abstractmethod

import torch


class DynamicConfig:
    placement_policy = None

    max_transferred_expert_per_layer = 100  # Maximum number of experts that can be migrated per layer on a single host
    ep_worldsize = 64  # Total number of dies across the entire cluster where experts are distributed
    num_die_per_host = 8  # Number of dies on each host machine


class EplbPolicy:

    def __init__(self, config: DynamicConfig):
        self.config = config

    @abstractmethod
    def rebalance_experts(self, current_expert_table, expert_workload):
        """
        Pass in the weights and return expert replication and placement under relevant constraints.
        INPUT:
        current_expert_table: [layerId, rankId, expert_num_i]
        expert_workload = expert_table[layer0][rankId][expert_num_i]

        RETURNED: (res, expert_table)
        res:
        1 -- table_changed
        0 -- not_changed

        expert_table: [layerId, rankId, expert_num_i]
        expert_num_i --- [0, MaxExpertPerRank]
        expertID = expert_table[layer0][rankId][expert_num_i]
        array_values:
        [0, 1, 2, 3, 248]
        [4, 5, 6, 7, 254]
        [8, 9, 10, 11, 71]
        ...
        [252, 253, 254, 255, 0]
        """
        pass


