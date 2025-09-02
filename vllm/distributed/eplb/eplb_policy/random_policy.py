# Copyright # Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import copy
import random

from vllm.distributed.eplb.eplb_policy.abstract_v2_policy import DynamicConfig, EplbPolicy

random.seed(42)


class RandomLoadBalance(EplbPolicy):

    def __init__(self, config: DynamicConfig):
        super().__init__(config)

    def rebalance_experts(self, current_expert_table, expert_workload):
        new_table = copy.deepcopy(current_expert_table)
        num_layers = len(current_expert_table)

        for i in range(num_layers):
            # randomly choose two card
            # indices = random.sample(range(num_card), 2)
            indices = [3, 1]

            # swap redundant experts
            expert_id_to_exchange = new_table[i][indices[0]][-1].clone()
            new_table[i][indices[0]][-1] = new_table[i][indices[1]][-1]
            new_table[i][indices[1]][-1] = expert_id_to_exchange

        return 1, [-i for i in range(num_layers)], new_table
