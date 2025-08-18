# Copyright Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
from collections import defaultdict
from typing import cast

import numpy as np
import torch

from .policy_abstract import DynamicConfig, EplbPolicy


class DynamicTable:

    workload_table = None

    placement_table = None


class DynamicEplb(EplbPolicy):

    def __init__(self, config: DynamicConfig):
        super().__init__(config)

    @staticmethod
    def add_redundant(current_expert_table, expert_workload,
                      num_original_expert):
        layer_num, npu_num, experts_per_npu = expert_workload.shape
        workload_new = np.zeros((layer_num, num_original_expert))
        for layer_idx in range(layer_num):
            workload_dict: dict[int, int] = defaultdict(int)
            placement_layer = current_expert_table[layer_idx].copy()
            workload_layer = expert_workload[layer_idx].copy()
            for npu_idx in range(npu_num):
                for expert_idx in range(experts_per_npu):
                    workload_dict[placement_layer[npu_idx][
                        expert_idx]] += workload_layer[npu_idx][expert_idx]
            for expert_idx in range(num_original_expert):
                workload_new[layer_idx][expert_idx] = workload_dict[expert_idx]
        return workload_new


    # Split hot (high-load) experts into redundant experts
    @staticmethod
    def compute_balanced_pack_redundancy(origin_weights, card_num,
                                         num_redundancy_expert):
        route_expert_num = len(origin_weights)
        route_expert_redundancy: list[list[int]] = [
            [] for _ in range(route_expert_num)
        ]
        for i in range(num_redundancy_expert):
            sorted_indices = np.argsort([t[1] for t in origin_weights],
                                        kind='stable')[::-1]
            weights = [origin_weights[idx] for idx in sorted_indices]
            tmp_raw_weight = weights[0][1] * (
                len(route_expert_redundancy[weights[0][0]]) + 1)
            route_expert_redundancy[weights[0][0]].append(route_expert_num + i)
            avg_weight = tmp_raw_weight / (
                len(route_expert_redundancy[weights[0][0]]) + 1)
            weights[0] = (weights[0][0], avg_weight)
            origin_weights = weights

        expert_num = route_expert_num + num_redundancy_expert
        if card_num == 0:
            raise RuntimeError("card_num can not be 0.")
        items_per_box = expert_num // card_num
        remaining_items = expert_num % card_num

        boxes: list[list[int]] = [[] for _ in range(card_num)]
        boxes_weights: list[list[float]] = [[] for _ in range(card_num)]
        box_weights = [0] * card_num
        box_counts = [0] * card_num

        all_weights = np.zeros((expert_num, ), dtype='object')
        all_weights[:route_expert_num] = origin_weights

        index = route_expert_num
        for i in range(route_expert_num):
            redundancy_num = len(route_expert_redundancy[i])
            for _ in range(redundancy_num):
                for item, weight in origin_weights:
                    if item == i:
                        all_weights[index] = (item, weight)
                        index += 1

        sorted_indices = np.argsort([t[1] for t in all_weights],
                                    kind='stable')[::-1]
        all_weights = [all_weights[idx] for idx in sorted_indices]
        for item_id, weight in all_weights:
            min_box_index = -1
            for i in range(card_num):
                if box_counts[i] < items_per_box or (box_counts[i]
                                                     == items_per_box
                                                     and remaining_items > 0):
                    if min_box_index == -1 or box_weights[i] < box_weights[
                            min_box_index]:
                        if item_id not in boxes[i]:
                            min_box_index = i

            boxes[min_box_index].append(item_id)
            boxes_weights[min_box_index].append(weight)
            box_weights[min_box_index] += weight
            box_counts[min_box_index] += 1

            if box_counts[min_box_index] == (items_per_box + 1) and remaining_items > 0:
                remaining_items -= 1

        result = []
        for i in range(card_num):
            result.append({
                "box_index": i + 1,
                "items": boxes[i],
                "weight": boxes_weights[i],
                "total_weight": box_weights[i],
                "item_count": box_counts[i]
            })

        return result, boxes

    # Scheme without redundant experts
    @staticmethod
    def compute_balanced_pack(origin_weights, card_num):
        sorted_indices = np.argsort([t[1] for t in origin_weights])[::-1]
        weights = origin_weights[sorted_indices]
        expert_num = len(weights)
        if card_num == 0:
            raise RuntimeError("card_num can not be 0.")
        items_per_box = expert_num // card_num
        remaining_items = expert_num % card_num

        boxes: list[list[int]] = [[] for _ in range(card_num)]
        boxes_weights: list[list[float]] = [[] for _ in range(card_num)]
        box_weights = [0] * card_num
        box_counts = [0] * card_num

        for item_id, weight in weights:
            min_box_index = -1
            for i in range(card_num):
                if box_counts[i] < items_per_box or (box_counts[i]
                                                     == items_per_box
                                                     and remaining_items > 0):
                    if min_box_index == -1 or box_weights[i] < box_weights[
                            min_box_index]:
                        min_box_index = i

            boxes[min_box_index].append(item_id)
            boxes_weights[min_box_index].append(weight)
            box_weights[min_box_index] += weight
            box_counts[min_box_index] += 1

            if box_counts[min_box_index] == (items_per_box +
                                             1) and remaining_items > 0:
                remaining_items -= 1

        result = []
        for i in range(card_num):
            result.append({
                "box_index": i + 1,
                "items": boxes[i],
                "weight": boxes_weights[i],
                "total_weight": box_weights[i],
                "item_count": box_counts[i]
            })

        return result, boxes

    @staticmethod
    def get_redundant_num(npu_num, counts):
        redundant_num_each_npu: int = np.sum(counts - 1)
        return redundant_num_each_npu

    @staticmethod
    def calculate_max_heat_per_layer(global_deployment, layer_workloads):

        max_heat_per_layer = []
        expert_num = np.zeros_like(layer_workloads)
        for layer_id, layer in enumerate(global_deployment):
            for device in layer:
                for expert_id in device:
                    expert_num[layer_id][expert_id] += 1

        for layer_id, layer in enumerate(global_deployment):
            cur_layer_max_workload = 0
            for box in layer:
                box_workload = 0
                for expert_id in box:
                    update_workload = layer_workloads[layer_id][expert_id] / expert_num[layer_id][expert_id]
                    box_workload += update_workload
                if cur_layer_max_workload < box_workload:
                    cur_layer_max_workload = box_workload
            max_heat_per_layer.append(cur_layer_max_workload)

        return max_heat_per_layer

    @staticmethod
    def constraint_expert_local_exchange(current_expert_table,
                                         global_deployment):
        for layer_id in range(len(global_deployment)):
            for card_id in range(len(global_deployment[layer_id])):
                current_list = [
                    int(x) for x in current_expert_table[layer_id][card_id]
                ]
                new_list = [
                    int(x) for x in global_deployment[layer_id][card_id]
                ]
                num = len(new_list)

                new_index = [-1] * num
                new_result = [-1] * num
                remaining_elements = []

                for i in range(num):
                    flag = True
                    for j in range(num):
                        if new_list[i] == current_list[j] and new_index[
                                j] == -1:
                            new_index[j] = 0
                            new_result[j] = current_list[j]
                            flag = False
                            break
                    if flag:
                        remaining_elements.append(new_list[i])

                index = 0
                for k in range(num):
                    if new_result[k] == -1:
                        new_result[k] = remaining_elements[index]
                        index += 1

                global_deployment[layer_id][card_id] = new_result

        return global_deployment

    def gen_result(self, global_deployment, layer_num, local_expert_num):

        logical_replica_count = torch.zeros((layer_num, local_expert_num), dtype=torch.int32)
        for layer_id, layer in enumerate(global_deployment):
            for device in layer:
                for expert_id in device:
                    logical_replica_count[layer_id][expert_id] += 1

        max_expert_num = max(logical_replica_count)
        logical_to_physical_map = torch.full((layer_num, local_expert_num, max_expert_num), -1, dtype=torch.long)

        new_global_deployment = global_deployment.reshape(layer_num, -1)
        for layer_id, layer in enumerate(new_global_deployment):
            cur_expert_num = np.zeros(local_expert_num, dtype=int)
            for index, expert_id in enumerate(layer):
                logical_to_physical_map[layer_id][expert_id][cur_expert_num[expert_id]] = index
                cur_expert_num[expert_id] += 1

        physical_to_logical_map = torch.from_numpy(new_global_deployment)

        return physical_to_logical_map, logical_to_physical_map, logical_replica_count

    def rebalance_experts(self, current_expert_table, expert_workload, num_replicas, num_groups, num_nodes, num_ranks):

        info = DynamicTable()
        info.workload_table = np.array(expert_workload)
        layer_num = info.workload_table.shape[0]
        info.placement_table = np.array(current_expert_table).reshape(layer_num, num_ranks, -1)
        assert info.placement_table is not None
        row = cast(np.ndarray, info.placement_table[0])
        expert_ids, counts = np.unique(row, return_counts=True)
        num_redundancy_expert = self.get_redundant_num(num_ranks, counts)
        num_original_expert = len(expert_ids)
        layer_workloads = expert_workload
        max_heat_per_layer_before = self.calculate_max_heat_per_layer(
            info.workload_table, layer_num)
        npu_heat_all_origin = sum(max_heat_per_layer_before)

        # Perform load balancing and deploy redundant experts
        layer_num = layer_workloads.shape[0]
        expert_num = layer_workloads.shape[1]
        # Validate that the number of experts, number of cards, and number of redundant experts do not exceed the number of cards
        if num_original_expert != expert_num:
            raise ValueError(
                f"the number of original experts {num_original_expert} must be equal to expert_num {expert_num}"
            )

        if num_ranks <= 0:
            raise ValueError("the number of NPUs must be greater than 0")

        if num_ranks < num_redundancy_expert:
            raise ValueError(
                f"the number of NPUs {num_ranks} must be greater than or equal to the number of redundant experts {num_redundancy_expert}"
            )

        # Number of experts deployed on each card includes one redundant expert
        global_deployment: list[list[list[int]]] = [[[]
                                                     for _ in range(num_ranks)]
                                                    for _ in range(layer_num)]
        # Iterate to obtain the placement strategy for each layer, taking computational balance into account
        max_heat_per_layer_after = np.zeros([layer_num])
        for layer in range(layer_num):
            # Get the expert IDs and their corresponding workloads for the current layer;
            # workloads need to be normalized, and one redundant expert is added per card
            weights = np.zeros((expert_num, ), dtype='object')
            for expert_id, workload_weight in enumerate(
                    layer_workloads[layer]):
                weights[expert_id] = (expert_id, workload_weight)

            # Obtain the globally balanced placement strategy for each layer
            result, layer_deployment = self.compute_balanced_pack_redundancy(weights, num_ranks, num_redundancy_expert)

            global_deployment[layer] = layer_deployment
            max_heat_per_layer_after[layer] = max(
                result, key=lambda x: x['total_weight'])['total_weight']

        new_global_deployment = self.constraint_expert_local_exchange(
            current_expert_table, global_deployment)
        # Obtain the priority of each layer
        layer_changed_ratio = []
        for layer_idx in range(layer_num):
            layer_changed_ratio.append(max_heat_per_layer_after[layer_idx] /
                                       max_heat_per_layer_before[layer_idx])

        per_layer_priority = np.argsort(layer_changed_ratio)
        npu_heat_all_after = sum(max_heat_per_layer_after)

        change = 0
        if npu_heat_all_after < 0.95 * npu_heat_all_origin:
            change = 1

        physical_to_logical_map, logical_to_physical_map, logical_replica_count = self.gen_result(new_global_deployment, layer_num, expert_num)

        return physical_to_logical_map, logical_to_physical_map, logical_replica_count
