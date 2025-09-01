# Copyright Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
from collections import defaultdict
from typing import cast

import numpy as np

from vllm.distributed.eplb.eplb_policy.abstract_policy import DynamicConfig, EplbPolicy


class DynamicTable:
    # workload_table:
    # 3D matrix: [layer, gpus, experts_per_gpu_per_layer] -> value: workload (heat) at the corresponding position
    # Size: number of layers * number of GPUs * number of experts per GPU per layer
    # The element at (i, j, k) represents the workload (heat) of the k-th expert on the j-th GPU in the i-th layer
    # For experts that are not available or collected, the value is set to -1
    workload_table = None

    # placement_table:
    # 3D matrix: [layer, gpus, experts_per_gpu_per_layer] -> value: physical expert ID at the corresponding position
    # Size: number of layers * number of GPUs * number of experts per GPU per layer
    # The element at (i, j, k) represents the physical expert ID of the k-th expert on the j-th GPU in the i-th layer
    # For experts that are not available or collected, the value is set to -1
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

    @staticmethod
    # Split hot (high-load) experts into redundant experts
    def original_compute_balanced_pack_redundancy(origin_weights, card_num,
                                                  num_redundancy_expert):
        # Step 1: Sort the items by weight in descending order (we are sorting by weight now)
        # Sort based on the second element (the second value of each tuple)
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

        # Step 2: Calculate the number of items per box
        expert_num = route_expert_num + num_redundancy_expert
        items_per_box = expert_num // card_num  # Number of items per box
        remaining_items = expert_num % card_num  # Number of items per box

        # Step 3: Initialize card_num boxes with empty lists to store item IDs
        boxes: list[list[int]] = [[] for _ in range(card_num)]
        boxes_weights: list[list[float]] = [[] for _ in range(card_num)]
        box_weights = [0] * card_num  # To store the total weight of each box
        box_counts = [0] * card_num  # To store the number of items in each box
        index = 0
        for i in range(route_expert_num):
            redundancy_num = len(route_expert_redundancy[i])
            for _ in range(redundancy_num):
                cur_weight = 0
                for item, weight in origin_weights:
                    if item == i:
                        cur_weight = weight

                boxes[index].append(i)
                boxes_weights[index].append(cur_weight)
                box_weights[index] += cur_weight
                box_counts[index] += 1
                index += 1

        sorted_indices = np.argsort([t[1] for t in origin_weights],
                                    kind='stable')[::-1]
        origin_weights = [origin_weights[idx] for idx in sorted_indices]
        # Step 4: Distribute items into boxes based on weight
        for item_id, weight in origin_weights:
            # Find the box with the least items but not full
            min_box_index = -1
            for i in range(card_num):
                if item_id in boxes[i]:
                    continue
                # Only choose boxes that still have space (box_counts[i] < items_per_box)
                if box_counts[i] < items_per_box or (box_counts[i]
                                                     == items_per_box
                                                     and remaining_items > 0):
                    if min_box_index == -1 or box_weights[i] < box_weights[
                            min_box_index]:
                        min_box_index = i

            # Place the item (id) into the selected box
            boxes[min_box_index].append(item_id)
            boxes_weights[min_box_index].append(weight)
            box_weights[min_box_index] += weight
            box_counts[min_box_index] += 1

            # If there's an imbalance in the remaining items, reduce the "remaining_items" counter
            if box_counts[min_box_index] == (items_per_box +
                                             1) and remaining_items > 0:
                remaining_items -= 1

        # Step 5: Output each box's contents and total weight
        result = []
        for i in range(card_num):
            result.append({
                "box_index": i + 1,
                "items": boxes[i],  # List of item IDs in the box
                "weight": boxes_weights[i],
                "total_weight": box_weights[i],  # Total weight in this box
                "item_count": box_counts[i]  # Number of items in the box
            })

        return result, boxes

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
    def calculate_max_heat_per_layer(workload_table, layer_num):
        max_heat_per_layer: list[float] = []
        for layer_idx in range(layer_num):
            npu_heats_now = np.sum(workload_table[layer_idx], axis=1)
            max_heat_per_layer.append(np.max(npu_heats_now))
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

    def rebalance_experts(self, current_expert_table, expert_workload):

        info = DynamicTable()
        info.workload_table = np.array(expert_workload)
        info.placement_table = np.array(current_expert_table)
        assert info.workload_table is not None
        layer_num, num_npus, experts_per_npu = info.workload_table.shape
        assert info.placement_table is not None
        row = cast(np.ndarray, info.placement_table[0])
        expert_ids, counts = np.unique(row, return_counts=True)
        num_redundancy_expert = self.get_redundant_num(num_npus, counts)
        num_original_expert = len(expert_ids)
        layer_workloads = self.add_redundant(info.placement_table,
                                             info.workload_table,
                                             num_original_expert)
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

        if num_npus <= 0:
            raise ValueError("the number of NPUs must be greater than 0")

        if num_npus < num_redundancy_expert:
            raise ValueError(
                f"the number of NPUs {num_npus} must be greater than or equal to the number of redundant experts {num_redundancy_expert}"
            )

        # Number of experts deployed on each card includes one redundant expert
        global_deployment: list[list[list[int]]] = [[[]
                                                     for _ in range(num_npus)]
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
            result, layer_deployment = self.original_compute_balanced_pack_redundancy(
                weights, num_npus, num_redundancy_expert)

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

        return change, per_layer_priority, np.array(
            new_global_deployment).tolist()
