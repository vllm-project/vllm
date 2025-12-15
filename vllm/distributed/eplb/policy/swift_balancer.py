# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import numpy as np
import torch
from typing import Optional
from .abstract import AbstractEplbPolicy


class DynamicTable:

    workload_table = None

    placement_table = None


class SwiftBalancerPolicy(AbstractEplbPolicy):

    @staticmethod
    def safe_divide(a, b):
        if b == 0:
            raise ValueError("Division by zero is not allowed")
        return a / b

    @staticmethod
    def safe_exact_divide(a, b):
        if b == 0:
            raise ValueError("Division by zero is not allowed")
        return a // b

    @staticmethod
    def safe_mod(a, b):
        if b == 0:
            raise ValueError("Division by zero is not allowed")
        return a % b

    @staticmethod
    def get_redundant_num(counts):
        redundant_num_each_rank = np.sum(counts - 1)
        return redundant_num_each_rank

    def calculate_initial_imbalance(self, global_deployment,
                                    new_layer_workloads):

        device_num = global_deployment.shape[1]
        layer_imbalance = []
        max_heat_per_layer = []
        expert_num = np.zeros_like(new_layer_workloads)
        for layer_id, layer in enumerate(global_deployment):
            for device in layer:
                for expert_id in device:
                    expert_num[layer_id][expert_id] += 1

        for layer_id, layer in enumerate(global_deployment):
            cur_layer_max_workload = 0
            total_workload = 0
            for box in layer:
                box_workload = 0
                for expert_id in box:
                    update_workload = self.safe_divide(
                        new_layer_workloads[layer_id][expert_id],
                        expert_num[layer_id][expert_id])
                    box_workload += update_workload
                    total_workload += update_workload
                if cur_layer_max_workload < box_workload:
                    cur_layer_max_workload = box_workload
            max_heat_per_layer.append(cur_layer_max_workload)

            cur_layer_imbalance = self.safe_divide(
                cur_layer_max_workload,
                self.safe_divide(total_workload, device_num))
            layer_imbalance.append(cur_layer_imbalance)

        return layer_imbalance, max_heat_per_layer

    def compute_redundant_assignments(
            self, weights: np.ndarray, num_redundant_experts: int,
            num_experts: int, node_id: int,
            per_node_route_expert_num: int) -> tuple[np.ndarray, dict]:
        """
        Each time, select the expert with the highest load,
        generate a redundant expert, and then update the load.

        Parameters:
            weights: [logic_experts_num_per_node], The weight of each expert
             or the weight of each expert within each node
            num_redundant_experts: The number of redundancy experts
            num_experts:The number of logic experts
            node_id: The index of node
            per_node_route_expert_num: The number of routing
            experts at each node

        Returns:
            redundancy_counts: [logic_experts_num], The number of
            redundant experts
            weight_dict:[logic_experts_num], The weight of each expert
            or the weight of each expert within each node
        """

        redundancy_counts = np.zeros(num_experts, dtype=np.int32)
        node_offset_position = node_id * per_node_route_expert_num

        for i in range(num_redundant_experts):

            index = np.argmax(weights)
            expert_idx = index + node_offset_position
            redundancy_counts[expert_idx] += 1
            cur_count = redundancy_counts[expert_idx]
            weights[index] = weights[index] * cur_count / (cur_count + 1)

        weight_dict = {
            index + node_offset_position: weight
            for index, weight in enumerate(weights)
        }

        return redundancy_counts, weight_dict

    def repeat_compute_redundant_assignments(
            self, layer_workloads: np.ndarray, rendun_pos: list,
            num_experts: int, num_exist_expert: list, device_assignments: list,
            device_counts: list, expert_from_device: np.ndarray,
            com_between_devices: list[dict[int, int]]):
        """
        Each time, select the hottest expert and redundantly place
        them on a card with available space, then update the data.
        If it cannot be placed, choose the next hottest expert,
        and so on, until an expert who can be placed is found.

        Parameters:
            layer_workloads: [logic_experts_num], the weight of each item
            rendun_pos: [num_ranks, p] Redundant positions on each device
            num_experts: Number of Logic Experts
            num_exist_expert: [logic_experts_num], Number of Exist Experts
            device_assignments: [num_ranks, num_experts_per_rank], the
            assignment of each device
            device_counts: [num_ranks] the number of devices
            expert_from_device: [logic_experts_num] The logic expert
            was on that device.
            com_between_devices: Communication status between devices

        Returns:
            sorted_weights: [logic_experts_num], the weight of each expert
            device_assignments: [num_ranks, num_experts_per_rank], the
            assignment of each device
            device_counts: [num_ranks], the number of devices
            com_between_devices: Communication status between devices
        """

        current_weights = np.zeros((num_experts, ), dtype='object')
        for expert_id, workload_weight in enumerate(layer_workloads):
            current_weights[expert_id] = (expert_id, workload_weight)

        sorted_indices = np.argsort([w for _, w in current_weights],
                                    kind='stable')[::-1]
        sorted_weights = [current_weights[i] for i in sorted_indices]

        devices_with_slots = []
        for device_id, device_rendun_pos in enumerate(rendun_pos):
            if len(device_rendun_pos) != 0:
                devices_with_slots.append(device_id)

        while devices_with_slots:
            sorted_indices = np.argsort([w for _, w in current_weights],
                                        kind='stable')[::-1]
            sorted_weights = [current_weights[i] for i in sorted_indices]

            for index, target_weight in enumerate(sorted_weights):
                expert_id, original_weight = target_weight

                if original_weight == -1:
                    raise RuntimeError(
                        "Error:Redundant expert failure re-occurred")

                redundancy_successful = False
                for cur_device_id in devices_with_slots:
                    if expert_id not in device_assignments[cur_device_id]:
                        pos = rendun_pos[cur_device_id].pop()

                        if len(rendun_pos[cur_device_id]) == 0:
                            devices_with_slots = [
                                device_id for device_id in devices_with_slots
                                if device_id != cur_device_id
                            ]

                        device_assignments[cur_device_id][pos] = expert_id
                        device_counts[cur_device_id] += 1
                        com_box_index = expert_from_device[expert_id]
                        com_between_devices[cur_device_id][com_box_index] = (
                            expert_id)
                        new_weight = self.safe_divide(
                            (original_weight * num_exist_expert[expert_id]),
                            (num_exist_expert[expert_id] + 1))
                        sorted_weights[index] = (expert_id, new_weight)
                        num_exist_expert[expert_id] += 1
                        redundancy_successful = True
                        break

                if redundancy_successful:
                    break

        sorted_indices = np.argsort(
            [weight_id for weight_id, _ in sorted_weights], kind='stable')
        sorted_weights = [sorted_weights[i][1] for i in sorted_indices]

        return (sorted_weights, device_assignments, device_counts,
                com_between_devices)

    @staticmethod
    def prepare_expert_list(updated_weights: dict[int, float],
                            redundancy_counts: np.ndarray,
                            num_redundant_experts: int) -> list:
        """
        Statistically selected redundant expert information

        Parameters:
            updated_weights: [logic_experts_num], The weight of each expert
            or the weight of each expert within each node
            num_redundant_experts: The number of redundancy experts
            redundancy_counts: [logic_experts_num], The number of
            redundant experts

        Returns:
            A list of redundant experts,
            ordered from highest to lowest load.
        """
        redundant_expert_list = np.empty(num_redundant_experts, dtype=object)
        index = 0
        num_experts = len(redundancy_counts)

        for expert_id in range(num_experts):
            for _ in range(redundancy_counts[expert_id]):
                redundant_expert_list[index] = (expert_id,
                                                updated_weights[expert_id])
                index += 1
        sorted_indices = np.argsort([w for _, w in redundant_expert_list],
                                    kind='stable')[::-1]
        return [redundant_expert_list[i] for i in sorted_indices]

    @staticmethod
    def non_redundant_expert_information(
            origin_deployment: np.ndarray, updated_weights: dict[int, float],
            rendun_pos: list) -> tuple[list, list, list, list]:
        """
        Collect information on non-experts under new load conditions

        Parameters:
            origin_deployment: [num_ranks, num_physics experts_per_rank] The
            deployment status of experts on each device
            updated_weights: [logic_experts_num], The weight of each expert
             or the weight of each expert within each node
            rendun_pos: [num_ranks, p] Redundant positions on each device

        Returns:
            device_assignments: [num_ranks, num_experts_per_rank],
            the assignment of each device
            device_weights: [num_ranks, num_experts_per_rank],
            the load of each expert on the equipment
            device_loads: [num_ranks] the load of each device
            device_counts: [num_ranks] the number of devices
        """

        device_num = len(origin_deployment)
        num_experts_per_device = origin_deployment.shape[1]

        device_assignments = [[-1] * num_experts_per_device
                              for _ in range(device_num)]
        device_weights = [[0.0] * num_experts_per_device
                          for _ in range(device_num)]
        device_loads = [0.0] * device_num
        device_counts = [0] * device_num

        rendun_pos_set = [set(positions) for positions in rendun_pos]

        for device_id in range(device_num):
            device_experts = origin_deployment[device_id]
            skip_positions = rendun_pos_set[device_id]

            for index in range(num_experts_per_device):
                if index in skip_positions:
                    continue

                expert_id = int(device_experts[index])
                cur_weight = updated_weights[expert_id]

                device_assignments[device_id][index] = expert_id
                device_weights[device_id][index] = cur_weight
                device_loads[device_id] += cur_weight
                device_counts[device_id] += 1

        return device_assignments, device_weights, device_loads, device_counts

    def recomputing_initial_weight(self, layer_workloads, device_assignments):
        """
        Recalculate the load based on
        the current expert on the card.

        Parameters:
            layer_workloads: [n], the weight of each item
            device_assignments: [m, k], the assignment of each device

        Returns:
            cur_layer_workload: [n], the weight of each item
            num_all_experts: [n], the number of each expert
        """

        num_all_experts = [0] * len(layer_workloads)
        for device in device_assignments:
            for expert_id in device:
                if expert_id != -1:
                    num_all_experts[expert_id] += 1

        cur_layer_workload = []
        for expert_id, weight in enumerate(layer_workloads):
            if num_all_experts[expert_id] == 0:
                cur_layer_workload.append(-1)
            else:
                cur_layer_workload.append(
                    self.safe_divide(weight, num_all_experts[expert_id]))

        return cur_layer_workload, num_all_experts

    def distribute_redun_experts(
        self, layer_workloads: np.ndarray, device_assignments: list,
        device_weights: list, device_loads: list, device_counts: list,
        redundant_expert_list: list, expert_from_device: np.ndarray,
        num_experts: int, rendun_pos: list
    ) -> tuple[list, list, list, list, list[dict[int, int]]]:
        """
        The position of the redundant expert before
        placing the redundant expert on the card.
        Place the currently hottest redundant expert
        onto the card with the remaining available
        slot and the least load.
        If there are still devices with empty slots
        that have not been filled with redundant
        experts, reselect appropriate redundant experts.

        Parameters:
            layer_workloads: [logic_experts_num], the weight of each item
            device_assignments: [num_ranks, num_experts_per_rank],
            the assignment of each device
            device_weights: [num_ranks, num_experts_per_rank],
            the load of each expert on the equipment
            device_loads: [num_ranks] the load of each device
            device_counts: [num_ranks] the number of devices
            redundant_expert_list: A list of redundant experts,
            ordered from highest to lowest load.
            expert_from_device:[logic_experts_num] The logic expert
            was on that device.
            num_experts: Number of Logic Experts
            rendun_pos:[num_ranks, p] Redundant positions on each device

        Returns:
            device_assignments: the assignment of each device
            device_weights: the load of each expert on the equipment
            device_loads: the load of each device
            device_counts: the number of devices
            com_between_devices:[num_ranks, num_ranks] Communication status
            between devices
        """

        num_devices = len(device_assignments)
        com_between_devices: list[dict[int,
                                       int]] = [{} for _ in range(num_devices)]

        for expert_id, weight in redundant_expert_list:
            candidate = -1
            for dev_id in range(num_devices):
                if len(rendun_pos[dev_id]) == 0:
                    continue
                if expert_id in device_assignments[dev_id]:
                    continue
                if (candidate == -1
                        or device_loads[dev_id] < device_loads[candidate]):
                    candidate = dev_id
            if candidate != -1:
                pos = rendun_pos[candidate].pop()
                device_assignments[candidate][pos] = expert_id
                device_weights[candidate][pos] = weight
                device_loads[candidate] += weight
                device_counts[candidate] += 1

                com_box_index = expert_from_device[expert_id]
                com_between_devices[candidate][com_box_index] = expert_id

        if any(sublist for sublist in rendun_pos):

            cur_layer_workload, num_exist_expert = (
                self.recomputing_initial_weight(layer_workloads,
                                                device_assignments))

            (update_workload, device_assignments, device_counts,
             com_between_devices) = self.repeat_compute_redundant_assignments(
                 cur_layer_workload, rendun_pos, num_experts, num_exist_expert,
                 device_assignments, device_counts, expert_from_device,
                 com_between_devices)

            device_loads = [0] * len(device_counts)
            for device_id, device in enumerate(device_assignments):
                for index, expert_id in enumerate(device):
                    device_weights[device_id][index] = (
                        update_workload[expert_id])
                    device_loads[device_id] += update_workload[expert_id]

        return (device_assignments, device_weights, device_loads,
                device_counts, com_between_devices)

    def redundancy_again(
        self,
        layer_workloads: np.ndarray,
        origin_weights: np.ndarray,
        origin_deployment: np.ndarray,
        expert_from_device: np.ndarray,
        rendun_pos: list,
        node_id: int = 0,
        per_node_route_expert_num: int = 0
    ) -> tuple[list, list, list, list, list[dict[int, int]]]:
        """
        Calculate the appropriate redundant expert
        based on the current load and assign them
        to the redundant expert position.

        Parameters:
            layer_workloads: [logic_experts_num], the weight of each item
            origin_weights: [logic_experts_num_per_node], The weight of each
            expert or the weight of each expert within each node
            origin_deployment: [num_ranks, num_physics experts_per_rank] The
            deployment status of experts on each device
            expert_from_device:[logic_experts_num] The logic expert was on
            that device
            rendun_pos: [num_ranks, p] Redundant positions on each device
            node_id: The index of node
            per_node_route_expert_num: The number of
            routing experts at each node

        Returns:
            device_assignments: [num_ranks, num_experts_per_rank],
            the assignment of each device
            device_weights: [num_ranks, num_experts_per_rank],
            the load of each expert on the equipment
            device_loads: [num_ranks] the load of each device
            device_counts: [num_ranks] the number of devices
            com_between_devices:[num_ranks] Communication status
            between devices
        """

        num_experts = len(layer_workloads)
        num_redundant_experts = 0
        for rank_empty_pos in rendun_pos:
            num_redundant_experts += len(rank_empty_pos)

        (redundancy_counts,
         updated_weights) = self.compute_redundant_assignments(
             origin_weights, num_redundant_experts, num_experts, node_id,
             per_node_route_expert_num)

        redundant_expert_list = self.prepare_expert_list(
            updated_weights, redundancy_counts, num_redundant_experts)

        (device_assignments, device_weights, device_loads,
         device_counts) = self.non_redundant_expert_information(
             origin_deployment, updated_weights, rendun_pos)

        (device_assignments, device_weights, device_loads, device_counts,
         com_between_devices) = self.distribute_redun_experts(
             layer_workloads, device_assignments, device_weights, device_loads,
             device_counts, redundant_expert_list, expert_from_device,
             num_experts, rendun_pos)

        return (device_assignments, device_weights, device_loads,
                device_counts, com_between_devices)

    @staticmethod
    def generate_allocation_report(device_assignments, device_weights,
                                   device_loads, device_counts):

        report = []
        max_load = 0.0

        for dev_id in range(len(device_assignments)):
            current_load = device_loads[dev_id]
            max_load = max(max_load, current_load)

            report.append({
                "device_id": dev_id + 1,
                "assigned_experts": device_assignments[dev_id],
                "expert_weights": device_weights[dev_id],
                "total_load": current_load,
                "expert_count": device_counts[dev_id]
            })

        return report, max_load

    @staticmethod
    def exchange_expert(cur_exchange_index, next_exchange_index, cur_device_id,
                        next_device_id, cur_layer_result, com_between_devices):

        cur_device_deployment = cur_layer_result[cur_device_id][
            'assigned_experts']
        next_device_deployment = cur_layer_result[next_device_id][
            'assigned_experts']

        cur_device_weight = cur_layer_result[cur_device_id]['expert_weights']
        next_device_weight = cur_layer_result[next_device_id]['expert_weights']

        cur_expert_id = cur_device_deployment[cur_exchange_index]
        next_expert_id = next_device_deployment[next_exchange_index]
        cur_device_deployment[cur_exchange_index] = next_expert_id
        next_device_deployment[next_exchange_index] = cur_expert_id

        cur_expert_weight = cur_device_weight[cur_exchange_index]
        next_expert_weight = next_device_weight[next_exchange_index]
        cur_device_weight[cur_exchange_index] = next_expert_weight
        next_device_weight[next_exchange_index] = cur_expert_weight

        cur_layer_result[cur_device_id]['total_load'] += (next_expert_weight -
                                                          cur_expert_weight)
        cur_layer_result[next_device_id]['total_load'] += (cur_expert_weight -
                                                           next_expert_weight)

        com_between_devices[cur_device_id][next_device_id] = next_expert_id
        com_between_devices[next_device_id][cur_device_id] = cur_expert_id

    def redundant_expert_deployment(
        self, layer_workloads: np.ndarray, original_deployment: np.ndarray,
        expert_from_device: np.ndarray, nodes_num: int,
        is_node_redundant: bool, rendun_pos: list
    ) -> tuple[list[dict[str, Any]], int, list[dict[int, int]]]:
        """
        Choose different methods based on whether
        it is node internal redundancy.

        Parameters:
            layer_workloads: [logic_experts_num], the weight of each item
            original_deployment: [num_ranks, num_physics experts_per_rank] The
            deployment status of experts on each device
            expert_from_device:[logic_experts_num] The logic expert
            was on that device.
            nodes_num: The number of nodes
            is_node_redundant: Intra-Node Redundancy
            rendun_pos: [num_ranks, p] Redundant positions on each device

        Returns:
            report: Expert information on the device
            max_load: The load of the hottest device
            com_between_devices: [num_ranks] Communication status
            between devices
        """

        device_num, per_device_expert_num = original_deployment.shape
        route_expert_num = layer_workloads.shape[0]
        per_node_device_num = self.safe_exact_divide(device_num, nodes_num)
        per_node_route_expert_num = self.safe_exact_divide(
            route_expert_num, nodes_num)

        if is_node_redundant:

            device_assignments = []
            device_weights = []
            device_loads = []
            device_counts = []
            com_between_devices = []

            for node_id in range(nodes_num):

                cur_node_weights = np.array(
                    layer_workloads[node_id *
                                    per_node_route_expert_num:(node_id + 1) *
                                    per_node_route_expert_num])
                cur_original_deployment = original_deployment[
                    node_id * per_node_device_num:(node_id + 1) *
                    per_node_device_num]
                cur_node_rendun_pos = rendun_pos[node_id *
                                                 per_node_device_num:(node_id +
                                                                      1) *
                                                 per_node_device_num]

                (cur_device_assignments, cur_device_weights, cur_device_loads,
                 cur_device_counts,
                 cur_com_between_devices) = self.redundancy_again(
                     layer_workloads, cur_node_weights,
                     cur_original_deployment, expert_from_device,
                     cur_node_rendun_pos, node_id, per_node_route_expert_num)

                device_assignments += cur_device_assignments
                device_weights += cur_device_weights
                device_loads += cur_device_loads
                device_counts += cur_device_counts
                com_between_devices += cur_com_between_devices

        else:
            cur_weights = np.array(layer_workloads)
            (device_assignments, device_weights, device_loads, device_counts,
             com_between_devices) = self.redundancy_again(
                 layer_workloads, cur_weights, original_deployment,
                 expert_from_device, rendun_pos)

        report, max_load = self.generate_allocation_report(
            device_assignments, device_weights, device_loads, device_counts)

        report, max_load = self.generate_allocation_report(
            device_assignments, device_weights, device_loads, device_counts)

        return report, max_load, com_between_devices

    @staticmethod
    def two_device_exchange_experts(cur_device_result: dict[str, Any],
                                    exchange_device_result: dict[str, Any],
                                    cur_exchanged_expert_id: list,
                                    next_exchanged_expert_id: list,
                                    ave_workload: float, increment: float):
        """
        Experts from both devices attempted to conduct an exchange.

        Parameters:
            cur_device_result: Information on the hottest device
            exchange_device_result: Device information selected for exchange
            cur_exchanged_expert_id:  Experts who have already
            conducted exchanges on the hottest device
            next_exchanged_expert_id: Experts who have already
            conducted exchanges on the target device
            ave_workload: Average load of all equipment
            increment: Switch Expert's Load Threshold

        Returns:
            best_cur_index: Expert index for selecting
            exchanges on the hottest devices.
            The value cannot be swapped. The value is - 1.
            best_next_index: Expert index for selecting
            exchanges on the target devices
            The value cannot be swapped. The value is - 1.
        """

        cur_weights = cur_device_result['expert_weights']
        next_weights = exchange_device_result['expert_weights']
        cur_expert_ids = cur_device_result['assigned_experts']
        next_expert_ids = exchange_device_result['assigned_experts']
        cur_total = cur_device_result['total_load']
        next_total = exchange_device_result['total_load']

        cur_expert_set = set(cur_expert_ids)
        next_expert_set = set(next_expert_ids)
        cur_exchanged_set = set(cur_exchanged_expert_id)
        next_exchanged_set = set(next_exchanged_expert_id)

        max_weight = max(cur_total, next_total)
        threshold = ave_workload * increment
        best_cur_index = -1
        best_next_index = -1
        best_max_weight = max_weight

        for cur_idx, (cur_weight,
                      cur_expert) in enumerate(zip(cur_weights,
                                                   cur_expert_ids)):
            if cur_expert in cur_exchanged_set:
                continue
            for next_idx, (next_weight, next_expert) in enumerate(
                    zip(next_weights, next_expert_ids)):
                if (next_expert in next_exchanged_set
                        or cur_expert in next_expert_set
                        or next_expert in cur_expert_set):
                    continue
                new_cur_load = cur_total - cur_weight + next_weight
                new_next_load = next_total - next_weight + cur_weight
                if new_cur_load >= max_weight or new_next_load >= max_weight:
                    continue
                new_max = max(new_cur_load, new_next_load)
                improvement = max_weight - new_max
                if new_max < best_max_weight and improvement >= threshold:
                    best_max_weight = new_max
                    best_cur_index = cur_idx
                    best_next_index = next_idx

        return best_cur_index, best_next_index

    def expert_exchange_between_devices(self,
                                        ave_workload: float,
                                        increment: float,
                                        cur_layer_result: list[dict[str, Any]],
                                        com_between_devices: list[dict[int,
                                                                       int]],
                                        node_idx: int = 0,
                                        per_node_device_num: int = 0,
                                        is_node_redundant: bool = False):
        """
        Each time, identify the hottest and the coldest devices,
        and iterate through the experts of both to attempt an exchange.
        If an exchange cannot be made, try the next coldest device,
        and so on, until a device that can be exchanged is found,
        and then proceed with the exchange. If no other devices
        can be exchanged with the hottest device, stop the exchange.

        Exchange Restrictions:
            1. Each pair of devices can only communicate once.
            2. No relay is allowed.
            3. The experts on the devices must not be the same.
            4. The profit after the exchange must exceed the threshold.

        Parameters:
            ave_workload: Average load of all equipment
            increment: Switch Expert's Load Threshold
            cur_layer_result:  Expert information on the device
            com_between_devices: Communication status
            between devices
            node_idx: The index of the node
            per_node_device_num: The number of devices within each node
            is_node_redundant: Intra-Node Redundancy

        """

        if is_node_redundant:
            cur_devices_result = cur_layer_result[node_idx *
                                                  per_node_device_num:
                                                  (node_idx + 1) *
                                                  per_node_device_num]
        else:
            cur_devices_result = cur_layer_result

        devices_total_weight = [(device['total_load'], device['device_id'] - 1)
                                for device in cur_devices_result]
        sorted_weights = sorted(devices_total_weight, key=lambda x: x[0])

        exchange_frequency = 100
        while exchange_frequency > 0:
            exchange_frequency -= 1
            exchange_occurred = False

            max_weight, max_device_id = sorted_weights[-1]

            for i in range(len(sorted_weights) - 1):
                min_weight, min_device_id = sorted_weights[i]
                if min_device_id not in com_between_devices[max_device_id]:

                    cur_exchange_ids = list(
                        com_between_devices[max_device_id].values())
                    next_exchange_ids = list(
                        com_between_devices[min_device_id].values())

                    cur_idx, next_idx = self.two_device_exchange_experts(
                        cur_layer_result[max_device_id],
                        cur_layer_result[min_device_id],
                        cur_exchange_ids,
                        next_exchange_ids,
                        ave_workload,
                        increment,
                    )

                    if cur_idx != -1:
                        self.exchange_expert(cur_idx, next_idx, max_device_id,
                                             min_device_id, cur_layer_result,
                                             com_between_devices)

                        new_max_load = cur_layer_result[max_device_id][
                            'total_load']
                        new_min_load = cur_layer_result[min_device_id][
                            'total_load']

                        # Update Load Sorting
                        del sorted_weights[-1]
                        del sorted_weights[i]

                        lo, hi = 0, len(sorted_weights)
                        while lo < hi:
                            mid = (lo + hi) // 2
                            if new_min_load < sorted_weights[mid][0]:
                                hi = mid
                            else:
                                lo = mid + 1
                        sorted_weights.insert(lo,
                                              (new_min_load, min_device_id))

                        lo, hi = 0, len(sorted_weights)
                        while lo < hi:
                            mid = (lo + hi) // 2
                            if new_max_load < sorted_weights[mid][0]:
                                hi = mid
                            else:
                                lo = mid + 1
                        sorted_weights.insert(lo,
                                              (new_max_load, max_device_id))

                        exchange_occurred = True
                        break

            if not exchange_occurred:
                break

    def exchange_experts(self, layer_result: list[dict[str, Any]],
                         layer_com_between_devices: list[dict[int, int]],
                         num_nodes: int, num_devices: int,
                         is_node_redundant: bool, ave_workload: float,
                         increment: float):
        """
        Select the corresponding switching expert
        method based on whether there is redundancy within the node.

        Parameters:
            layer_result:  Expert information on the device
            layer_com_between_devices: Communication status
            between devices
            num_nodes: The number of nodes
            num_devices: The number of devices
            is_node_redundant: Intra-Node Redundancy
            ave_workload: Average load of all equipment
            increment: Switch Expert's Load Threshold

        Returns:
            global_deployment: Current Layer Expert Deployment
            max_workload: The load of the hottest device
        """

        global_deployment = []

        if is_node_redundant:
            per_node_device_num = self.safe_exact_divide(
                num_devices, num_nodes)
            for node_idx in range(num_nodes):
                self.expert_exchange_between_devices(
                    ave_workload, increment, layer_result,
                    layer_com_between_devices, node_idx, per_node_device_num,
                    is_node_redundant)
        else:
            self.expert_exchange_between_devices(ave_workload, increment,
                                                 layer_result,
                                                 layer_com_between_devices)

        max_workload = 0
        for box in layer_result:
            global_deployment.append(box['assigned_experts'])
            if max_workload < box['total_load']:
                max_workload = box['total_load']

        global_deployment = np.array(global_deployment)

        return global_deployment, max_workload

    def count_elements(self, lst):
        count = 0
        for item in lst:
            if isinstance(item, list):
                count += self.count_elements(item)
            else:
                count += 1
        return count

    @staticmethod
    def constraint_expert_local_exchange(
            current_expert_table: np.ndarray,
            global_deployment: np.ndarray) -> np.ndarray:
        for layer_id in range(global_deployment.shape[0]):
            for card_id in range(global_deployment.shape[1]):
                current_list = [
                    int(x) for x in current_expert_table[layer_id][card_id]
                ]
                new_list = [
                    int(x) for x in global_deployment[layer_id][card_id]
                ]
                num = len(new_list)

                new_index = np.full(num, -1)
                new_result = np.full(num, -1)
                remaining_elements = []

                for i in range(num):
                    flag = True
                    for j in range(num):
                        if (new_list[i] == current_list[j]
                                and new_index[j] == -1):
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

    @staticmethod
    def gen_result(new_global_deployment: np.ndarray, layer_num: int,
                   local_expert_num: int):

        logical_replica_count = torch.zeros((layer_num, local_expert_num),
                                            dtype=torch.int32)
        for layer_id, layer in enumerate(new_global_deployment):
            for device in layer:
                for expert_id in device:
                    logical_replica_count[layer_id][expert_id] += 1

        max_expert_num = logical_replica_count.max()
        logical_to_physical_map = torch.full(
            (layer_num, local_expert_num, max_expert_num),
            -1,
            dtype=torch.int32)

        new_global_deployment = new_global_deployment.reshape(layer_num, -1)
        for layer_id, layer in enumerate(new_global_deployment):
            cur_expert_num = np.zeros(local_expert_num, dtype=int)
            for index, expert_id in enumerate(layer):
                logical_to_physical_map[layer_id][expert_id][
                    cur_expert_num[expert_id]] = index
                cur_expert_num[expert_id] += 1

        physical_to_logical_map = torch.from_numpy(new_global_deployment)

        return (physical_to_logical_map, logical_to_physical_map,
                logical_replica_count)

    def rebalance_experts(
        self,
        weight: torch.Tensor,
        num_replicas: int,
        num_groups: int,
        num_nodes: int,
        num_ranks: int,
        old_global_expert_indices: Optional[torch.Tensor] = None,
        is_node_redundant: bool = False,
        increment: float = 0.01,
        imbalance_threshold: float = 1.01,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Entry point for expert-parallelism load balancer.

        Parameters:
            old_global_expert_indices: [num_moe_layers, num_physical_experts],
            mapping from physical experts to logical experts.
            weight: [layers, num_logical_experts], the load statistics for all
                logical experts
            num_replicas: number of physical experts, must be a multiple of
                `num_gpus`
            num_groups: number of expert groups
            num_nodes: number of server nodes, where the intra-node network
                (e.g, NVLink) is faster
            num_ranks: number of GPUs, must be a multiple of `num_nodes`
            is_node_redundant: Check for intra-node redundancy
            increment: Switch Expert's Load Threshold
            imbalance_threshold: Adjust deployment if the imbalance
            degree exceeds this value.

        Returns:
            physical_to_logical_map: [layers, num_replicas], the expert
             index of each replica
            logical_to_physical_map: [layers, num_logical_experts, X],
             the replica indices for each expert
            expert_count: [layers, num_logical_experts], number of
            physical replicas for each logical expert
        """

        # Processing and analyzing data
        info = DynamicTable()
        info.workload_table = weight.numpy()
        if info.workload_table is None:
            raise ValueError(" workload_table cannot be None.")
        layer_num = info.workload_table.shape[0]
        info.placement_table = old_global_expert_indices.numpy().reshape(
            layer_num, num_ranks, -1)
        if info.placement_table is None:
            raise ValueError(" placement_table cannot be None.")
        expert_ids, counts = np.unique(info.placement_table[0],
                                       return_counts=True)
        num_redundancy_expert = self.get_redundant_num(counts)
        num_original_expert = len(expert_ids)
        layer_workloads = info.workload_table

        layer_num = layer_workloads.shape[0]
        expert_num = layer_workloads.shape[1]
        expert_from_device = np.zeros((layer_num, num_original_expert))

        if num_original_expert != expert_num:
            raise ValueError(f"The number of original experts"
                             f"({num_original_expert}) must match"
                             f"expert_num ({expert_num})")

        if num_ranks <= 0:
            raise ValueError("The number of ranks must be greater than 0")

        if num_ranks < num_redundancy_expert:
            raise ValueError(f"The number of ranks ({num_ranks}) must be"
                             f"greater than or equal to the number of"
                             f"redundant experts ({num_redundancy_expert})")

        global_deployment = np.zeros_like(info.placement_table)

        (layer_initial_imbalance,
         max_heat_per_layer_before) = self.calculate_initial_imbalance(
             info.placement_table, layer_workloads)

        max_heat_per_layer_after = np.zeros([layer_num])
        sum_num = 0

        # Calculate the new deployment for each layer
        for layer in range(layer_num):
            if layer_initial_imbalance[layer] < imbalance_threshold:
                global_deployment[layer] = info.placement_table[layer]
                continue

            ave_workload = self.safe_divide(np.sum(layer_workloads[layer]),
                                            num_ranks)

            # The position of the statistical logic expert
            # on the card and the location of the redundancy
            # expert on the card.
            rendun_pos: list[list[int]] = [[] for _ in range(num_ranks)]
            existing_experts = set()
            for device_id, device in enumerate(info.placement_table[layer]):
                for index, expert_id in enumerate(device):
                    if expert_id not in existing_experts:
                        existing_experts.add(expert_id)
                        expert_from_device[layer][expert_id] = device_id
                    else:
                        rendun_pos[device_id].append(index)

            (result, max_workload,
             com_between_devices) = self.redundant_expert_deployment(
                 layer_workloads[layer], info.placement_table[layer],
                 expert_from_device[layer], num_nodes, is_node_redundant,
                 rendun_pos)

            (global_deployment[layer],
             new_max_workload) = self.exchange_experts(result,
                                                       com_between_devices,
                                                       num_nodes, num_ranks,
                                                       is_node_redundant,
                                                       ave_workload, increment)

            for device_id in range(num_ranks):
                com_between_devices[device_id] = {
                    key: value
                    for key, value in com_between_devices[device_id].items()
                }
                sum_num += self.count_elements(com_between_devices[device_id])
            max_heat_per_layer_after[layer] = max(
                result, key=lambda x: x['total_load'])['total_load']

        layer_changed_ratio = []
        for layer_idx in range(layer_num):
            layer_changed_ratio.append(
                self.safe_divide(max_heat_per_layer_after[layer_idx],
                                 max_heat_per_layer_before[layer_idx]))

        # New deployment and old deployment aligned at
        # the same expert position on the same device.
        new_global_deployment = self.constraint_expert_local_exchange(
            info.placement_table, global_deployment)

        # Construct the output based on the
        # newly generated deployment.
        (physical_to_logical_map, logical_to_physical_map,
         logical_replica_count) = self.gen_result(new_global_deployment,
                                                  layer_num, expert_num)

        return (physical_to_logical_map, logical_to_physical_map,
                logical_replica_count)
