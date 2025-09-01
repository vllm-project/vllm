# Copyright Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
from abc import abstractmethod
from collections import defaultdict

import numpy as np


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


class DynamicEplbV2(EplbPolicy):

    def __init__(self, config: DynamicConfig):
        super().__init__(config)

    @staticmethod
    def safe_divide(a, b):
        if b == 0:
            print("Division by zero is not allowed")
            return 0
        return a / b

    @staticmethod
    def safe_exact_divide(a, b):
        if b == 0:
            print("Division by zero is not allowed")
            return 0
        return a // b

    @staticmethod
    def safe_mod(a, b):
        if b == 0:
            print("Division by zero is not allowed")
            return 0
        return a % b

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
    def get_redundant_num(npu_num, counts):
        redundant_num_each_npu: int = int(np.sum(counts - 1))
        return redundant_num_each_npu

    @staticmethod
    def calculate_max_heat_per_layer(workload_table, layer_num):
        max_heat_per_layer: list[float] = []
        for layer_idx in range(layer_num):
            npu_heats_now = np.sum(workload_table[layer_idx], axis=1)
            max_heat_per_layer.append(np.max(npu_heats_now))
        return max_heat_per_layer

    def calculate_initial_imbalance(self, global_deployment,
                                    new_layer_workloads):

        device_num = global_deployment.shape[1]
        layer_imbalance = []
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

            cur_layer_imbalance = self.safe_divide(
                cur_layer_max_workload,
                (self.safe_divide(total_workload, device_num)))
            layer_imbalance.append(cur_layer_imbalance)

        return layer_imbalance

    def compute_redundant_assignments(self, base_experts,
                                      num_redundant_experts, num_experts):

        redundant_assignments: list[list[int]] = [[]
                                                  for _ in range(num_experts)]
        current_weights = base_experts.copy()

        for i in range(num_redundant_experts):
            sorted_indices = np.argsort([w for _, w in current_weights],
                                        kind='stable')[::-1]
            sorted_weights = [current_weights[i] for i in sorted_indices]

            target_expert = sorted_weights[0]
            expert_id, original_weight = target_expert

            current_redundancy = len(redundant_assignments[expert_id])
            new_avg_weight = self.safe_divide(
                original_weight * (current_redundancy + 1),
                (current_redundancy + 2))

            redundant_assignments[expert_id].append(num_experts + i)
            current_weights[sorted_indices[0]] = (expert_id, new_avg_weight)

        sorted_indices = np.argsort([w for _, w in current_weights],
                                    kind='stable')[::-1]
        sorted_weights = [current_weights[i] for i in sorted_indices]

        return redundant_assignments, sorted_weights

    def repeat_compute_redundant_assignments(self, layer_workloads, rendun_pos,
                                             num_experts, num_exist_expert,
                                             device_assignments, device_counts,
                                             expert_from_device,
                                             com_between_devices):

        current_weights = np.zeros((num_experts, ), dtype='object')
        for expert_id, workload_weight in enumerate(layer_workloads):
            current_weights[expert_id] = (expert_id, workload_weight)

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
                    print("Error:Redundant expert failure re-occurred")
                    redundancy_successful = True
                    break
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
                        communication_box_index = expert_from_device[expert_id]
                        com_between_devices[cur_device_id][
                            communication_box_index] = expert_id
                        new_weight = self.safe_divide(
                            (original_weight * num_exist_expert[expert_id]),
                            (num_exist_expert[expert_id] + 1))
                        sorted_weights[index] = (expert_id, new_weight)
                        num_exist_expert[expert_id] += 1
                        redundancy_successful = True
                        break
                if redundancy_successful:
                    break

        sorted_indices = np.argsort([id for id, _ in sorted_weights],
                                    kind='stable')
        sorted_weights = [sorted_weights[i][1] for i in sorted_indices]

        return sorted_weights, device_assignments, device_counts, com_between_devices

    @staticmethod
    def prepare_expert_list(base_experts, redundant_assignments,
                            num_redundant_experts):
        redundant_expert_list = np.empty(num_redundant_experts, dtype=object)

        index = 0
        num_experts = len(redundant_assignments)
        for expert_id in range(num_experts):
            for _ in redundant_assignments[expert_id]:
                redundant_expert_list[index] = (expert_id,
                                                next(w
                                                     for eid, w in base_experts
                                                     if eid == expert_id))
                index += 1

        sorted_indices = np.argsort([w for _, w in redundant_expert_list],
                                    kind='stable')[::-1]
        return [redundant_expert_list[i] for i in sorted_indices]

    @staticmethod
    def non_redundant_expert_information(origin_deployment, updated_weights,
                                         rendun_pos):

        device_num = len(origin_deployment)
        num_experts_per_device = origin_deployment.shape[1]
        device_assignments = [[-1 for _ in range(num_experts_per_device)]
                              for _ in range(device_num)]
        device_weights = [[0 for _ in range(num_experts_per_device)]
                          for _ in range(device_num)]
        device_loads = [0] * device_num
        device_counts = [0] * device_num

        for device_id, device in enumerate(origin_deployment):
            for index, expert_id in enumerate(device):
                if index in rendun_pos[device_id]:
                    continue
                device_assignments[device_id][index] = expert_id
                cur_weight = next(
                    weight for expert_id_of_weight, weight in updated_weights
                    if expert_id_of_weight == expert_id)
                device_weights[device_id][index] = cur_weight
                device_loads[device_id] += cur_weight
                device_counts[device_id] += 1

        return device_assignments, device_weights, device_loads, device_counts

    def recomputing_initial_weight(self, layer_workloads, device_assignments):
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

    def distribute_redun_experts(self, layer_workloads, device_assignments,
                                 device_weights, device_loads, device_counts,
                                 redundant_expert_list, expert_from_device,
                                 num_experts, rendun_pos):

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
                if candidate == -1 or device_loads[dev_id] < device_loads[
                        candidate]:
                    candidate = dev_id
            if candidate != -1:
                pos = rendun_pos[candidate].pop()
                device_assignments[candidate][pos] = expert_id
                device_weights[candidate][pos] = weight
                device_loads[candidate] += weight
                device_counts[candidate] += 1

                communication_box_index = expert_from_device[expert_id]
                com_between_devices[candidate][
                    communication_box_index] = expert_id

        if any(sublist for sublist in rendun_pos):
            cur_layer_workload, num_exist_expert = self.recomputing_initial_weight(
                layer_workloads, device_assignments)

            update_workload, device_assignments, device_counts, com_between_devices = self.repeat_compute_redundant_assignments(
                cur_layer_workload, rendun_pos, num_experts, num_exist_expert,
                device_assignments, device_loads, expert_from_device,
                com_between_devices)

            device_loads = [0] * len(device_counts)
            for device_id, device in enumerate(device_assignments):
                for index, expert_id in enumerate(device):
                    device_weights[device_id][index] = update_workload[
                        expert_id]
                    device_loads[device_id] += update_workload[expert_id]

        return device_assignments, device_weights, device_loads, device_counts, com_between_devices

    def redundancy_again(self, layer_workloads, origin_weights,
                         origin_deployment, expert_from_device, num_node,
                         is_node_redundant, rendun_pos):

        num_experts = len(origin_weights)
        if is_node_redundant:
            num_experts = num_experts * num_node

        num_redundant_experts = 0
        for rank_empty_pos in rendun_pos:
            num_redundant_experts += len(rank_empty_pos)

        redundant_assignments, updated_weights = self.compute_redundant_assignments(
            origin_weights, num_redundant_experts, num_experts)

        redundant_expert_list = self.prepare_expert_list(
            updated_weights, redundant_assignments, num_redundant_experts)

        device_assignments, device_weights, device_loads, device_counts = self.non_redundant_expert_information(
            origin_deployment, updated_weights, rendun_pos)

        device_assignments, device_weights, device_loads, device_counts, com_between_devices = self.distribute_redun_experts(
            layer_workloads, device_assignments, device_weights, device_loads,
            device_counts, redundant_expert_list, expert_from_device,
            num_experts, rendun_pos)

        return device_assignments, device_weights, device_loads, device_counts, com_between_devices

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

        cur_layer_result[cur_device_id][
            'total_load'] += next_expert_weight - cur_expert_weight
        cur_layer_result[next_device_id][
            'total_load'] += cur_expert_weight - next_expert_weight

        com_between_devices[cur_device_id][next_device_id] = next_expert_id
        com_between_devices[next_device_id][cur_device_id] = cur_expert_id

    def redundant_expert_deployment(self, layer_workloads, original_deployment,
                                    expert_from_device, node_num,
                                    is_node_redundant, rendun_pos):
        device_num, per_device_expert_num = original_deployment.shape
        route_expert_num = layer_workloads.shape[0]
        per_node_device_num = self.safe_exact_divide(device_num, node_num)
        per_node_route_expert_num = per_node_device_num * (
            per_device_expert_num - 1)

        weights = np.zeros((route_expert_num, ), dtype='object')
        for expert_id, workload_weight in enumerate(layer_workloads):
            weights[expert_id] = (expert_id, workload_weight)

        if is_node_redundant:

            device_assignments = []
            device_weights = []
            device_loads = []
            device_counts = []
            com_between_devices = []

            for node_id in range(node_num):
                cur_node_weights = weights[node_id *
                                           per_node_route_expert_num:(node_id +
                                                                      1) *
                                           per_node_route_expert_num]
                cur_original_deployment = original_deployment[
                    node_id * per_node_device_num:(node_id + 1) *
                    per_node_device_num]

                cur_node_rendun_pos = rendun_pos[node_id *
                                                 per_node_device_num:(node_id +
                                                                      1) *
                                                 per_node_device_num]

                cur_device_assignments, cur_device_weights, cur_device_loads, cur_device_counts, cur_com_between_devices = self.redundancy_again(
                    layer_workloads, cur_node_weights, cur_original_deployment,
                    expert_from_device, node_num, is_node_redundant,
                    cur_node_rendun_pos)
                device_assignments += cur_device_assignments
                device_weights += cur_device_weights
                device_loads += cur_device_loads
                device_counts += cur_device_counts
                com_between_devices += cur_com_between_devices

        else:
            device_assignments, device_weights, device_loads, device_counts, com_between_devices = self.redundancy_again(
                layer_workloads, weights, original_deployment,
                expert_from_device, node_num, is_node_redundant, rendun_pos)
        report, max_load = self.generate_allocation_report(
            device_assignments, device_weights, device_loads, device_counts)

        return report, max_load, com_between_devices

    @staticmethod
    def two_device_exchange_experts(cur_device_result, exchange_device_result,
                                    cur_exchanged_expert_id,
                                    next_exchanged_expert_id, ave_workload,
                                    increment, num_redundancy_expert):

        cur_device_weight = cur_device_result['expert_weights']
        next_device_weight = exchange_device_result['expert_weights']

        cur_device_expert_id = cur_device_result['assigned_experts']
        next_device_expert_id = exchange_device_result['assigned_experts']

        cur_device_total_weight = cur_device_result['total_load']
        next_device_total_weight = exchange_device_result['total_load']
        max_weight = max(cur_device_total_weight, next_device_total_weight)

        cur_exchange_index = -1
        next_exchange_index = -1

        for index, weight in enumerate(cur_device_weight):
            for next_index, next_weight in enumerate(next_device_weight):
                change_flag = True
                if (cur_device_expert_id[index] in next_device_expert_id
                        or next_device_expert_id[next_index]
                        in cur_device_expert_id):
                    change_flag = False
                if (cur_device_expert_id[index] not in cur_exchanged_expert_id
                    ) and (next_device_expert_id[next_index]
                           not in next_exchanged_expert_id) and change_flag:

                    cur_total_weight_after_exchange = cur_device_total_weight - weight + next_weight
                    next_total_weight_after_exchange = next_device_total_weight - next_weight + weight
                    exchange_max_weight = max(
                        cur_total_weight_after_exchange,
                        next_total_weight_after_exchange)
                    if exchange_max_weight < max_weight and (
                            max_weight -
                            exchange_max_weight) >= (ave_workload * increment):
                        max_weight = exchange_max_weight
                        cur_exchange_index = index
                        next_exchange_index = next_index

        return cur_exchange_index, next_exchange_index

    def expert_exchange_between_devices(self,
                                        ave_workload,
                                        increment,
                                        cur_layer_result,
                                        com_between_devices,
                                        num_redundancy_expert,
                                        node_idx=0,
                                        per_node_device_num=0,
                                        is_node_redundant=False):

        if is_node_redundant:
            cur_devices_result = cur_layer_result[node_idx *
                                                  per_node_device_num:
                                                  (node_idx + 1) *
                                                  per_node_device_num]
        else:
            cur_devices_result = cur_layer_result

        devices_total_weight = []
        for device in cur_devices_result:
            devices_total_weight.append(
                (device['total_load'], device['device_id'] - 1))

        exchange_frequency = 100
        while exchange_frequency > 0:
            exchange_frequency -= 1
            devices_total_weight.sort(key=lambda x: x[0])
            max_weight_device_id = devices_total_weight[-1][1]
            exchange = False
            for index in range(0, len(devices_total_weight) - 1):
                min_weight_device_id = devices_total_weight[index][1]
                if min_weight_device_id not in com_between_devices[
                        max_weight_device_id]:
                    cur_exchanged_expert_id = list(
                        com_between_devices[max_weight_device_id].values())
                    next_exchanged_expert_id = list(
                        com_between_devices[min_weight_device_id].values())

                    cur_exchange_index, next_exchange_index = self.two_device_exchange_experts(
                        cur_layer_result[max_weight_device_id],
                        cur_layer_result[min_weight_device_id],
                        cur_exchanged_expert_id, next_exchanged_expert_id,
                        ave_workload, increment, num_redundancy_expert)

                    if cur_exchange_index != -1:
                        self.exchange_expert(cur_exchange_index,
                                             next_exchange_index,
                                             max_weight_device_id,
                                             min_weight_device_id,
                                             cur_layer_result,
                                             com_between_devices)

                        devices_total_weight[-1] = (
                            cur_layer_result[max_weight_device_id]
                            ['total_load'], max_weight_device_id)
                        devices_total_weight[index] = (
                            cur_layer_result[min_weight_device_id]
                            ['total_load'], min_weight_device_id)
                        exchange = True
                        break

            if not exchange:
                break

    def exchange_experts(self, layer_result, layer_com_between_devices,
                         num_nodes, device_num, is_node_redundant,
                         ave_workload, increment, num_redundancy_expert,
                         org_deployment):

        global_deployment = []

        if is_node_redundant:
            per_node_device_num = self.safe_exact_divide(device_num, num_nodes)
            for node_idx in range(num_nodes):
                self.expert_exchange_between_devices(
                    ave_workload, increment, layer_result,
                    layer_com_between_devices, num_redundancy_expert, node_idx,
                    per_node_device_num, is_node_redundant)
        else:
            self.expert_exchange_between_devices(ave_workload, increment,
                                                 layer_result,
                                                 layer_com_between_devices,
                                                 num_redundancy_expert)

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

    def rebalance_experts(self,
                          current_expert_table,
                          expert_workload,
                          is_node_redundant=False,
                          increment=0.01):
        info = DynamicTable()
        info.workload_table = expert_workload.numpy()
        info.placement_table = current_expert_table.numpy()
        assert info.workload_table is not None
        layer_num, num_npus, experts_per_npu = info.workload_table.shape
        expert_ids, counts = np.unique(info.placement_table[0],
                                       return_counts=True)
        num_redundancy_expert = self.get_redundant_num(num_npus, counts)
        num_original_expert = len(expert_ids)
        layer_workloads = self.add_redundant(info.placement_table,
                                             info.workload_table,
                                             num_original_expert)
        max_heat_per_layer_before = self.calculate_max_heat_per_layer(
            info.workload_table, layer_num)
        npu_heat_all_origin = sum(max_heat_per_layer_before)

        num_node = self.safe_exact_divide(num_npus, 8)
        layer_num = layer_workloads.shape[0]
        expert_num = layer_workloads.shape[1]
        expert_from_device = np.zeros((layer_num, num_original_expert))

        if num_original_expert != expert_num:
            raise ValueError(
                f"The number of original experts ({num_original_expert}) must match expert_num ({expert_num})"
            )

        if num_npus <= 0:
            raise ValueError("The number of NPUs must be greater than 0")

        if num_npus < num_redundancy_expert:
            raise ValueError(
                f"The number of NPUs ({num_npus}) must be greater than or equal to the number of redundant experts ({num_redundancy_expert})"
            )

        global_deployment: list[list[list[int]]] = [[[]
                                                     for _ in range(num_npus)]
                                                    for _ in range(layer_num)]
        layer_initial_imbalance = self.calculate_initial_imbalance(
            info.placement_table, layer_workloads)
        max_heat_per_layer_after = np.zeros([layer_num])
        sum_num = 0
        for layer in range(layer_num):
            # print(f"Load imbalance ratio of layer {layer} under the new workload", layer_initial_imbalance[layer])
            if layer_initial_imbalance[layer] < 1.01:
                global_deployment[layer] = info.placement_table[layer]
                continue

            ave_workload = self.safe_divide(np.sum(layer_workloads[layer]),
                                            num_npus)

            rendun_pos: list[list[int]] = [[] for _ in range(num_npus)]
            existing_experts = set()
            for device_id, device in enumerate(info.placement_table[layer]):
                for index, expert_id in enumerate(device):
                    if expert_id not in existing_experts:
                        existing_experts.add(expert_id)
                        expert_from_device[layer][expert_id] = device_id
                    else:
                        rendun_pos[device_id].append(index)

            result, max_workload, com_between_devices = self.redundant_expert_deployment(
                layer_workloads[layer], info.placement_table[layer],
                expert_from_device[layer], num_node, is_node_redundant,
                rendun_pos)
            # print(layer, f"Imbalance Ratio after Redundancy Adjustment:", self.safe_divide(max_workload, ave_workload))

            global_deployment[layer], new_max_workload = self.exchange_experts(
                result, com_between_devices, num_node, num_npus,
                is_node_redundant, ave_workload, increment,
                num_redundancy_expert, info.placement_table[layer])
            # print(layer, f"Imbalance Ratio after Swap Adjustment:", self.safe_divide(new_max_workload, ave_workload))

            for device_id in range(num_npus):
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

        per_layer_priority = np.argsort(layer_changed_ratio)
        npu_heat_all_after = sum(max_heat_per_layer_after)

        change = 0
        if npu_heat_all_after < 0.95 * npu_heat_all_origin:
            change = 1

        new_global_deployment = self.constraint_expert_local_exchange(
            current_expert_table, global_deployment)

        return change, per_layer_priority, np.array(
            new_global_deployment).tolist()
