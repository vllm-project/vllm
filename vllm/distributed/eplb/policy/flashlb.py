# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from collections import defaultdict, deque

import numpy as np
import torch
from numba import njit

from .abstract import AbstractEplbPolicy

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)


@njit
def compute_piece_counts(X, P, stage_weights):
    '''
    Greedy iterative expert partitioning strategy to calculate the optimal
    number of replicas (pieces) for each expert.

    Parameters:
        X (np.ndarray): Multi-stage expert hotness matrix with shape
            (n_stage, num_expert),
        P (int): Total number of replicas
        stage_weights (np.ndarray): Multi-stage hotness weight array with
            shape (n_stage,).
    Returns:
        pieces (np.ndarray): Optimal expert partitioning scheme with shape
            (num_expert,).
    '''
    n_stage, N = X.shape
    S = P - N
    pieces = np.ones(N, dtype=np.int32)
    unit = X / pieces  # unit[i, j] = X[i, j] / pieces[j]

    for _ in range(S):
        deltas = np.zeros(N, dtype=np.float32)
        for i in range(n_stage):
            idx1 = -1
            idx2 = -1
            val1 = -1.0
            val2 = -1.0
            for j in range(N):
                v = unit[i, j]
                if v > val1:
                    val2 = val1
                    idx2 = idx1
                    val1 = v
                    idx1 = j
                elif v >= val2:
                    val2 = v
                    idx2 = j
            origin = unit[i, idx1]
            secv = unit[i, idx2]
            alt = X[i, idx1] / (pieces[idx1] + 1)
            delta = origin - (alt if alt > secv else secv)
            deltas[idx1] += (delta * stage_weights[i]
                             if np.any(delta) != 0 else stage_weights[i])

        max_idx = np.argmax(deltas)
        pieces[max_idx] += 1

        for i in range(n_stage):
            unit[i, max_idx] = X[i, max_idx] / pieces[max_idx]

    return pieces


@njit
def lpt_placement(X, pieces, M, stage_weights):
    '''
    A LPT (Longest Process Time First)-based expert deployment strategy
    function, designed to map expert replicas to target devices optimally.

    Parameters:
        X (np.ndarray): Multi-stage expert hotness matrix with shape
            (n_stage, num_expert),
        pieces (np.ndarray): Optimal expert partitioning scheme with shape
            (num_expert,).
        M (int): Number of devices
        stage_weights (np.ndarray): Multi-stage hotness weight array with
            shape (n_stage,).
    Returns:
        deployment (np.ndarray): Optimal expert deployment matrix with shape
            (M, num_group)
    '''

    n_stage, N = X.shape
    total_piece = pieces.sum()
    num_per_group = total_piece // M
    unit_hotness = np.empty((n_stage, N), dtype=np.float32)

    for i in range(N):
        for s in range(n_stage):
            unit_hotness[s, i] = X[s, i] / pieces[i]

    scores = np.zeros(N, dtype=np.float32)

    for i in range(N):
        for s in range(n_stage):
            scores[i] += unit_hotness[s, i]

    idx = np.argsort(-scores)
    loads = np.zeros((n_stage, M), dtype=np.float32)
    dev_phy_exp_n = np.zeros(M, dtype=np.int32)
    deployment = -np.ones((M, num_per_group), dtype=np.int32)
    dep_ptr = np.zeros(M, dtype=np.int32)

    for t in range(N):
        i = idx[t]
        used_device = list()
        for _ in range(pieces[i]):
            w = np.empty(n_stage, dtype=np.float32)
            for s in range(n_stage):
                w[s] = unit_hotness[s, i]
            stage_max = np.empty(n_stage, dtype=np.float32)

            for s in range(n_stage):
                max_val = loads[s, 0]
                for k in range(1, M):
                    if loads[s, k] > max_val:
                        max_val = loads[s, k]
                stage_max[s] = max_val
            denom = np.empty(n_stage, dtype=np.float32)
            for s in range(n_stage):
                sum_tmp = 0.0
                for j in range(M):
                    sum_tmp += loads[s, j] + w[s]
                denom[s] = sum_tmp / M + 1e-2
            best_j = -1
            best_val = 1e30
            for j in range(M):
                if dev_phy_exp_n[j] >= num_per_group:
                    continue
                if j in used_device:
                    continue
                score = 0.0
                for s in range(n_stage):
                    tmp_sj = loads[s, j] + w[s]
                    numer_sj = tmp_sj if tmp_sj > stage_max[s] else stage_max[s]
                    score += stage_weights[s] * (numer_sj / denom[s])
                if score < best_val:
                    best_val = score
                    best_j = j
            if best_j == -1:
                continue
            used_device.append(best_j)

            for s in range(n_stage):
                loads[s, best_j] += w[s]
            ptr = dep_ptr[best_j]
            deployment[best_j, ptr] = i
            dep_ptr[best_j] += 1
            dev_phy_exp_n[best_j] += 1
    for rank in range(M):
        for col in range(num_per_group):
            if deployment[rank, col] == -1:
                current_rank_elements = deployment[rank, :]
                available = []
                for x in range(N):
                    is_in = False
                    for val in current_rank_elements:
                        if x == val:
                            is_in = True
                            break
                    if not is_in:
                        available.append(x)
                if available:
                    rand_idx = np.random.randint(0, len(available))
                    deployment[rank, col] = available[rand_idx]
    return deployment


@njit
def slice_values(X, pieces):
    total_len = 0
    for i in range(X.shape[0]):
        total_len += pieces[i]
    result = np.empty(total_len, dtype=np.float32)
    idx = 0
    for i in range(X.shape[0]):
        val = X[i] / pieces[i]
        for _ in range(pieces[i]):
            result[idx] = val
            idx += 1
    return result


@njit
def group_based_adaptive_searching_kernel(X, P, M, simulated_pieces,
                                          simulated_deployment, stage_weights):
    """
    Group-based adaptive searching kernel function for calculating the optimal
    expert partitioning strategy.

    Parameters:
        X (np.ndarray): Multi-stage expert hotness matrix with shape
            (n_stage, num_expert),
        P (int): Number of expert replicas
        M (int): Number of devices
        simulated_pieces (np.ndarray): Historically predicted optimal expert
            partitioning scheme with shape (num_expert,).
        simulated_deployment (np.ndarray): Historically predicted optimal
            expert deployment scheme, typically with shape (M, num_group)
            (where num_group = P//M).
        stage_weights (np.ndarray): Multi-stage hotness weight array with
            shape (n_stage,).
    Returns:
        pieces (np.ndarray): Optimal expert partitioning scheme with shape
            (num_expert,).
    """

    n_stage, N = X.shape
    num_group = P // M

    X_all = np.zeros(N, dtype=np.float32)

    for i in range(n_stage):
        for j in range(N):
            X_all[j] += X[i, j]

    sort_idx = np.argsort(np.negative(X_all))
    X_sorted = X[:, sort_idx]
    unit_load = np.empty(N, dtype=np.float32)

    for j in range(N):
        unit_load[j] = X_all[j] / simulated_pieces[j]

    flat_deployment = simulated_deployment.reshape(-1)
    simulated_load = np.zeros(M, dtype=np.float32)

    for i in range(flat_deployment.shape[0]):
        simulated_load[i // (flat_deployment.shape[0] //
                             M)] += unit_load[flat_deployment[i]]

    slice_vals = slice_values(X_all, simulated_pieces)
    sorted_slices = np.sort(slice_vals)[::-1]
    simulated_slopes = (sorted_slices[:-M + 1] - sorted_slices[M - 1:]) / M
    cumulative_slices_used = np.zeros(N, dtype=np.int32)

    acc = 0
    for i in range(N):
        acc += simulated_pieces[sort_idx[i]]
        cumulative_slices_used[i] = acc
    group_boundary_indices = np.empty(num_group, dtype=np.int32)

    for i in range(1, num_group + 1):
        for j in range(N):
            if cumulative_slices_used[j] >= i * M:
                group_boundary_indices[i - 1] = j
                break

    slices_used_per_group = np.empty(num_group, dtype=np.int32)
    slices_used_per_group[0] = group_boundary_indices[0]
    for i in range(1, num_group):
        slices_used_per_group[i] = (group_boundary_indices[i] -
                                    group_boundary_indices[i - 1])
    slices_used_per_group = M - slices_used_per_group

    loads = np.zeros(M, dtype=np.float32)
    pieces = np.zeros(N, dtype=np.int32)
    num_remain_slice = P - N
    current_idx = 0
    for g in range(num_group):
        window = X_sorted[:, current_idx:current_idx + 2 * M]
        low = max(0, current_idx + M - N)
        high = min(num_remain_slice, M - 1)
        while high - low > 1:
            mid = (high + low) // 2
            keep = M - mid
            current_group = window[:, :keep]
            current_pieces = compute_piece_counts(current_group, M,
                                                  stage_weights)
            current_slice = slice_values(current_group.sum(0), current_pieces)
            current_slice_sorted = np.sort(current_slice)
            current_loads = loads + current_slice_sorted
            current_slope = (np.max(current_loads) - np.min(current_loads)) / M
            next_slope = np.max(simulated_slopes[current_idx + keep:])

            if abs(current_slope) > abs(next_slope):
                low = mid
            else:
                high = mid
        S = high
        keep = M - S
        current_group = window[:, :keep]
        current_pieces = compute_piece_counts(current_group, M, stage_weights)

        for i in range(keep):
            pieces[sort_idx[current_idx + i]] = current_pieces[i]

        current_slice = slice_values(current_group.sum(0), current_pieces)
        current_slice_sorted = np.sort(current_slice)
        loads += current_slice_sorted
        loads = np.sort(loads)[::-1]
        current_idx += keep
        num_remain_slice -= S
    return pieces


@njit
def auto_fix_new_placement(old_placement, new_placement):
    """
    Adjust the new_placement matrix to ensure elements (including duplicates)
    that exist in both old_placement and new_placement remain in their original
    positions from old_placement. New elements (unique to new_placement) will
    fill the remaining empty positions.

    Parameters:
        old_placement: Old deployment matrix with shape
            (num_ranks, num_experts)
        new_placement: New deployment matrix to be fixed, must have the same
            shape as old_placement
    Returns:
        fixed_new: adjusted version of the new_placement matrix
    """

    num_ranks, num_experts = old_placement.shape
    fixed_new = np.empty_like(new_placement)
    max_expert_old = old_placement.max() if num_experts > 0 else 0
    max_expert_new = new_placement.max() if num_experts > 0 else 0
    max_expert = max(max_expert_old, max_expert_new)

    for rank_id in range(num_ranks):
        old_row = old_placement[rank_id]
        new_row = new_placement[rank_id]
        index_array = np.full((max_expert + 1, num_experts),
                              -1,
                              dtype=np.int32)
        count_array = np.zeros(max_expert + 1, dtype=np.int32)
        for idx in range(num_experts):
            val = old_row[idx]
            if val >= 0 and val <= max_expert:
                pos = count_array[val]
                index_array[val, pos] = idx
                count_array[val] += 1
        old_counter = np.zeros(max_expert + 1, dtype=np.int32)
        for idx in range(num_experts):
            val = old_row[idx]
            if val >= 0 and val <= max_expert:
                old_counter[val] += 1
        retain_elements = np.empty(num_experts, dtype=new_placement.dtype)
        new_elements = np.empty(num_experts, dtype=new_placement.dtype)
        retain_ptr = 0
        new_ptr = 0

        for val in new_row:
            if val >= 0 and val <= max_expert and old_counter[val] > 0:
                retain_elements[retain_ptr] = val
                retain_ptr += 1
                old_counter[val] -= 1
            else:
                new_elements[new_ptr] = val
                new_ptr += 1
        current_fixed = np.full(num_experts, -1, dtype=new_placement.dtype)
        for i in range(retain_ptr):
            val = retain_elements[i]
            if val >= 0 and val <= max_expert:
                pos = count_array[val] - 1
                if pos >= 0:
                    idx = index_array[val, pos]
                    current_fixed[idx] = val
                    count_array[val] -= 1
        empty_indices = np.empty(num_experts, dtype=np.int32)
        empty_ptr = 0
        for idx in range(num_experts):
            if current_fixed[idx] == -1:
                empty_indices[empty_ptr] = idx
                empty_ptr += 1
        for i in range(new_ptr):
            if i < empty_ptr:
                current_fixed[empty_indices[i]] = new_elements[i]
        fixed_new[rank_id] = current_fixed
    return fixed_new


@njit
def compute_objective(deployment, X, pieces):
    M, P = deployment.shape
    loads = np.zeros(M)
    for i in range(M):
        for j in range(P):
            expert = deployment[i, j]
            if pieces[expert] == 0:
                continue
            loads[i] += X[expert] / pieces[expert]

    mean_load = np.mean(loads)
    max_load = np.max(loads)
    obj = max_load / mean_load
    return obj, loads


@njit
def compute_logical_to_physical_map(phy2log, num_layers, num_expert,
                                    num_replicas, maxlogcnt):
    log2phy = -1 * np.ones((num_layers, num_expert, maxlogcnt), dtype=np.int64)
    for layer in range(num_layers):
        filled_counts = np.zeros(num_expert, dtype=np.int64)
        for p in range(num_replicas):
            e = phy2log[layer, p]
            rank = filled_counts[e]
            log2phy[layer, e, rank] = p
            filled_counts[e] += 1
    return log2phy


class FlashlbEplbPolicy(AbstractEplbPolicy):

    def __init__(self):
        self.max_stage_window = 32
        self.buffer_expert_layer_num = 8
        self.threshold_ratio = 1.15
        self.par_history: dict[int, float] = {}
        self.hotness_window: dict[int, deque[float]] = {}

    def compute_expert_hotness(self, num_of_expert: int,
                               deployment: np.ndarray, rank_load: np.ndarray):
        hotness = np.zeros(num_of_expert, dtype=rank_load.dtype)
        deployment_flat = deployment.ravel()
        rank_load_flat = rank_load.ravel()
        np.add.at(hotness, deployment_flat, rank_load_flat)
        return hotness

    def compute_rank_load(self, deployment: np.ndarray, hotness: np.ndarray):
        n_stage, N = hotness.shape
        unit_hotness = hotness / np.bincount(deployment.reshape(-1))
        stage_par = np.zeros(n_stage)
        for i in range(n_stage):
            stage_load = unit_hotness[i][deployment].sum(-1)
            stage_par[i] = stage_load.max() / stage_load.mean()
        return stage_par.mean()

    def group_based_adaptive_searching(self,
                                       X,
                                       P,
                                       M,
                                       stage_weights=None,
                                       recorsive=False):
        n_stage, N = X.shape
        if stage_weights is None:
            stage_weights = np.ones(n_stage, dtype=np.float32)
        if recorsive:
            simulated_deployment, simulated_pieces = (
                self.group_based_adaptive_searching(X,
                                                    P,
                                                    M,
                                                    stage_weights,
                                                    recorsive=False))
        else:
            simulated_pieces = compute_piece_counts(X, P, stage_weights)
            simulated_deployment = lpt_placement(X, simulated_pieces, M,
                                                 stage_weights)
        if M < 2:
            return simulated_deployment, simulated_pieces
        pieces = group_based_adaptive_searching_kernel(
            X.astype(np.float32),
            P,
            M,
            simulated_pieces.astype(np.int32),
            simulated_deployment.astype(np.int32),
            stage_weights.astype(np.float32),
        )
        deployment = lpt_placement(X, pieces, M, stage_weights)
        X_all = X.sum(0)
        unit_load = X_all / pieces
        load = unit_load[deployment].sum(-1)
        sim_unit_load = X_all / simulated_pieces
        sim_load = sim_unit_load[simulated_deployment].sum(-1)
        if load.max() > sim_load.max():
            return simulated_deployment, simulated_pieces
        return deployment, pieces

    def need_update(self, current_par, layer_id=0):
        threshold = self.par_history.get(layer_id, 0.0)
        return current_par >= self.threshold_ratio * threshold

    def compute_stage_weight(self, hotness):
        n_stage = hotness.shape[0]
        stage_weights = np.zeros(n_stage)
        for i in range(n_stage):
            stage_weights[i] = hotness[i].sum()
        stage_weights = stage_weights / stage_weights.max()
        return stage_weights

    def rebalance_layer(self,
                        num_replicas,
                        num_rank,
                        deployment,
                        hotness,
                        layer_id=0):
        current_par = self.compute_rank_load(deployment, hotness)
        if not self.need_update(current_par, layer_id):
            pieces = np.bincount(deployment.ravel())
            return deployment, pieces, current_par, current_par

        stage_weights = self.compute_stage_weight(hotness)
        new_deployment, pieces = self.group_based_adaptive_searching(
            hotness, num_replicas, num_rank, stage_weights, recorsive=False)
        assert not np.any(new_deployment < 0), (
            f"Deployment contains {np.sum(new_deployment < 0)} negative "
            "values (invalid empty places)")
        new_par = self.compute_rank_load(new_deployment, hotness)
        return new_deployment, pieces, new_par, current_par

    def register_hotness(self, num_layer, hotness):
        for layer in range(num_layer):
            if layer not in self.hotness_window:
                self.hotness_window[layer] = deque(
                    maxlen=self.max_stage_window)
            self.hotness_window[layer].append(hotness[layer])

    def compress_by_avg_pooling_fast_nd(self, arr, m):
        n, d = arr.shape
        idx = (np.arange(n) * m // n)
        result = np.zeros((m, d))
        counts = np.zeros((m, 1))
        np.add.at(result, idx, arr)
        np.add.at(counts, idx, 1)
        return result / counts

    def rebalance_experts(
        self, weight: torch.Tensor, num_replicas: int, num_groups: int,
        num_nodes: int, num_ranks: int, old_global_expert_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Entry point for expert-parallelism load balancer.
        Parameters:
            old_global_expert_indices: [num_moe_layers, num_physical_experts],
                mapping from physical experts to logical experts.
            weight: [layers, num_logical_experts], the load statistics
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
        assert num_ranks % num_nodes == 0
        assert num_replicas % num_ranks == 0

        expert_workload = weight.cpu().numpy()
        expert_workload += 1
        num_layer, num_expert = expert_workload.shape
        if old_global_expert_indices is not None:
            current_deployment = old_global_expert_indices.cpu().numpy()
            new_deployment = current_deployment.copy()
        else:
            new_deployment = np.zeros(
                (num_layer, num_ranks, num_replicas // num_ranks))

        self.register_hotness(num_layer, expert_workload)
        layers_need_update = np.arange(num_layer)
        new_par = np.zeros(layers_need_update.shape[0])
        current_par = np.ones(layers_need_update.shape[0])
        expert_count = np.zeros((num_layer, num_expert))

        for i, layer in enumerate(layers_need_update):
            hotness = np.array(self.hotness_window[layer])
            if hotness.shape[0] > self.max_stage_window:
                hotness = self.compress_by_avg_pooling_fast_nd(
                    hotness, self.max_stage_window)
            if old_global_expert_indices is None:
                stage_weights = self.compute_stage_weight(hotness)
                new_deployment[layer], expert_count[layer] = (
                    self.group_based_adaptive_searching(hotness,
                                                        num_replicas,
                                                        num_ranks,
                                                        stage_weights,
                                                        recorsive=False))
            else:
                (new_deployment[layer], expert_count[layer], new_par[i],
                 current_par[i]) = self.rebalance_layer(
                     num_replicas,
                     num_ranks,
                     current_deployment[layer],
                     hotness,
                     layer_id=layer)

        priority = new_par / current_par
        priority_idx = np.argsort(priority)
        priority_idx = priority_idx[priority[priority_idx] <
                                    1][:self.buffer_expert_layer_num]

        change = len(priority_idx) > 0
        if np.all(expert_workload == 1):
            for _, layer in enumerate(layers_need_update):
                self.hotness_window[layer].pop()
            change = False

        if old_global_expert_indices is None:
            physical_to_logical_map = np.array(new_deployment,
                                               dtype=np.int32).reshape(
                                                   (num_layer, num_replicas))
            maxlogcnt = int(expert_count.max())
            logical_to_physical_map = compute_logical_to_physical_map(
                physical_to_logical_map, num_layer, num_expert, num_replicas,
                maxlogcnt)
            physical_to_logical_map = torch.tensor(physical_to_logical_map,
                                                   dtype=torch.int32,
                                                   device=weight.device)
            logical_to_physical_map = torch.tensor(logical_to_physical_map,
                                                   dtype=torch.int32,
                                                   device=weight.device)
            expert_count = torch.tensor(expert_count,
                                        dtype=torch.int32,
                                        device=weight.device)
            return (physical_to_logical_map, logical_to_physical_map,
                    expert_count)

        deployment = current_deployment.copy()

        if change:
            for idx in priority_idx:
                deployment[idx] = auto_fix_new_placement(
                    current_deployment[idx], new_deployment[idx])
                self.par_history[idx] = new_par[idx]
        physical_to_logical_ndarray = deployment.reshape(
            (num_layer, num_replicas)).astype(np.int32)
        maxlogcnt = int(expert_count.max())
        logical_to_physical_ndarray = compute_logical_to_physical_map(
            physical_to_logical_ndarray, num_layer, num_expert, num_replicas,
            maxlogcnt)

        physical_to_logical_map = torch.tensor(physical_to_logical_ndarray,
                                               dtype=torch.int32,
                                               device=weight.device)
        logical_to_physical_map = torch.tensor(logical_to_physical_ndarray,
                                               dtype=torch.int32,
                                               device=weight.device)
        expert_count_tensor = torch.tensor(expert_count,
                                           dtype=torch.int32,
                                           device=weight.device)
        return (physical_to_logical_map, logical_to_physical_map,
                expert_count_tensor)


def warmup_flashlb():
    algo = FlashlbEplbPolicy()

    def generate_layered_experts(num_layers=58,
                                 layer_shape=(32, 9),
                                 expert_min=0,
                                 expert_max=255):
        expert_num = expert_max - expert_min + 1
        layer_total = layer_shape[0] * layer_shape[1]
        extra_slots = layer_total - expert_num

        assert layer_total >= expert_num, (
            f"Layer elements {layer_total} < experts {expert_num}")

        layers = []
        for _ in range(num_layers):
            full_experts = torch.arange(expert_min,
                                        expert_max + 1,
                                        dtype=torch.int64)
            extra_experts = torch.randint(expert_min,
                                          expert_max + 1,
                                          size=(extra_slots, ),
                                          dtype=torch.int64)
            layer_flat = torch.cat([full_experts, extra_experts], dim=0)
            shuffle_idx = torch.randperm(layer_flat.shape[0])
            layer_shuffled = layer_flat[shuffle_idx]
            layers.append(layer_shuffled.reshape(layer_shape))
        return torch.stack(layers, dim=0)

    expert_tensor = generate_layered_experts(num_layers=58,
                                             layer_shape=(32, 9))
    hotness = torch.randint(1, 100, (58, 256))
    physical_to_logical_map, _, _ = algo.rebalance_experts(
        hotness, 288, 4, 2, 32, expert_tensor)
    expert_tensor = physical_to_logical_map.reshape((58, 32, 9))
    for _ in range(3):
        physical_to_logical_map, _, _ = algo.rebalance_experts(
            hotness, 288, 4, 2, 32, expert_tensor)
        expert_tensor = physical_to_logical_map.reshape((58, 32, 9))


# Execute warmup on import
warmup_flashlb()
