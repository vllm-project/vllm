# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import math
from typing import Any

import numpy as np
import torch
from numba import njit  # type: ignore
from scipy import stats  # type: ignore
from scipy.optimize import linear_sum_assignment  # type: ignore

from .abstract import AbstractEplbPolicy

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)


@njit(fastmath=True, cache=True)
def min_max_replica(
    mu: np.ndarray, var: np.ndarray, num_available_replicas: int, current_replicas: np.ndarray, z_score: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Original min-max replica allocation algorithm
    Allocates replicas iteratively to expert with maximum unit load value

    Args:
        mu: Mean load of each expert (N,)
        var: Variance of each expert's load (N,)
        num_available_replicas: Total available replicas to allocate
        current_replicas: Initial replica count per expert (N,)
        z_score: Z-score for risk calculation (confidence level)

    Returns:
        current_replicas: Updated replica count per expert (N,)
        replicas_history: Replica allocation history (num_available_replicas+1, N)
    """
    N = mu.shape[0]
    unit_value = (mu + z_score * np.sqrt(var)) / current_replicas
    replicas_history = np.ones((num_available_replicas + 1, N), dtype=np.int32)
    replicas_history[0, :] = current_replicas[:]

    # Allocate replicas to expert with maximum unit value iteratively
    for r in range(num_available_replicas):
        max_idx = -1
        max_value = -1.0
        for idx in range(N):
            value = unit_value[idx]
            if value > max_value:
                max_value = value
                max_idx = idx

        current_replicas[max_idx] += 1
        unit_value[max_idx] = (mu[max_idx] + z_score * np.sqrt(var[max_idx])) / current_replicas[max_idx]
        replicas_history[r + 1, :] = current_replicas[:]

    return current_replicas, replicas_history


@njit
def max_delta_replica(
    mu: np.ndarray, var: np.ndarray, num_available_replicas: int, current_replicas: np.ndarray, z_score: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Maximum delta replica allocation algorithm
    Allocates replicas by maximum unit value delta after increment

    Args:
        mu: Mean load of each expert (N,)
        var: Variance of each expert's load (N,)
        num_available_replicas: Total available replicas to allocate
        current_replicas: Initial replica count per expert (N,)
        z_score: Z-score for risk calculation (confidence level)

    Returns:
        current_replicas: Updated replica count per expert (N,)
        replicas_history: Replica allocation history (num_available_replicas+1, N)
    """
    N = mu.shape[0]
    unit_value = (mu + z_score * np.sqrt(var)) / current_replicas
    replicas_history = np.ones((num_available_replicas + 1, N), dtype=np.int32)
    replicas_history[0, :] = current_replicas[:]

    # Allocate replicas by maximum unit value delta after increment
    for r in range(num_available_replicas):
        max_idx = -1
        max_value = -1.0
        for idx in range(N):
            value = unit_value[idx] / (current_replicas[idx] + 1)
            if value > max_value:
                max_value = value
                max_idx = idx

        current_replicas[max_idx] += 1
        unit_value[max_idx] = (mu[max_idx] + z_score * np.sqrt(var[max_idx])) / current_replicas[max_idx]
        replicas_history[r + 1, :] = current_replicas[:]

    return current_replicas, replicas_history


@njit
def percentage_replica(
    mu: np.ndarray, var: np.ndarray, num_available_replicas: int, current_replicas: np.ndarray, z_score: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Proportional replica allocation algorithm
    Allocates replicas proportionally to expert total load

    Args:
        mu: Mean load of each expert (N,)
        var: Variance of each expert's load (N,)
        num_available_replicas: Total available replicas to allocate
        current_replicas: Initial replica count per expert (N,)
        z_score: Z-score for risk calculation (confidence level)

    Returns:
        current_replicas: Updated replica count per expert (N,)
        replicas_history: Replica allocation history (num_available_replicas+1, N)
    """
    N = mu.shape[0]
    total_load = mu + z_score * np.sqrt(var)
    sum_total_load: float = np.sum(total_load)  # Add type annotation for mypy

    replicas_history = np.ones((num_available_replicas + 1, N), dtype=np.int32)
    replicas_history[0, :] = current_replicas[:]

    # Allocate replicas proportionally to expert total load
    for r in range(1, num_available_replicas + 1):
        add_slots = np.zeros(N, dtype=np.int32)

        if sum_total_load == 0.0:
            # Average allocation if total load is zero
            base_add = r // N
            extra = r % N
            add_slots[:] = base_add
            add_slots[:extra] += 1
        else:
            # Proportional allocation with remainder compensation
            quotas = (total_load / sum_total_load) * r
            base_add = np.floor(quotas).astype(np.int32)
            add_slots[:] = base_add
            remaining = r - np.sum(base_add)

            if remaining > 0:
                fractions = quotas - base_add
                indices = np.argsort(-fractions)
                add_slots[indices[:remaining]] += 1

        replicas_history[r] = current_replicas + add_slots

    return replicas_history[-1], replicas_history


def make_replica(
    mu: np.ndarray,
    var: np.ndarray,
    num_available_replicas: int,
    current_replicas: np.ndarray,
    z_score: float,
    method: str = "percentage",
) -> tuple[np.ndarray, np.ndarray]:
    if method == "percentage":
        return percentage_replica(mu, var, num_available_replicas, current_replicas, z_score)
    elif method == "max_delta":
        return max_delta_replica(mu, var, num_available_replicas, current_replicas, z_score)
    else:
        return min_max_replica(mu, var, num_available_replicas, current_replicas, z_score)


@njit(fastmath=True, cache=True)
def compute_updated_device_variance(
    new_expert_id: int,
    device_slots: np.ndarray,
    current_device_var: float,
    expert_var: np.ndarray,
    expert_cov: np.ndarray,
    expert_replicas: np.ndarray,
) -> float:
    """
    Compute updated device variance after adding a new expert
    Includes both individual variance and covariance with existing experts

    Args:
        new_expert_id: ID of the new expert to add
        device_slots: Current expert slots on the device (-1 for empty)
        current_device_var: Current variance of the device
        expert_var: Variance of each expert's load (N,)
        expert_cov: Covariance matrix between experts (N,N)
        expert_replicas: Replica count per expert (N,)

    Returns:
        new_device_var: Updated device variance after adding the expert
    """
    # Add variance of new expert
    new_device_var = current_device_var + expert_var[new_expert_id] / expert_replicas[new_expert_id] ** 2

    # Add covariance between new expert and existing experts on device
    for slot in device_slots:
        if slot == -1:
            break
        new_device_var += 2 * expert_cov[new_expert_id, slot] / expert_replicas[new_expert_id] / expert_replicas[slot]

    return new_device_var


@njit(fastmath=True, cache=True)
def lpt_deployment(
    mu: np.ndarray,
    var: np.ndarray,
    cov: np.ndarray,
    deployment: np.ndarray,
    deployed_replicas: np.ndarray,
    total_replicas: np.ndarray,
    z_score: float,
) -> np.ndarray:
    """
    Largest Processing Time (LPT) deployment algorithm
    Greedily deploys experts to device with minimal risk (mean + z*std)

    Args:
        mu: Mean load of each expert (N,)
        var: Variance of each expert's load (N,)
        cov: Covariance matrix between experts (N,N)
        deployment: Initial deployment matrix (num_ranks, num_slots_per_device)
        deployed_replicas: Already deployed replica count per expert (N,)
        total_replicas: Total target replica count per expert (N,)
        z_score: Z-score for risk calculation (confidence level)

    Returns:
        new_deployment: Updated expert deployment matrix
    """
    num_ranks, num_slots_per_device = deployment.shape

    # Initialize unit value and sort experts by load
    unit_value = mu / total_replicas
    sorted_indices = np.argsort(-unit_value)

    new_deployment = -np.ones_like(deployment)
    device_mu = np.zeros(num_ranks, dtype=np.float32)
    device_var = np.zeros(num_ranks, dtype=np.float32)
    dev_ptr = np.zeros(num_ranks, dtype=np.int32)

    # Copy existing deployment first
    for dev in range(num_ranks):
        for slot in deployment[dev]:
            if slot != -1:
                device_mu[dev] += mu[slot] / total_replicas[slot]
                device_var[dev] += compute_updated_device_variance(
                    slot, new_deployment[dev], device_var[dev], var, cov, total_replicas
                )
                new_deployment[dev, dev_ptr[dev]] = slot
                dev_ptr[dev] += 1

    # Greedily deploy remaining replicas to device with minimal risk
    for idx in sorted_indices:
        for _ in range(total_replicas[idx] - deployed_replicas[idx]):
            best_dev = -1
            best_risk = 1e30
            best_mu = -1.0
            best_var = -1.0
            for dev in range(num_ranks):
                if dev_ptr[dev] >= num_slots_per_device:
                    continue
                if idx in new_deployment[dev]:
                    continue
                # Calculate temporary device load and risk
                temp_mu = device_mu[dev] + mu[idx] / total_replicas[idx]
                temp_var = compute_updated_device_variance(
                    idx, new_deployment[dev], device_var[dev], var, cov, total_replicas
                )

                risk = temp_mu + z_score * np.sqrt(temp_var)
                if risk < best_risk:
                    best_risk = risk
                    best_dev = dev
                    best_mu = temp_mu
                    best_var = temp_var

            # Update device state with best deployment choice
            device_mu[best_dev] = best_mu
            device_var[best_dev] = best_var
            new_deployment[best_dev, dev_ptr[best_dev]] = idx
            dev_ptr[best_dev] += 1

    return new_deployment


@njit(fastmath=True, cache=True)
def compute_score(val_data: np.ndarray, simulated_replicas: np.ndarray, simulated_deployment: np.ndarray) -> np.float32:
    """
    Calculate load balance score: (max_device_load * num_ranks) / total_load
    Lower score means better load balance

    Args:
        val_data: Validation load data (T, N) - T time steps, N experts
        simulated_replicas: Replica count per expert (N,)
        simulated_deployment: Expert deployment matrix (D, K) - D devices, K slots

    Returns:
        mean_score: Average load balance score over time steps
    """
    T, N = val_data.shape
    D, K = simulated_deployment.shape
    scores = np.empty((T,), dtype=np.float32)
    for t in range(T):
        max_load = 0.0  # Explicit float type to avoid int/float mix
        tot_load = 0.0
        for d in range(D):
            s = 0.0
            for k in range(K):
                idx = simulated_deployment[d, k]
                s += val_data[t, idx] / simulated_replicas[idx]
            tot_load += s
            max_load = max(max_load, s)
        # Add small epsilon to avoid division by zero
        scores[t] = (max_load * D + 1e-2) / (tot_load + 1e-2)

    return np.mean(scores)



def generate_layered_experts(
    num_layers: int = 58, layer_shape: tuple[int, int] = (32, 9), expert_min: int = 0, expert_max: int = 255
) -> torch.Tensor:
    """
    Generate layered expert deployment matrix
    Each layer contains all experts [expert_min, expert_max] at least once
    Remaining slots filled with random experts

    Args:
        num_layers: Number of layers to generate
        layer_shape: Shape of each layer's deployment matrix (rows, cols)
        expert_min: Minimum expert ID (inclusive)
        expert_max: Maximum expert ID (inclusive)

    Returns:
        layers: Layered expert deployment tensor (num_layers, *layer_shape)
    """
    # Basic parameter calculation
    expert_num = expert_max - expert_min + 1
    layer_total = layer_shape[0] * layer_shape[1]
    extra_slots = layer_total - expert_num

    # Feasibility check: layer capacity ≥ expert count
    assert layer_total >= expert_num, (
        f"Layer element count {layer_total} < expert count {expert_num}, cannot cover all experts"
    )

    # Generate each layer
    layers: list[torch.Tensor] = []
    for _ in range(num_layers):
        # Full expert sequence (cover all experts once)
        full_experts = torch.arange(expert_min, expert_max + 1, dtype=torch.int64)  # (expert_num,)

        # Random extra experts for remaining slots
        extra_experts = torch.randint(
            expert_min, expert_max + 1, size=(extra_slots,), dtype=torch.int64
        )  # (extra_slots,)

        # Concatenate and shuffle for random distribution
        layer_flat = torch.cat([full_experts, extra_experts], dim=0)  # (layer_total,)
        shuffle_idx = torch.randperm(layer_flat.shape[0])
        layer_shuffled = layer_flat[shuffle_idx]

        # Reshape to target layer shape
        layer = layer_shuffled.reshape(layer_shape)
        layers.append(layer)

    # Stack all layers
    return torch.stack(layers, dim=0)  # (num_layers, *layer_shape)


class FlashTree:
    def __init__(self, X, num_replicas, num_ranks, z_score=0.674, depth=4, width=8):
        super().__init__()
        self.num_replicas = num_replicas
        self.num_ranks = num_ranks
        self.z_score = z_score
        self.depth = depth
        self.width = width

        self.X = X
        self.mu, self.var, self.cov = FlashTree.compute_statistics(X)

    @staticmethod
    def compute_statistics(X):
        T, N = X.shape
        mean_ = np.mean(X, axis=0)
        if T > 1:
            X_centered = X - mean_
            variance_ = np.sum(X_centered**2, axis=0) / (T - 1)
            cov_matrix = (X_centered.T @ X_centered) / (T - 1)
        else:
            variance_ = np.zeros((N,))
            cov_matrix = np.zeros((N, N))
        return mean_, variance_, cov_matrix

    def neighbor_search(
        self, low: int, high: int, initial: int, max_range: int, get_score: Any, *args: Any
    ) -> tuple[int, float, np.ndarray]:
        """
        Local neighbor search for optimal replica number
        Search [initial-max_range, initial+max_range] within [low, high]

        Args:
            low: Lower bound of search range
            high: Upper bound of search range
            initial: Initial replica number to start search
            max_range: Maximum search range from initial value
            get_score: Function to compute score for a given replica number
            *args: Additional arguments for get_score function

        Returns:
            best_x: Optimal replica number
            best_score: Best load balance score
            best_sim: Corresponding deployment simulation result
        """
        max_range = min(max(initial - low, high - initial), max_range)
        best_x = initial
        best_score, best_sim = get_score(initial, *args)

        # Search left and right neighbors
        for r in range(1, max_range + 1):
            left = initial - r
            if left >= low:
                score, sim = get_score(left, *args)
                if score < best_score:
                    best_x, best_score, best_sim = left, score, sim

            right = initial + r
            if right <= high:
                score, sim = get_score(right, *args)
                if score < best_score:
                    best_x, best_score, best_sim = right, score, sim

        return best_x, best_score, best_sim

    def optimize_balanceness(self):
        X_row = self.X
        mu, var, cov = self.mu, self.var, self.cov
        num_total_replicas = self.num_replicas
        num_ranks = self.num_ranks
        z_score = self.z_score
        depth, width = self.depth, self.width

        num_experts = mu.shape[0]
        num_available_replicas = num_total_replicas - num_experts

        if depth <= 1:
            default_replicas = np.ones(num_experts, dtype=np.int32)
            default_replicas = make_replica(mu, var, num_available_replicas, default_replicas, z_score)[0]
            default_deployment = -np.ones((num_ranks, num_total_replicas // num_ranks), dtype=np.int32)
            default_deployment = lpt_deployment(
                mu, var, cov, default_deployment, np.zeros(num_experts, dtype=np.int32), default_replicas, z_score
            )
            default_par = compute_score(X_row, default_replicas, default_deployment)
            return default_deployment, default_replicas, default_par

        interval_size = math.ceil(num_experts / depth)
        weight = mu + z_score * np.sqrt(var)
        idx = np.argsort(-weight)

        deployed_replicas = np.zeros(num_experts, dtype=np.int32)
        deployment = -np.ones((num_ranks, num_total_replicas // num_ranks), dtype=np.int32)

        def _lpt_deployment(replicas):
            nonlocal mu, var, cov, deployment, deployed_replicas, z_score
            return lpt_deployment(mu, var, cov, deployment, np.zeros_like(replicas), replicas, z_score)

        def get_score(
            f: Any,
            val_data: np.ndarray,
            deployed_replicas: np.ndarray,
            current_idx: np.ndarray,
            current_replicas: np.ndarray,
            remaind_idx: np.ndarray,
            remaind_replicas: np.ndarray,
        ) -> tuple[float, np.ndarray]:
            """
            Wrapper to compute load balance score for replica allocation simulation

            Args:
                f: Deployment function (e.g., _lpt_deployment)
                val_data: Validation load data (T, N)
                deployed_replicas: Already deployed replica count per expert (N,)
                current_idx: Indices of current expert group
                current_replicas: Replica count for current expert group
                remaind_idx: Indices of remaining expert group
                remaind_replicas: Replica count for remaining expert group

            Returns:
                score: Load balance score
                simulated_deployment: Simulated expert deployment matrix
            """
            # Simulate replica allocation and deployment
            simulated_replicas = deployed_replicas.copy()
            simulated_replicas[current_idx] = current_replicas
            simulated_replicas[remaind_idx] = remaind_replicas
            simulated_deployment = f(simulated_replicas)

            # Calculate load balance score
            score = compute_score(val_data, simulated_replicas, simulated_deployment)
            return score, simulated_deployment

        for node in range(depth - 1):
            low, high = 0, num_available_replicas
            simulation_idx = idx[node * interval_size :]
            current_idx = idx[node * interval_size : (node + 1) * interval_size]
            remaind_idx = idx[(node + 1) * interval_size :]

            simulation_replicas = make_replica(
                mu[simulation_idx], var[simulation_idx], high, np.ones(simulation_idx.shape[0], dtype=np.int32), z_score
            )[0]
            current_replicas_f = make_replica(
                mu[current_idx], var[current_idx], high, np.ones(current_idx.shape[0], dtype=np.int32), z_score
            )[1]
            remaind_replicas_f = make_replica(
                mu[remaind_idx], var[remaind_idx], high, np.ones(remaind_idx.shape[0], dtype=np.int32), z_score
            )[1]

            initial_replicas = (simulation_replicas[:interval_size] - 1).sum()

            best_replica, _, _ = self.neighbor_search(
                low,
                high,
                initial_replicas,
                width,
                lambda mid,
                ci=current_idx,
                crf=current_replicas_f,
                ri=remaind_idx,
                rrf=remaind_replicas_f,
                nar=num_available_replicas: get_score(
                    _lpt_deployment, X_row, deployed_replicas, ci, crf[mid], ri, rrf[nar - mid]
                ),
            )

            deployed_replicas[current_idx] = current_replicas_f[best_replica]
            num_available_replicas -= best_replica

            if not num_available_replicas or node == depth - 2:
                deployed_replicas[remaind_idx] = remaind_replicas_f[num_available_replicas]
                break

        final_deployment = -np.ones((num_ranks, num_total_replicas // num_ranks), dtype=np.int32)
        final_deployment = lpt_deployment(
            mu, var, cov, final_deployment, np.zeros_like(deployed_replicas), deployed_replicas, z_score
        )
        final_par = compute_score(X_row, deployed_replicas, final_deployment)

        return final_deployment, deployed_replicas, final_par


class FlashlbEplbPolicy(AbstractEplbPolicy):
    """
    Flash Load Balancing (FlashLB) policy for expert deployment optimization
    Implements layered tree search with load balance score optimization
    """

    def __init__(self):
        """
        Initialize FlashLB policy with dynamic configuration

        Args:
            config: Dynamic configuration object containing policy parameters
        """
        # Max window size for expert hotness observation
        self.max_observation_window = 2000
        # Threshold ratio for load balance update trigger
        self.update_threshold_ratio = 0.95
        # Threshold value for load balance update trigger
        self.update_threshold_value = 0.9
        # Upper bound of layers to update per iteration
        self.update_layers_upper_bound = -1
        # Z-score for risk calculation (default: 75% confidence)
        self.z_score = stats.norm.ppf(0.75)
        # Tree search depth for flash_tree algorithm
        self.depth = 4
        # Tree search width for neighbor search
        self.width = 8
        self.sample_size = 64

        # Runtime state storage with type annotations
        self.average_to_peak_history: dict[int, float] = {}  # Layer-wise load balance history
        self.hotness_window: dict[int, dict[str, Any]] = {}  # Layer-wise hotness stats and buffer
        self.current_deployment: dict[int, np.ndarray] = {}  # Current expert deployment per layer
        self.current_deployed_replicas: dict[int, np.ndarray] = {}  # Current replica count per expert per layer

    def min_max_replica(
        self, mu: np.ndarray, var: np.ndarray, num_available_replicas: int, current_replicas: np.ndarray, z_score: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Wrapper for original min-max replica allocation

        Args:
            mu: Mean load of each expert (N,)
            var: Variance of each expert's load (N,)
            num_available_replicas: Total available replicas to allocate
            current_replicas: Initial replica count per expert (N,)
            z_score: Z-score for risk calculation (confidence level)

        Returns:
            current_replicas: Updated replica count per expert (N,)
            replicas_history: Replica allocation history (num_available_replicas+1, N)
        """
        return min_max_replica(mu, var, num_available_replicas, current_replicas, z_score)

    @staticmethod
    def compute_statistics(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute mean, variance and covariance matrix from time series data

        Args:
            X: Time series data (T, N) - T steps, N experts

        Returns:
            mean: Mean load per expert (N,)
            variance: Variance of load per expert (N,)
            cov_matrix: Covariance matrix between experts (N,N)
        """
        T, N = X.shape
        mean_ = np.mean(X, axis=0)
        if T > 1:
            X_centered = X - mean_
            variance_ = np.sum(X_centered**2, axis=0) / (T - 1)
            cov_matrix = (X_centered.T @ X_centered) / (T - 1)
        else:
            # Zero stats if only one sample
            variance_ = np.zeros((N,))
            cov_matrix = np.zeros((N, N))
        return mean_, variance_, cov_matrix

    @staticmethod
    def sliding_update_stats(
        mean: np.ndarray, cov: np.ndarray, x_old: np.ndarray, x_new: np.ndarray, T: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Update statistics with sliding window (replace old data with new data)

        Args:
            mean: Current mean statistics (N,)
            cov: Current covariance matrix (N,N)
            x_old: Old data batch to remove (t, N)
            x_new: New data batch to add (t, N)
            T: Window size

        Returns:
            new_mean: Updated mean (N,)
            new_var: Updated variance (N,)
            new_cov: Updated covariance matrix (N,N)
        """
        assert x_new.shape == x_old.shape
        mean = mean.astype(np.float64, copy=False)
        cov = cov.astype(np.float64, copy=False)
        x_old = x_old.astype(np.float64, copy=False)
        x_new = x_new.astype(np.float64, copy=False)

        # Update mean
        sum_old = np.sum(x_old, axis=0)
        sum_new = np.sum(x_new, axis=0)
        deltaS = sum_new - sum_old
        new_mean = mean + deltaS / T

        # Update covariance matrix
        x_old_centered = x_old - mean
        x_new_centered = x_new - mean

        SA_mu = np.dot(x_old_centered.T, x_old_centered)
        SB_mu = np.dot(x_new_centered.T, x_new_centered)

        Sigma = cov * (T - 1)
        Sigma_new = Sigma + SB_mu - SA_mu - np.outer(deltaS, deltaS) / T
        new_cov = Sigma_new / (T - 1)

        new_var = np.diag(new_cov)
        return new_mean, new_var, new_cov

    @staticmethod
    def incremental_update_stats(
        mean: np.ndarray, cov: np.ndarray, x_new: np.ndarray, T: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Incrementally update statistics with new data (expand window)

        Args:
            mean: Current mean statistics (N,)
            cov: Current covariance matrix (N,N)
            x_new: New data batch to add (t, N)
            T: Current window size

        Returns:
            new_mean: Updated mean (N,)
            new_var: Updated variance (N,)
            new_cov: Updated covariance matrix (N,N)
            new_T: Updated window size
        """
        t, N = x_new.shape
        sum_new = np.sum(x_new, axis=0)
        new_T = T + t

        # Update mean
        new_mean = (T * mean + sum_new) / new_T

        # Update covariance matrix
        if T > 1:
            x_new_centered = x_new - new_mean
            cov_new = cov * (T - 1)
            cov_new += np.dot(x_new_centered.T, x_new_centered)
            cov_new += T * np.outer(mean - new_mean, mean - new_mean)
            new_cov = cov_new / (new_T - 1)
        else:
            # Special case for initial single sample
            x_old = mean.reshape(1, -1)
            x_old_centered = x_old - new_mean
            x_new_centered = x_new - new_mean
            sum_squares = np.dot(x_old_centered.T, x_old_centered) + np.dot(x_new_centered.T, x_new_centered)
            new_cov = sum_squares / (new_T - 1)

        new_var = np.diag(new_cov)
        return new_mean, new_var, new_cov, new_T

    def register_hotness(
        self, rank_load: np.ndarray, num_layers: int, num_experts: int
    ) -> None:
        """
        Update expert hotness statistics with sliding window for all layers

        Args:
            deployment: Expert deployment matrix (num_layers, num_ranks, num_slots)
            rank_load: Load data (num_stages, num_layers, num_experts)
            num_layers: Total number of layers
            num_experts: Total number of experts
        """
        num_stage = rank_load.shape[0]
        hotness = np.zeros((num_stage, num_layers, num_experts), dtype=rank_load.dtype)
        for stage in range(num_stage):
            for layer in range(num_layers):
                hotness[stage, layer, :] += rank_load[stage, layer, :]

        hotness += 1
        window_length = self.max_observation_window

        for layer in range(num_layers):
            new_X = hotness[-window_length:, layer, :]
            t = new_X.shape[0]

            if layer not in self.hotness_window:
                self.hotness_window[layer] = {
                    "buffer": np.zeros((window_length, num_experts), dtype=new_X.dtype),
                    "start": 0,
                    "length": 0,
                }

            info = self.hotness_window[layer]
            buf = info["buffer"]
            start = info["start"]
            length = info["length"]

            if start + t <= window_length:
                buf[start : start + t] = new_X
            else:
                first_part = window_length - start
                buf[start:] = new_X[:first_part]
                buf[: t - first_part] = new_X[first_part:]

            start = (start + t) % window_length
            length = min(window_length, length + t)

            self.hotness_window[layer]["buffer"] = buf
            self.hotness_window[layer]["start"] = start
            self.hotness_window[layer]["length"] = length

    def need_update(self, layer_id: int = 0) -> bool:
        """
        Check if layer needs load balance update
        Trigger update if load balance ratio drops below threshold

        Args:
            layer_id: Layer index to check

        Returns:
            bool: True if update is needed, False otherwise
        """
        past_average_to_peak_ratio = self.average_to_peak_history.get(layer_id, 0.0)
        if past_average_to_peak_ratio == 0.0:
            # Force update for first iteration
            return True

        # Calculate current load balance ratio (average/peak load)
        hotness = self.hotness_window[layer_id]["buffer"]
        average_to_peak_ratio = 1 / compute_score(
            hotness, self.current_deployed_replicas[layer_id], self.current_deployment[layer_id]
        )

        # Check update conditions
        return (
            average_to_peak_ratio < past_average_to_peak_ratio * self.update_threshold_ratio
            or average_to_peak_ratio < self.update_threshold_value
        )

    @staticmethod
    @njit
    def compute_match(src_counts: np.ndarray, dst_counts: np.ndarray, N: int, M: int) -> np.ndarray:
        """
        Compute match matrix between source and destination expert counts
        match[i,j] = total min(src_counts[i,k], dst_counts[j,k]) for all k

        Args:
            src_counts: Source expert count histogram (N, max_val)
            dst_counts: Destination expert count histogram (N, max_val)
            N: Number of rows in deployment matrix
            M: Number of columns in deployment matrix

        Returns:
            matches: Match matrix (N, N)
        """
        matches = np.zeros((N, N), dtype=np.int32)
        for i in range(N):
            for j in range(N):
                match = 0
                for k in range(N * M):
                    match += min(src_counts[i, k], dst_counts[j, k])
                matches[i, j] = match
        return matches

    @staticmethod
    def minimize_redeploy_with_inner_permutation(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """
        Minimize expert redeployment by permuting destination rows/columns
        Aligns destination deployment with source to reduce expert movement

        Args:
            src: Source deployment matrix (N, M)
            dst: Destination deployment matrix (N, M)

        Returns:
            dst_reordered: Reordered destination deployment matrix
        """
        if src.shape != dst.shape:
            raise ValueError("src and dst must have same shape (N, M)")
        N, M = src.shape
        valid_src = src
        valid_dst = dst

        # Calculate expert count histogram for each row
        max_val = N * M
        src_counts = np.array([np.bincount(row[row != -1], minlength=max_val) for row in valid_src], dtype=np.int32)
        dst_counts = np.array([np.bincount(row[row != -1], minlength=max_val) for row in valid_dst], dtype=np.int32)

        # Compute match matrix and optimal row mapping (Hungarian algorithm)
        matches = FlashlbEplbPolicy.compute_match(src_counts, dst_counts, N, M)
        cost = M - matches
        row_ind, col_ind = linear_sum_assignment(cost)
        mapping = list(zip(row_ind.tolist(), col_ind.tolist()))

        # Reorder dst rows and columns to align with src
        dst_reordered = np.empty_like(dst)
        for src_idx, dst_idx in mapping:
            s_row = src[src_idx]
            d_row = dst[dst_idx]
            # Map expert values to their positions in dst row
            val_to_positions: dict[int, list[int]] = {}  # Add type annotation for mypy
            for pos, v in enumerate(d_row):
                val_to_positions.setdefault(v, []).append(pos)

            reordered = np.empty(M, dtype=dst.dtype)
            assigned = [False] * M
            used_dst_positions = set()

            # Assign existing experts first to minimize movement
            for pos_src, v in enumerate(s_row):
                positions = val_to_positions.get(v)
                if positions:
                    dst_pos = positions.pop()
                    reordered[pos_src] = v
                    assigned[pos_src] = True
                    used_dst_positions.add(dst_pos)

            # Fill remaining positions with unassigned experts
            remaining = [d_row[p] for p in range(M) if p not in used_dst_positions]
            ri = 0
            for pos in range(M):
                if not assigned[pos]:
                    reordered[pos] = remaining[ri]
                    ri += 1
            dst_reordered[src_idx] = reordered
        return dst_reordered
    
    def rebalance_experts(
        self,
        weight: torch.Tensor,
        num_replicas: int,
        num_groups: int,
        num_nodes: int,
        num_ranks: int,
        old_global_expert_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Entry point for expert-parallelism load balancer.

        Parameters:
            weight: [layers, num_logical_experts], the load statistics for all
                logical experts
            num_replicas: number of physical experts, must be a multiple of
                `num_gpus`
            num_groups: number of expert groups
            num_nodes: number of server nodes, where the intra-node network
                (e.g, NVLink) is faster
            num_ranks: number of ranks, must be a multiple of `num_nodes`
            old_global_expert_indices: [layers, num_logical_experts], the old global
                expert indices. Used to avoid unnecessary weight copying
                for experts moving within one rank.
        Returns:
            phy2log: [layers, num_replicas], the expert
                index of each replica
        """
        assert num_ranks % num_nodes == 0
        assert num_replicas % num_ranks == 0
        assert old_global_expert_indices is not None

        current_deployment = np.array(old_global_expert_indices)
        expert_workload = np.array(weight)

        # Add batch dimension if missing
        if expert_workload.ndim == 3:
            expert_workload = expert_workload[np.newaxis, ...]
        num_layers = expert_workload.shape[1]
        num_expert = np.unique(current_deployment[0].reshape(-1)).shape[0]

        self.register_hotness(expert_workload, num_layers, num_expert)

        # Initialize current deployment state for all layers
        for layer in range(num_layers):
            self.current_deployment[layer] = current_deployment[layer]
            self.current_deployed_replicas[layer] = np.bincount(
                current_deployment[layer].reshape(-1), minlength=num_expert
            )

        # Initialize output variables
        new_par = np.zeros((num_layers,), dtype=np.float32)
        new_deployment = np.zeros((num_layers, num_ranks, num_replicas // num_ranks), dtype=np.int32)
        new_deployed_replicas = np.zeros((num_layers, num_expert), dtype=np.int32)
        new_average_to_peak_ratio = np.zeros((num_layers,), dtype=np.float32)
        delta_average_to_peak_ratio = np.zeros((num_layers,), dtype=np.float32)
        pars = np.zeros((num_layers,), dtype=np.float32)

        # Optimize each layer
        for layer in range(num_layers):
            if not self.need_update(layer):
                # Keep current deployment if no update needed
                new_deployment[layer] = self.current_deployment[layer]
                new_deployed_replicas[layer] = self.current_deployed_replicas[layer]
                new_average_to_peak_ratio[layer] = self.average_to_peak_history.get(layer, 0.0)
                new_par[layer] = 1 / new_average_to_peak_ratio[layer] if new_average_to_peak_ratio[layer] != 0 else 0.0
                delta_average_to_peak_ratio[layer] = 0
                continue

            # Get layer hotness stats
            layer_info = self.hotness_window[layer]
            buf = layer_info["buffer"]
            start = layer_info["start"]
            length = layer_info["length"]

            # Get valid hotness data from sliding window
            idx = np.arange(start, start + length) % self.max_observation_window
            data = buf[idx]

            shape = data.shape
            window = max(length // self.sample_size, 1)
            data = data[-window * self.sample_size :].reshape((-1, window, *shape[1:])).sum(1)
            # Flash tree search for optimal deployment
            flash_tree = FlashTree(data, num_replicas, num_ranks, self.z_score, self.depth, self.width)
            best_deployment, best_replicas, best_score = flash_tree.optimize_balanceness()

            # Update layer state
            new_deployed_replicas[layer] = best_replicas
            new_average_to_peak_ratio[layer] = 1 / best_score

            current_deployment = self.current_deployment.get(layer, None)

            new_deployment[layer] = best_deployment
            # Minimize redeployment by permuting new deployment
            new_deployment[layer] = FlashlbEplbPolicy.minimize_redeploy_with_inner_permutation(
                current_deployment, best_deployment
            )
            current_average_to_peak_ratio = 1 / compute_score(
                buf, self.current_deployed_replicas.get(layer), current_deployment
            )
            delta_average_to_peak_ratio[layer] = new_average_to_peak_ratio[layer] - current_average_to_peak_ratio
            pars[layer] = best_score

        # Select layers to update (sorted by improvement, positive delta only)
        priority_idx = np.argsort(-delta_average_to_peak_ratio)
        priority_idx = priority_idx[delta_average_to_peak_ratio[priority_idx] > 0]
        # Apply upper bound of layers to update
        if self.update_layers_upper_bound > 0:
            priority_idx = priority_idx[: self.update_layers_upper_bound]

        # Update global state with optimal deployments
        for layer in priority_idx:
            self.current_deployment[layer] = new_deployment[layer]
            self.current_deployed_replicas[layer] = new_deployed_replicas[layer]
            self.average_to_peak_history[layer] = new_average_to_peak_ratio[layer]

        deployment = current_deployment.copy()

        physical_to_logical_ndarray = deployment.reshape(
            (num_layers, num_replicas)
        ).astype(np.int32)

        phy2log = torch.tensor(
            physical_to_logical_ndarray, dtype=torch.int32, device=weight.device
        )
        return phy2log
    
    @classmethod
    def warm_up(cls) -> None:
        """
        Warm up FlashLB algorithm with dummy data
        Pre-compiles numba functions and initializes state
        """

        algo = cls()
        # Generate dummy expert deployment tensor
        expert_tensor = generate_layered_experts(num_layers=58, layer_shape=(32, 9))
        # Run rebalance with dummy workload data
        algo.rebalance_experts(weight=torch.randint(1, 1000, (100, 58, 256)), num_replicas=288, num_groups=1, num_nodes=2, num_ranks=32, old_global_expert_indices=expert_tensor)
        