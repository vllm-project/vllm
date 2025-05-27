# SPDX-License-Identifier: Apache-2.0
"""
Expert parallelism load balancer (EPLB) metrics and states.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.distributed import all_reduce

from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import get_ep_group
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import MixtureOfExperts

from .rebalance_algo import rebalance_experts
from .rebalance_execute import rearrange_expert_weights_inplace

logger = init_logger(__name__)


@dataclass
class EplbState:
    """EPLB metrics."""

    physical_to_logical_map: torch.Tensor
    """
    Mapping from physical experts to logical experts.

    Shape: (num_moe_layers, num_physical_experts)
    """
    logical_to_physical_map: torch.Tensor
    """
    Mapping from logical experts to physical experts.

    This is a sparse matrix, where -1 indicates no mapping.

    Shape: (num_moe_layers, num_logical_experts, num_redundant_experts + 1)
    """
    logical_replica_count: torch.Tensor
    """
    Number of replicas for each logical expert.

    Shape: (num_moe_layers, num_logical_experts)
    """

    expert_load_pass: torch.Tensor
    """
    Expert load during this forward pass. 
    We use the token count each expert processes as the load.

    Shape: (num_moe_layers, num_logical_experts)
    """
    expert_load_window: torch.Tensor
    """
    A sliding window of expert load.

    Shape: (window_size, num_moe_layers, num_logical_experts)
    """
    expert_load_window_step: int = 0
    """Current step in the sliding window."""
    expert_load_window_size: int = 0
    """Size of the expert load sliding window."""

    expert_rearrangement_step: int = 0
    """
    Steps after last rearrangement.
    Will trigger a rearrangement if it exceeds the threshold.
    """
    expert_rearrangement_step_interval: int = 0
    """Interval for expert rearrangement steps."""

    @staticmethod
    def build_initial_global_physical_to_logical_map(
        num_routed_experts: int,
        num_redundant_experts: int,
    ) -> Sequence[int]:
        """
        Build an initial expert arrangement using the following structure:
        [original routed experts, redundant experts]

        Returns:
            physical_to_logical_map (Sequence[int]): A list of integers,
                where each integer is the index of the logical expert
                that the corresponding physical expert maps to.
        """
        global_physical_to_logical_map = list(range(num_routed_experts))
        global_physical_to_logical_map += [
            i % num_routed_experts for i in range(num_redundant_experts)
        ]
        return global_physical_to_logical_map

    @classmethod
    def build(
        cls,
        model: MixtureOfExperts,
        device: torch.device,
        parallel_config: ParallelConfig,
    ) -> "EplbState":
        """
        Build the initial EPLB state.
        """
        physical_to_logical_map = (
            cls.build_initial_global_physical_to_logical_map(
                model.num_routed_experts,
                model.num_redundant_experts,
            ))
        physical_to_logical_map = torch.tensor(
            physical_to_logical_map,
            device=device,
        )
        logical_to_physical_map = torch.full(
            (model.num_logical_experts, model.num_redundant_experts + 1),
            -1,
            device=device,
        )
        logical_replica_count = torch.zeros(
            (model.num_logical_experts, ),
            device=device,
            dtype=torch.long,
        )

        for i in range(model.num_physical_experts):
            logical_idx = physical_to_logical_map[i]
            logical_to_physical_map[logical_idx,
                                    logical_replica_count[logical_idx]] = i
            logical_replica_count[logical_idx] += 1

        # Duplicate initial mapping for all layers
        physical_to_logical_map = physical_to_logical_map.unsqueeze(0).expand(
            model.num_moe_layers,
            -1,
        ).contiguous()
        logical_to_physical_map = logical_to_physical_map.unsqueeze(0).expand(
            model.num_moe_layers,
            -1,
            -1,
        ).contiguous()
        logical_replica_count = logical_replica_count.unsqueeze(0).expand(
            model.num_moe_layers,
            -1,
        ).contiguous()

        expert_load_pass = torch.zeros(
            (model.num_moe_layers, model.num_logical_experts),
            device=device,
        )
        expert_load_window_size = parallel_config.eplb_window_size
        expert_load_window = torch.zeros(
            (expert_load_window_size, model.num_moe_layers,
             model.num_logical_experts),
            device=device,
        )

        # Set the initial progress of rearrangement to 3/4
        eplb_step_interval = parallel_config.eplb_step_interval
        expert_rearrangement_step = max(
            0, eplb_step_interval - eplb_step_interval // 4)

        model.set_eplb_state(
            expert_load_pass,
            logical_to_physical_map,
            logical_replica_count,
        )

        return EplbState(
            physical_to_logical_map,
            logical_to_physical_map,
            logical_replica_count,
            expert_load_pass,
            expert_load_window,
            expert_load_window_size=expert_load_window_size,
            expert_rearrangement_step=expert_rearrangement_step,
            expert_rearrangement_step_interval=eplb_step_interval,
        )

    def step(self, model: MixtureOfExperts) -> None:
        """
        Step the EPLB state.
        """

        # Update the expert load sliding window
        self.expert_load_window[self.expert_load_window_step] = (
            self.expert_load_pass.clone())
        self.expert_load_window_step += 1
        if self.expert_load_window_step >= self.expert_load_window_size:
            self.expert_load_window_step = 0
        self.expert_load_pass.zero_()

        # Step the expert rearrangement step
        self.expert_rearrangement_step += 1
        if (self.expert_rearrangement_step
                >= self.expert_rearrangement_step_interval):
            self.expert_rearrangement_step = 0
            self.rearrange(model)

    def rearrange(self, model: MixtureOfExperts) -> None:
        """
        Rearrange the experts according to the current load.
        """

        ep_group = get_ep_group().device_group
        ep_rank = ep_group.rank()

        time_start = None
        is_main_rank = ep_rank == 0
        if is_main_rank:
            torch.cuda.synchronize()
            time_start = time.perf_counter()
            logger.info("Rearranging experts...")

        # Perform all-reduce to get the expert load across all ranks
        global_expert_load_window = self.expert_load_window.clone()
        all_reduce(global_expert_load_window, group=ep_group)

        # TODO(bowen): Treat differently for prefill and decode nodes
        num_replicas = model.num_physical_experts
        num_groups = model.num_expert_groups
        # TODO(bowen): Remove magic numbers
        num_nodes = (ep_group.size() + 7) // 8
        num_gpus = ep_group.size()

        # Get new expert mappings
        (
            new_physical_to_logical_map,
            new_logical_to_physical_map,
            new_logical_replica_count,
        ) = (rebalance_experts(
            global_expert_load_window,
            num_replicas,
            num_groups,
            num_nodes,
            num_gpus,
        ))

        # Update expert weights
        rearrange_expert_weights_inplace(
            self.physical_to_logical_map,
            new_physical_to_logical_map,
            model.expert_weights,
            ep_group,
        )

        self.physical_to_logical_map.copy_(new_physical_to_logical_map)
        self.logical_to_physical_map.copy_(new_logical_to_physical_map)
        self.logical_replica_count.copy_(new_logical_replica_count)

        if is_main_rank:
            assert time_start is not None
            torch.cuda.synchronize()
            time_end = time.perf_counter()
            logger.info(
                "Rearranged experts in %.2f seconds.",
                time_end - time_start,
            )
