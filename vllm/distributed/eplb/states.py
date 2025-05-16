# SPDX-License-Identifier: Apache-2.0
"""
Expert parallelism load balancer (EPLB) metrics and states.
"""

from dataclasses import dataclass

import torch

from vllm.config import ParallelConfig
from vllm.model_executor.models.interfaces import IsMixtureOfExperts


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

    Shape: (num_moe_layers, num_logical_experts)
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

    Shape: (num_moe_layers, num_local_physical_experts)
    """
    expert_load_window: torch.Tensor
    """
    A sliding window of expert load.

    Shape: (window_size, num_moe_layers, num_local_physical_experts)
    """
    expert_load_window_step: int = 0
    """Current step in the sliding window."""

    expert_rearrangement_step: int = 0
    """
    Steps after last rearrangement.
    Will trigger a rearrangement if it exceeds the threshold.
    """

    @staticmethod
    def build_initial_global_physical_to_logical_map(
        num_routed_experts: int,
        num_redundant_experts: int,
    ) -> list[int]:
        """
        Build an initial expert arrangement using the following structure:
        [original routed experts, redundant experts]
        """
        global_physical_to_logical_map = list(range(num_routed_experts))
        global_physical_to_logical_map += [
            i % num_routed_experts for i in range(num_redundant_experts)
        ]
        return global_physical_to_logical_map

    @classmethod
    def build(
        cls,
        model: IsMixtureOfExperts,
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
            (model.num_moe_layers, model.num_local_physical_experts),
            device=device,
        )
        expert_load_window = torch.zeros(
            (parallel_config.eplb_window_size, model.num_moe_layers,
             model.num_local_physical_experts),
            device=device,
        )

        # Set the initial progress of rearrangement to 3/4
        eplb_step_interval = parallel_config.eplb_step_interval
        expert_rearrangement_step = max(
            0, eplb_step_interval - eplb_step_interval // 4)

        return EplbState(
            physical_to_logical_map,
            logical_to_physical_map,
            logical_replica_count,
            expert_load_pass,
            expert_load_window,
            expert_rearrangement_step=expert_rearrangement_step,
        )
