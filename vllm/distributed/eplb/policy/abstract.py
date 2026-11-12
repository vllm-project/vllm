# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class EplbStats:
    """Load and topology snapshot for one EPLB planning call.

    Policies may accept subclasses with additional planning inputs."""

    global_expert_load_window: torch.Tensor
    """
    Experts load window.
    Built-in policies use shape (num_moe_layers, num_logical_experts).
    """
    num_replicas: int
    """
    Number of physical experts.
    """
    num_groups: int
    """
    Number of expert groups.
    """
    num_nodes: int
    """
    Number of nodes.
    """
    num_gpus: int
    """
    Number of GPUs.
    """


class AbstractEplbPolicy(ABC):
    @abstractmethod
    def rebalance_experts(
        self,
        stats: EplbStats,
        old_global_expert_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Entry point for expert-parallelism load balancer.

        Args:
            stats: Load and topology snapshot. The load tensor is on CPU;
                policy-specific subclasses may provide additional inputs.
            old_global_expert_indices: [layers, num_replicas], the old global
                expert indices. Used to avoid unnecessary weight copying
                for experts moving within one rank.

        Returns:
            physical_to_logical_map: [layers, num_replicas], the expert
                index of each replica, on CPU.

        """
        raise NotImplementedError
