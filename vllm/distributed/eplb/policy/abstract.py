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
    Published expert load window. Planning uses the explicit CPU `weight`
    argument, leaving this source tensor unchanged.
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
        weight: torch.Tensor,
        stats: EplbStats,
        old_global_expert_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Entry point for expert-parallelism load balancer.

        Args:
            weight: CPU load tensor from the same snapshot as `stats`.
                Built-in policies use `[layers, num_logical_experts]`.
            stats: Published load/topology snapshot. Policy-specific subclasses
                may provide additional inputs. Use `weight` for computation;
                the snapshot's source tensor may remain on the device.
            old_global_expert_indices: [layers, num_replicas], the old global
                expert indices. Used to avoid unnecessary weight copying
                for experts moving within one rank.

        Returns:
            physical_to_logical_map: [layers, num_replicas], the expert
                index of each replica, on CPU.

        """
        raise NotImplementedError
