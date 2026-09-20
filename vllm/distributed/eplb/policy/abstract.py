# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch.distributed import ProcessGroup


@dataclass(frozen=True)
class EplbTopology:
    num_groups: int
    num_nodes: int
    num_ranks: int


@dataclass(frozen=True)
class EplbRebalanceContext:
    """CPU inputs shared by built-in and out-of-tree EPLB policies.

    Loads are a nonempty, oldest-to-newest ``[samples, layers, experts]``
    tensor; the current placement is ``[layers, physical_experts]``.
    """

    load_window_cpu: torch.Tensor
    physical_to_logical_map_cpu: torch.Tensor
    topology: EplbTopology
    num_replicas: int
    cpu_group: ProcessGroup


@dataclass(frozen=True)
class EplbPlan:
    """CPU integer ``[layers, physical_experts]`` target from an EPLB policy."""

    physical_to_logical_map: torch.Tensor


class EplbPolicyState:
    """Mutable per-model state, advanced only after a successful commit."""


class AbstractEplbPolicy(ABC):
    """Algorithm shared by all models in one EPLB state.

    Implementations may keep immutable configuration here; model-local mutable
    data belongs in :class:`EplbPolicyState`.
    """

    def create_state(self, num_moe_layers: int) -> EplbPolicyState:
        """Create independent state for one model."""
        return EplbPolicyState()

    @abstractmethod
    def plan_rebalance(
        self, context: EplbRebalanceContext, policy_state: EplbPolicyState
    ) -> EplbPlan:
        """Build a plan while treating committed state as read-only.

        The returned plan and its tensors remain read-only after this call.
        """
        raise NotImplementedError

    def on_layer_committed(
        self, policy_state: EplbPolicyState, plan: EplbPlan, layer_idx: int
    ) -> None:
        """Advance state after one layer of the read-only plan is committed.

        This is the only policy hook allowed to mutate ``policy_state``. The same
        complete plan is passed once for each successfully installed layer.
        """
        return None

    def _plan_from_legacy(self, context: EplbRebalanceContext) -> EplbPlan:
        physical_to_logical_map = self.rebalance_experts(
            context.load_window_cpu.sum(dim=0),
            context.num_replicas,
            context.topology.num_groups,
            context.topology.num_nodes,
            context.topology.num_ranks,
            context.physical_to_logical_map_cpu,
        )
        return EplbPlan(physical_to_logical_map)

    @classmethod
    def rebalance_experts(
        cls,
        weight: torch.Tensor,
        num_replicas: int,
        num_groups: int,
        num_nodes: int,
        num_ranks: int,
        old_global_expert_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Entry point for expert-parallelism load balancer.

        Parameters:
            weight: [layers, num_logical_experts], the load statistics
                for all logical experts
            num_replicas: number of physical experts, must be a multiple of
                `num_ranks`
            num_groups: number of expert groups
            num_nodes: number of server nodes
            num_ranks: number of ranks, must be a multiple of `num_nodes`
            old_global_expert_indices: [layers, num_logical_experts], the old global
                expert indices. Used to avoid unnecessary weight copying
                for experts moving within one rank.
        Returns:
            physical_to_logical_map: [layers, num_replicas], the expert
                index of each replica
        """
        raise NotImplementedError
