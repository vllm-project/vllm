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
    def rebalance_experts(
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
