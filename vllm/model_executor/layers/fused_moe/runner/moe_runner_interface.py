# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod
from collections.abc import Iterable

import torch

from vllm.config.parallel import ExpertPlacementStrategy
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
)


# Pluggable?
class MoERunnerInterface(torch.nn.Module):
    """
    Abstract base class for Mixture of Experts (MoE) runners.

    This class defines the interface that all MoE runner implementations must follow.
    MoE runners are responsible for executing the forward pass of MoE layers, handling
    expert routing, and managing tensor parallel operations.
    """

    def __init__(self):
        super().__init__()
        # HACK
        self._already_called_process_weights_after_loading = True

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def shared_experts(self) -> SharedExperts | None:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_internal_router(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def _quant_method(self) -> FusedMoEMethodBase:
        raise NotImplementedError

    # Temporary hack
    @abstractmethod
    def _replace_quant_method(self, quant_method: FusedMoEMethodBase):
        raise NotImplementedError

    ########################################################################
    #
    # FusedMoE layer methods
    #
    ########################################################################

    @abstractmethod
    def maybe_init_modular_kernel(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def layer_id(self):
        raise NotImplementedError

    #
    # Attributes still needed by models
    #

    @property
    @abstractmethod
    def is_monolithic(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def activation(self) -> MoEActivation:
        raise NotImplementedError

    #
    # Expert maps
    #

    @property
    @abstractmethod
    def expert_placement_strategy(self) -> ExpertPlacementStrategy:
        raise NotImplementedError

    @property
    @abstractmethod
    def expert_global_to_physical(self) -> torch.Tensor | None:
        raise NotImplementedError

    @property
    @abstractmethod
    def expert_physical_to_global(self) -> torch.Tensor | None:
        raise NotImplementedError

    @property
    @abstractmethod
    def expert_local_to_global(self) -> torch.Tensor | None:
        raise NotImplementedError

    @property
    @abstractmethod
    def expert_map(self) -> torch.Tensor | None:
        raise NotImplementedError

    @abstractmethod
    def _maybe_init_expert_routing_tables(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        raise NotImplementedError

    @abstractmethod
    def update_expert_map(self):
        raise NotImplementedError

    @abstractmethod
    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_expert_weights(self) -> Iterable[torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def set_eplb_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        raise NotImplementedError
