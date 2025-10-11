# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from typing import Any


class BaseAdaptor(ABC):
    """
    Abstract base class for Expert Parallel Load Balancer (EPLB) adaptors.

    This class defines the interface required for coordination with EPLB,
    including obtaining workloads, managing expert maps, and updating
    expert weights. Specific adaptor implementations (e.g., for vLLM)
    should inherit from this base class and implement all abstract methods.
    """

    @abstractmethod
    def __init__(self, **args):
        """
        Initializes the adaptor.

        Args:
            **args: Any additional initialization arguments.
        """
        pass

    @abstractmethod
    def get_rank_expert_workload(self):
        """
        Abstract method: Retrieves the expert workload statistics for the
        current rank.

        Concrete implementations should return a tensor or other data structure
        representing the workload metrics for MoE layers within the current
        process.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def get_init_expert_map(self, num_moe_layers: Any) -> Any:
        """
        Abstract method: Collects the initial expert mappings across all ranks.

        Concrete implementations should return a tensor or other data structure
        representing the global expert map.

        Args:
            num_moe_layers: The number of MoE layers to process.

        Returns:
            Any: The global expert map.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def do_update_expert_map(self, layer_id: Any, updated_expert_map: Any) -> Any:
        """
        Abstract method: Performs an update of the expert map.

        Concrete implementations should apply the updated expert map to the
        specified MoE layer.

        Args:
            layer_id: The ID of the MoE layer to update.
            updated_expert_map: The tensor or data structure containing the new
                expert map.

        Returns:
            Any: The result of the update operation (if applicable).

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def do_update_expert_weight(
        self, layer_id: Any, local_expert_to_replace: Any, buffer_tensor_id: Any
    ) -> Any:
        """
        Abstract method: Performs an update of expert weights.

        Concrete implementations should copy weights from a specified buffer
        tensor to the target local expert.

        Args:
            layer_id: The ID of the MoE layer containing the expert to update.
                local_expert_to_replace: The local ID of the expert whose
                weights are to be replaced.
            buffer_tensor_id: The ID of the buffer tensor containing the new
                weights.

        Returns:
            Any: The result of the update operation (if applicable).

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError
