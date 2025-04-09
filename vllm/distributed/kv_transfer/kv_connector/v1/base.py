# SPDX-License-Identifier: Apache-2.0
"""
KVConnectorBase_V1 Class for Distributed KV Cache & Hidden State
communication in vLLM v1

The class provides the following primitives:
"""

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.request import Request


class KVConnectorRole(enum.Enum):
    # Connector running in the scheduler process
    SCHEDULER = 0

    # Connector running in the worker process
    WORKER = 1


@dataclass
class KVConnectorMetadata:
    pass


class KVConnectorBase_V1(ABC):

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        self._connector_metadata = KVConnectorMetadata()
        self._vllm_config = vllm_config
        self._role = role

    @property
    def role(self) -> KVConnectorRole:
        return self._role

    def bind_connector_metadata(
            self, connector_metadata: KVConnectorMetadata) -> None:
        """Set the connector metadata from the scheduler.

        This function should be called by the model runner every time 
        before the model execution. The metadata will be used for runtime
        KV cache loading and saving.

        Args:
            connector_metadata (dict): the connector metadata.
        """
        self._connector_metadata = connector_metadata

    def clear_connector_metadata(self) -> None:
        """Clear the connector metadata.

        This function should be called by the model runner every time 
        after the model execution.
        """
        self._connector_metadata = KVConnectorMetadata()

    def _get_connector_metadata(self) -> KVConnectorMetadata:
        """Get the connector metadata.

        This function should only be called inside the connector.

        Returns:
            ConnectorMetadata: the connector metadata.
        """
        return self._connector_metadata

    # ==============================
    # Worker-side methods
    # ==============================

    @abstractmethod
    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        """
        Start loading the KV cache from the connector buffer to vLLM's 
        paged KV buffer. This is called from the forward context before
        the forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be 
            the same.
            
        """
        pass

    @abstractmethod
    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.
        
        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        pass

    @abstractmethod
    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """
        Start saving the a layer of KV cache from vLLM's paged buffer 
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current 
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        pass

    @abstractmethod
    def wait_for_save(self):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        pass

    # ==============================
    # Scheduler-side methods
    # ==============================
    @abstractmethod
    def get_external_prefix_cache_blocks(
        self,
        request: "Request",
        computed_blocks: list["KVCacheBlock"],
        num_computed_tokens: int,
        kv_cache_manager: "KVCacheManager",
    ) -> list["KVCacheBlock"]:
        """
        Get the external prefix cache blocks from the connector.

        This function may change the state of the connector, which will
        be used by `attach_connector_meta` later.

        This function will also allocate/free the blocks dynamically when  
        there is remote cache hit.

        Args:
            request (Request): the request object.
            computed_blocks (list[KVCacheBlock]): the 'local' computed blocks.
            num_computed_tokens (int): the number of 'local' computed tokens.
            kv_cache_manager (KVCacheManager): the KV cache manager to 
                allocate/free the blocks if needed.

        Returns:
            The updated list of the computed blocks (appended with the remote
            cached blocks)
        """
        pass

    @abstractmethod
    def attach_connector_meta(
            self, scheduler_output: SchedulerOutput) -> SchedulerOutput:
        """
        Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output 
        except the `connector_metadata` field.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        pass
