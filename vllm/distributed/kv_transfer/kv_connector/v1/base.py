# SPDX-License-Identifier: Apache-2.0
"""
KVConnectorBase_V1 Class for Distributed KV Cache & Hidden State
communication in vLLM v1

The class provides the following primitives:
    Scheduler-side: runs in the scheduler, binds metadata, which
    is used by the worker-side to load/save KV cache.
        get_num_new_matched_tokens() - get number of new tokens 
            that exist in the remote KV cache
        update_state_after_alloc() - update KVConnector state after
            temporary buffer alloc by the CacheManager.

    Worker-side: runs in each worker, loads/saves KV cache to/from
    the Connector based on the metadata.
        start_load_kv() - starts loading all KVs (maybe async)
        wait_for_layer_load() - blocks until layer i load is done

        save_kv_layer() - starts saving KV for layer i (maybe async)
        wait_for_save() - blocks until all saves are done
"""

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

import msgspec
import torch
from pydantic_core import core_schema

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class KVTransferFinishedResult:
    """Result of KV transfer get_finished operation."""

    finished_sending: set[str]
    finished_recving: set[str]
    pending_handshake: set[str]

    def has_any_finished(self) -> bool:
        """Check if any requests finished or are pending."""
        return bool(self.finished_sending or self.finished_recving
                    or self.pending_handshake)

    def is_empty(self) -> bool:
        """Check if all sets are empty."""
        return not self.has_any_finished()

    def get_all_finished_req_ids(self) -> set[str]:
        """Get all request IDs that have finished (sending or receiving)."""
        return self.finished_sending.union(self.finished_recving)

    def merge(self,
              other: 'KVTransferFinishedResult') -> 'KVTransferFinishedResult':
        """Merge with another result, combining all sets."""
        return KVTransferFinishedResult(
            finished_sending=self.finished_sending.union(
                other.finished_sending),
            finished_recving=self.finished_recving.union(
                other.finished_recving),
            pending_handshake=self.pending_handshake.union(
                other.pending_handshake))

    @classmethod
    def empty(cls) -> 'KVTransferFinishedResult':
        """Create an empty result."""
        return cls(finished_sending=set(),
                   finished_recving=set(),
                   pending_handshake=set())

    @classmethod
    def from_tuple(
        cls, result_tuple: tuple[set[str], set[str], set[str]]
    ) -> 'KVTransferFinishedResult':
        """Create from the old tuple format for backward compatibility."""
        finished_sending, finished_recving, pending_handshake = result_tuple
        return cls(finished_sending=finished_sending,
                   finished_recving=finished_recving,
                   pending_handshake=pending_handshake)

    def to_tuple(self) -> tuple[set[str], set[str], set[str]]:
        """Convert to the old tuple format for backward compatibility."""
        return (
            self.finished_sending,
            self.finished_recving,
            self.pending_handshake,
        )


class KVConnectorRole(enum.Enum):
    # Connector running in the scheduler process
    SCHEDULER = 0

    # Connector running in the worker process
    WORKER = 1


class KVConnectorMetadata:
    """
    Abstract Metadata used to communicate between the
    Scheduler KVConnector and Worker KVConnector.
    """
    pass


class KVConnectorHandshakeMetadata(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        # required for @cached_property.
        dict=True):
    """
    Metadata optionally used for out of band connector handshake between
    P/D workers.
    """
    connector_type: str = "base"

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: Callable[[Any],
                                                   core_schema.CoreSchema]
    ) -> core_schema.CoreSchema:
        """bridge msgspec.Struct with pydantic for schema generation"""
        return core_schema.no_info_after_validator_function(
            cls, core_schema.dict_schema())


class KVConnectorTransferMetadata(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        dict=True):
    """
    Wrapper for transfer handshake metadata sent between engine and utils.
    """
    tensor_parallel_rank: int
    data_parallel_rank: int
    content: Optional[dict]


class KVConnectorBase_V1(ABC):

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        logger.warning(
            "Initializing KVConnectorBase_V1. This API is experimental and "
            "subject to change in the future as we iterate the design.")
        self._connector_metadata = KVConnectorMetadata()
        self._vllm_config = vllm_config
        self._role = role
        self._handshake_metadata: Optional[KVConnectorHandshakeMetadata] = None

    @property
    def role(self) -> KVConnectorRole:
        return self._role

    # ==============================
    # Worker-side methods
    # ==============================

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

    def get_connector_metadata(self) -> KVConnectorMetadata:
        """Get the connector metadata.

        This function should only be called inside the connector.

        Returns:
            ConnectorMetadata: the connector metadata.
        """
        return self._connector_metadata

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Initialize with the KV caches. Useful for pre-registering the
        KV Caches in the KVConnector (e.g. for NIXL).

        Args: kv_caches:
            dictionary of layer names, kv cache
        """
        return

    @abstractmethod
    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

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
        Start saving a layer of KV cache from vLLM's paged buffer 
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

    def get_finished(self,
                     finished_req_ids: set[str]) -> KVTransferFinishedResult:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            KVTransferFinishedResult containing sets of finished sending,
            finished receiving, and pending handshake request IDs.
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        return KVTransferFinishedResult.empty()

    def get_pending_handshake_req_ids(self) -> Optional[set[str]]:
        """
        Get request IDs that are currently pending handshake completion.
        
        Returns:
            Set of request IDs waiting for handshake, or None if not applicable.
        """
        return None

    def get_handshake_metadata(self) -> Optional[KVConnectorHandshakeMetadata]:
        """
        Get the handshake metadata for the connector.

        Returns:
            KVConnectorHandshakeMetadata: the handshake metadata.
        """
        return self._handshake_metadata

    # ==============================
    # Scheduler-side methods
    # ==============================

    @abstractmethod
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.
        
        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded from the 
                  external KV cache beyond what is already computed.
                - `True` if external KV cache tokens will be loaded
                  asynchronously (between scheduler steps).
        """
        pass

    @abstractmethod
    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        """
        Update KVConnector state after block allocation.
        """
        pass

    @abstractmethod
    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        pass

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        return False, None
