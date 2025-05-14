# SPDX-License-Identifier: Apache-2.0
import copy
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)


class MultiKVConnectorMetadata(tuple[KVConnectorMetadata, ...],
                               KVConnectorMetadata):
    pass


class MultiConnector(KVConnectorBase_V1):
    """
    A wrapper for using multiple KVConnectors at the same time.

    The current logic is:
    - Load KV from the first connector that advertises available tokens from
      get_num_new_matched_tokens(), based on the order in the config.
    - Save to all connectors.
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self._connectors = []
        ktcs = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "connectors")
        assert ktcs is not None
        for ktc in ktcs:
            temp_config = copy.copy(vllm_config)
            temp_config.kv_transfer_config = KVTransferConfig(**ktc)
            self._connectors.append(
                KVConnectorFactory.create_connector_v1(temp_config, role))

        # A mapping from request id to the connector that is assigned to it.
        self._requests_to_connector: dict[str, KVConnectorBase_V1] = {}

        # Keeps track of *additional* remaining async saves (beyond 1) to be
        # finished per request. Not needed for async loads since we only allow
        # a single connector to load.
        self._extra_async_saves: dict[str, int] = {}

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        for c in self._connectors:
            c.register_kv_caches(kv_caches)

    # We must override the base class method here because we need to bind
    # the metadata to each connector in the order of the connectors in the
    # MultiKVConnectorMetadata.
    def bind_connector_metadata(
            self, connector_metadata: KVConnectorMetadata) -> None:
        assert isinstance(connector_metadata, MultiKVConnectorMetadata)
        for c, cm in zip(self._connectors, connector_metadata):
            c.bind_connector_metadata(cm)

    def clear_connector_metadata(self) -> None:
        for c in self._connectors:
            c.clear_connector_metadata()

    # ==============================
    # Worker-side methods
    # ==============================
    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        for c in self._connectors:
            c.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        for c in self._connectors:
            c.wait_for_layer_load(layer_name)

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        for c in self._connectors:
            c.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self):
        for c in self._connectors:
            c.wait_for_save()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        finished_recving: set[str] = set()
        finished_sending: set[str] = set()
        for c in self._connectors:
            recving, sending = c.get_finished(finished_req_ids)
            if not recving and not sending:
                continue
            # Aggregate finished recving request ids.
            finished_recving.update(recving or ())
            # Aggregate finished sending request ids - only include
            # once we've drained the "extra" count (for cases where
            # more than one connector is async-saving the same request).
            for req_id in sending or ():
                extra_pending = self._extra_async_saves.get(req_id)
                if extra_pending is None:
                    finished_sending.add(req_id)
                    continue
                assert extra_pending > 0
                if extra_pending == 1:
                    del self._extra_async_saves[req_id]
                else:
                    self._extra_async_saves[req_id] = extra_pending - 1

        return finished_recving or None, finished_sending or None

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        for c in self._connectors:
            toks, load_async = c.get_num_new_matched_tokens(
                request, num_computed_tokens)
            # The first connector that has new matched tokens will be assigned
            # to this request.
            if toks > 0:
                self._requests_to_connector[request.request_id] = c
                return toks, load_async
        return 0, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        # If the request is not assigned to any connector, we do nothing.
        if request.request_id not in self._requests_to_connector:
            return
        # We assume that the request is assigned to only one connector.
        c = self._requests_to_connector.pop(request.request_id)
        c.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(
            self,
            scheduler_output: SchedulerOutput) -> MultiKVConnectorMetadata:
        return MultiKVConnectorMetadata(
            c.build_connector_meta(scheduler_output) for c in self._connectors)

    def request_finished(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        async_saves = 0
        kv_txfer_params = None
        for c in self._connectors:
            async_save, txfer_params = c.request_finished(request, blocks)
            if async_save:
                async_saves += 1
            if txfer_params is not None:
                if kv_txfer_params is not None:
                    #TODO we can probably change this to merge the dicts here,
                    # checking for key clashes.
                    raise RuntimeError(
                        "Only one connector can produce KV transfer params")
                kv_txfer_params = txfer_params
        if async_saves > 1:
            self._extra_async_saves[request.request_id] = async_saves - 1
        return async_saves > 0, kv_txfer_params
