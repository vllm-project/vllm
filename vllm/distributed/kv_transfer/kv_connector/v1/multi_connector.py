# SPDX-License-Identifier: Apache-2.0
import copy
from typing import TYPE_CHECKING

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
    from vllm.v1.request import Request

logger = init_logger(__name__)


class MultiKVConnectorMetadata(tuple[KVConnectorMetadata, ...],
                               KVConnectorMetadata):
    pass


class MultiConnector(KVConnectorBase_V1):

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

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> int:
        for c in self._connectors:
            toks = c.get_num_new_matched_tokens(request, num_computed_tokens)
            # The first connector that has new matched tokens will be assigned
            # to this request.
            if toks > 0:
                self._requests_to_connector[request.request_id] = c
                return toks
        return 0

    def update_state_after_alloc(self, request: "Request",
                                 block_ids: list[int],
                                 num_external_tokens: int):
        # If the request is not assigned to any connector, we do nothing.
        if request.request_id not in self._requests_to_connector:
            return
        # We assume that the request is assigned to only one connector.
        c = self._requests_to_connector.pop(request.request_id)
        c.update_state_after_alloc(request, block_ids, num_external_tokens)

    def build_connector_meta(
            self,
            scheduler_output: SchedulerOutput) -> MultiKVConnectorMetadata:
        return MultiKVConnectorMetadata(
            c.build_connector_meta(scheduler_output) for c in self._connectors)
