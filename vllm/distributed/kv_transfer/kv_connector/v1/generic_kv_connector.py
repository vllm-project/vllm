# SPDX-License-Identifier: Apache-2.0
import importlib
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.request import Request

logger = init_logger(__name__)


class GenericKVConnector(KVConnectorBase_V1):

    class ExtraConfigKeys:
        CONNECTOR_IMPL_MODULE_PATH = "connector_impl_module_path"
        CONNECTOR_IMPL_CLASS_NAME = "connector_impl_class_name"

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        transfer_config = vllm_config.kv_transfer_config
        connector_impl_module_path = transfer_config.get_from_extra_config(
            self.ExtraConfigKeys.CONNECTOR_IMPL_MODULE_PATH, None)
        connector_impl_class_name = transfer_config.get_from_extra_config(
            self.ExtraConfigKeys.CONNECTOR_IMPL_CLASS_NAME, None)
        connector_impl_module = importlib.import_module(
            connector_impl_module_path)
        connector_impl = getattr(connector_impl_module,
                                 connector_impl_class_name)
        self._connector_impl = connector_impl(vllm_config, role, self)

    # ==============================
    # Worker-side methods
    # ==============================
    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        """ See KVConnectorBase_V1.start_load_kv."""
        self._connector_impl.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """ See KVConnectorBase_V1.wait_for_layer_load."""
        self._connector_impl.wait_for_layer_load(layer_name)

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """ See KVConnectorBase_V1.save_kv_layer."""
        self._connector_impl.save_kv_layer(layer_name, kv_layer, attn_metadata,
                                           **kwargs)

    def wait_for_save(self):
        """ See KVConnectorBase_V1.wait_for_save."""
        self._connector_impl.wait_for_save()

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> int:
        """ See KVConnectorBase_V1.get_num_new_matched_tokens."""
        return self._connector_impl.get_num_new_matched_tokens(
            request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request",
                                 num_external_tokens: int):
        """ See KVConnectorBase_V1.update_state_after_alloc."""
        self._connector_impl.update_state_after_alloc(request,
                                                      num_external_tokens)

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        """ See KVConnectorBase_V1.build_connector_meta."""
        return self._connector_impl.build_connector_meta(scheduler_output)
