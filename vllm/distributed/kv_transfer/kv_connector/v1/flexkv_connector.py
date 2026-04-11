# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.distributed.kv_events import KVCacheEvent
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


# FlexKV is a distributed KV Store and multi-level cache management system for
# ultra-large-scale LLM inference.
# GitHub: https://github.com/taco-project/FlexKV
# Install: git clone git@github.com:taco-project/FlexKV.git \
#          && cd FlexKV && bash build.sh
class FlexKVConnectorV1(KVConnectorBase_V1):
    """KV Connector that offloads KV cache to FlexKV.

    FlexKV is a distributed KV Store and multi-level cache management system
    designed for ultra-large-scale LLM inference. It supports offloading KV
    cache to CPU memory, SSD, and remote storage.

    Installation:
        See https://github.com/taco-project/FlexKV for installation instructions.
        Quick start::

            git clone git@github.com:taco-project/FlexKV.git
            cd FlexKV && bash build.sh

    Configuration:
        Pass ``kv_connector="FlexKVConnectorV1"`` via ``--kv-transfer-config``::

            --kv-transfer-config \
            '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}'
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        try:
            from flexkv.integration.vllm.vllm_v1_adapter import FlexKVConnectorV1Impl
        except ImportError as e:
            raise ImportError(
                "FlexKV is not installed. Please install it to use "
                "FlexKVConnectorV1. See https://github.com/taco-project/FlexKV "
                "for installation instructions."
            ) from e

        self._flexkv_connector = FlexKVConnectorV1Impl(vllm_config, role)

    def shutdown(self):
        self._flexkv_connector.shutdown()

    # ==============================
    # Worker-side methods
    # ==============================
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """No-op for FlexKV (currently).

        FlexKV manages all KV transfers on the **scheduler side** via
        ``build_connector_meta`` (which calls ``launch_tasks``) and
        ``update_connector_output`` (which polls ``query_finished_task``).
        KV blocks are transferred directly between the FlexKV server and
        vLLM's GPU memory without worker-side intervention during the
        forward pass — similar to how NIXL operates.

        These worker-side hooks are kept (rather than omitted) to satisfy
        the ``KVConnectorBase_V1`` interface contract and to serve as
        extension points for a future worker-side layer-pipelining path.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs (Any): additional arguments (unused).
        """
        self._flexkv_connector.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """No-op for FlexKV (currently).

        FlexKV manages all KV transfers on the scheduler side.
        This hook is retained for ``KVConnectorBase_V1`` API compatibility.

        Args:
            layer_name: the name of the layer (unused).
        """
        self._flexkv_connector.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """No-op for FlexKV (currently).

        FlexKV offloads KV cache asynchronously from the scheduler side
        after a request finishes (see ``request_finished``).  It does not
        intercept individual layer tensors during the forward pass.

        This hook is retained to satisfy ``KVConnectorBase_V1`` and as an
        extension point for future per-layer async offload support.

        Args:
            layer_name (str): the name of the layer (unused).
            kv_layer (torch.Tensor): the paged KV buffer (unused).
            attn_metadata (AttentionMetadata): the attention metadata (unused).
            **kwargs (Any): additional arguments (unused).
        """
        self._flexkv_connector.save_kv_layer(
            layer_name, kv_layer, attn_metadata, **kwargs
        )

    def wait_for_save(self):
        """No-op for FlexKV (currently).

        KV offload tasks are tracked asynchronously by the scheduler
        connector via ``request_finished`` / ``query_finished_task``.
        There is no pending worker-side save to wait for at
        forward-context exit.

        Retained to satisfy ``KVConnectorBase_V1`` and as an extension
        point for future worker-side save-completion signalling.
        """
        self._flexkv_connector.wait_for_save()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Notify worker-side connector of requests that have finished
        generating tokens.

        Returns:
            Tuple of (sending/saving ids, recving/loading ids) for requests
            that have finished asynchronous transfer. The finished saves/sends
            req ids must belong to a set provided in a call to this method
            (this call or a prior one).
        """
        return self._flexkv_connector.get_finished(finished_req_ids)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Initialize with the KV caches. Useful for pre-registering the
        KV caches in the KVConnector (e.g. for NIXL).

        Args:
            kv_caches: dictionary of layer names to kv cache tensors.
        """
        self._flexkv_connector.register_kv_caches(kv_caches)

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """Get the number of new tokens that can be loaded from the
        external KV cache beyond ``num_computed_tokens``.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally computed
                tokens for this request.

        Returns:
            Tuple of (num_external_tokens, is_ready) where
            num_external_tokens is the number of additional tokens that
            can be loaded from the external KV cache.
        """
        return self._flexkv_connector.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
        num_computed_tokens: int | None = None,
    ):
        """Update KVConnector state after block allocation."""
        self._flexkv_connector.update_state_after_alloc(
            request, blocks, num_external_tokens, num_computed_tokens
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        return self._flexkv_connector.build_connector_meta(scheduler_output)

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        self._flexkv_connector.update_connector_output(connector_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Called when a request has finished, before its blocks are freed.

        Returns:
            Tuple of (async_save, kv_transfer_params) where async_save is
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            :meth:`get_finished`. kv_transfer_params is an optional dict of
            KVTransferParams to be included in the request outputs.
        """
        return self._flexkv_connector.request_finished(request, block_ids)

    def take_events(self) -> Iterable["KVCacheEvent"]:
        """Collect buffered KV cache events.

        Returns:
            New KV cache events since the last call.
        """
        return self._flexkv_connector.take_events()

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """Get the KV connector stats collected during the last interval."""
        return self._flexkv_connector.get_kv_connector_stats()

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Get the block ids that have failed to load."""
        return self._flexkv_connector.get_block_ids_with_load_errors()
