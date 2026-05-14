# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from vllm-project/vllm-ascend
# (vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/).
"""MooncakeStoreConnector - KV cache connector using MooncakeDistributedStore.

Unlike MooncakeConnector which does direct P2P transfer, this connector
uses MooncakeDistributedStore as a shared KV cache pool. Both producer
and consumer instances read/write KV to/from the store independently,
enabling prefix caching via hash-based deduplication.
"""

from collections.abc import Iterable
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_events import (
    KVCacheEvent,
    KVConnectorKVEvents,
    KVEventAggregator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from .data import MooncakeStoreConnectorMetadata
from .scheduler import MooncakeStoreScheduler
from .worker import MooncakeStoreWorker

logger = init_logger(__name__)


class MooncakeStoreKVEvents(KVConnectorKVEvents):
    """KV event aggregation for MooncakeStoreConnector."""

    def __init__(self, num_workers: int) -> None:
        self._aggregator = KVEventAggregator(num_workers)

    def add_events(self, events: list[KVCacheEvent]) -> None:
        self._aggregator.add_events(events)

    def aggregate(self) -> "MooncakeStoreKVEvents":
        common_events = self._aggregator.get_common_events()
        self._aggregator.clear_events()
        self._aggregator.add_events(common_events)
        self._aggregator.reset_workers()
        return self

    def increment_workers(self, count: int = 1) -> None:
        self._aggregator.increment_workers(count)

    def get_all_events(self) -> list[KVCacheEvent]:
        return self._aggregator.get_all_events()

    def get_number_of_workers(self) -> int:
        return self._aggregator.get_number_of_workers()

    def clear_events(self) -> None:
        self._aggregator.clear_events()
        self._aggregator.reset_workers()

    def __repr__(self) -> str:
        return f"<MooncakeStoreKVEvents events={self.get_all_events()}>"


class MooncakeStoreConnector(KVConnectorBase_V1):
    """KV connector using MooncakeDistributedStore as shared KV pool."""

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        extra_config = self._kv_transfer_config.kv_connector_extra_config
        return (
            str(extra_config.get("enable_cross_layers_blocks", "False")).lower()
            == "true"
        )

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,  # type: ignore[arg-type]
        )
        assert vllm_config.kv_transfer_config is not None
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self._kv_cache_events: MooncakeStoreKVEvents | None = None

        self.connector_scheduler: MooncakeStoreScheduler | None = None
        self.connector_worker: MooncakeStoreWorker | None = None

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = MooncakeStoreScheduler(vllm_config)
        else:
            self.connector_worker = MooncakeStoreWorker(vllm_config)

    # ============================================================
    # Scheduler-side methods
    # ============================================================

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def update_connector_output(self, connector_output: KVConnectorOutput):
        kv_cache_events = connector_output.kv_cache_events
        if not kv_cache_events or not isinstance(
            kv_cache_events, MooncakeStoreKVEvents
        ):
            return

        if self._kv_cache_events is None:
            self._kv_cache_events = kv_cache_events
        else:
            self._kv_cache_events.add_events(kv_cache_events.get_all_events())
            self._kv_cache_events.increment_workers(
                kv_cache_events.get_number_of_workers()
            )

    def take_events(self) -> Iterable[KVCacheEvent]:
        if self._kv_cache_events is not None:
            self._kv_cache_events.aggregate()
            yield from self._kv_cache_events.get_all_events()
            self._kv_cache_events.clear_events()
            self._kv_cache_events = None

    # ============================================================
    # Worker-side methods
    # ============================================================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type
    ):
        assert self.connector_worker is not None
        self.connector_worker.register_cross_layers_kv_caches(kv_cache)

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        # No-op: loads are issued in get_finished() for compute overlap.
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        # No layerwise support - no-op
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        # No layerwise support - no-op
        return

    def wait_for_save(self):
        # No-op: stores are issued in get_finished() for compute overlap.
        pass

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        assert self.connector_worker is not None
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MooncakeStoreConnectorMetadata)
        return self.connector_worker.get_finished(finished_req_ids, metadata)

    def get_kv_connector_kv_cache_events(
        self,
    ) -> MooncakeStoreKVEvents | None:
        assert self.connector_worker is not None
        events = self.connector_worker.get_kv_events()
        if not events:
            return None

        kv_events = MooncakeStoreKVEvents(num_workers=1)
        kv_events.add_events(events)
        return kv_events
