# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ECCPUScheduler — CPU offload scheduler delegate.

Owns the mmap region and the embedding cache, and handles the producer
(GPU->CPU offload) and consumer (CPU->GPU reload) scheduler-side logic
for the ECCPUConnector.
"""

from typing import TYPE_CHECKING

from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    ECCPUConnectorMetadata,
    ECCPUWorkerMetadata,
    create_ec_shared_region,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.embedding_cache import (
    EmbeddingCache,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ECConnectorOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


class ECCPUScheduler:
    """Scheduler delegate for the ECCPUConnector."""

    def __init__(self, vllm_config: "VllmConfig") -> None:
        ec_config = vllm_config.ec_transfer_config
        assert ec_config is not None
        self._is_producer: bool = ec_config.is_ec_producer
        self._is_consumer: bool = ec_config.is_ec_consumer

        self._region = create_ec_shared_region(vllm_config)
        # Block allocator + LRU eviction policy for the shared region.
        self._cache = EmbeddingCache(self._region.num_blocks)

        # mm_hash → block IDs allocated this step for GPU→mmap saves.
        self._pending_saves: dict[str, list[int]] = {}
        # mm_hash → block IDs to load from mmap→GPU this step.
        self._pending_loads: dict[str, list[int]] = {}

    def has_cache_item(self, identifier: str) -> bool:
        if not self._is_consumer:
            return False
        entry = self._cache.get(identifier)
        return entry is not None and entry.ready

    def ensure_cache_available(
        self, request: "Request", num_computed_tokens: int
    ) -> bool:
        return True

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        feature = request.mm_features[index]
        mm_hash = feature.identifier

        if self._is_producer and self._cache.get(mm_hash) is None:
            entry = self._cache.alloc(mm_hash, feature.mm_position.length)
            if entry is not None:
                self._pending_saves[mm_hash] = list(entry.block_ids)

        if self._is_consumer and mm_hash not in self._pending_loads:
            entry = self._cache.get(mm_hash)
            if entry is not None and entry.ready:
                self._cache.pin(mm_hash)
                self._pending_loads[mm_hash] = list(entry.block_ids)

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> ECCPUConnectorMetadata:
        meta = ECCPUConnectorMetadata()
        if self._is_producer:
            meta.saves = self._pending_saves
            self._pending_saves = {}
        if self._is_consumer:
            meta.loads = self._pending_loads
            self._pending_loads = {}
        return meta

    def update_connector_output(self, connector_output: "ECConnectorOutput") -> None:
        """Apply the worker's memcpy-completion report to the cache.

        Completed saves become safe to mark ready; completed loads become safe
        to unpin.
        """
        meta = connector_output.ec_connector_worker_meta
        if not isinstance(meta, ECCPUWorkerMetadata):
            return
        for mm_hash in meta.completed_saves:
            entry = self._cache.get(mm_hash)
            if entry is not None and not entry.ready:
                self._cache.mark_ready(mm_hash)
        for mm_hash in meta.completed_loads:
            entry = self._cache.get(mm_hash)
            if entry is not None and not entry.evictable:
                self._cache.unpin(mm_hash)

    def has_pending_push_work(self) -> bool:
        """True while any dispatched save or load has not been confirmed done.

        Keeps the engine stepping so the worker's completion reports are polled
        even when no requests are otherwise runnable.
        """
        return self._cache.has_held_entries()

    def shutdown(self) -> None:
        self._pending_saves.clear()
        self._pending_loads.clear()

        self._is_producer = False
        self._is_consumer = False

        try:
            self._region.cleanup()
        except Exception:
            logger.debug("ec: region cleanup failed", exc_info=True)
