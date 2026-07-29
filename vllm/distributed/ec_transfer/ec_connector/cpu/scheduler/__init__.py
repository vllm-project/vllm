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
    create_ec_shared_region,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.embedding_cache import (
    EmbeddingCache,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.step_tracker import (
    StepTracker,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
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

        max_batches = vllm_config.max_concurrent_batches
        # Delays mark_ready until the GPU→mmap DMA is guaranteed complete.
        self._ready_tracker = StepTracker(max_batches)
        # Delays unpin until the mmap→GPU DMA is guaranteed complete.
        self._unpin_tracker = StepTracker(max_batches)

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
                self._ready_tracker.add(mm_hash, request.request_id)

        if self._is_consumer and mm_hash not in self._pending_loads:
            entry = self._cache.get(mm_hash)
            if entry is not None and entry.ready:
                self._cache.pin(mm_hash)
                self._pending_loads[mm_hash] = list(entry.block_ids)
                self._unpin_tracker.add(mm_hash, request.request_id)

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> ECCPUConnectorMetadata:
        finished = (
            scheduler_output.finished_req_ids if scheduler_output is not None else set()
        )

        for key in self._ready_tracker.step(finished):
            entry = self._cache.get(key)
            if entry is not None and not entry.ready:
                self._cache.mark_ready(key)

        for key in self._unpin_tracker.step(finished):
            self._cache.unpin(key)

        meta = ECCPUConnectorMetadata()
        if self._is_producer:
            meta.saves = self._pending_saves
            self._pending_saves = {}
        if self._is_consumer:
            meta.loads = self._pending_loads
            self._pending_loads = {}
        return meta

    def shutdown(self) -> None:
        # drain_all() covers both entries still in _current (never
        # consumed by build_connector_meta) and entries in slots.
        self._pending_loads.clear()
        for mm_hash in self._unpin_tracker.drain_all():
            self._cache.unpin(mm_hash)
        self._ready_tracker.drain_all()

        self._is_producer = False
        self._is_consumer = False

        try:
            self._region.cleanup()
        except Exception:
            logger.debug("ec: region cleanup failed", exc_info=True)
