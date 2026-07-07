# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ECCPUScheduler — CPU offload scheduler delegate.

Owns the mmap region and the offload bookkeeping dicts, and handles the
producer (GPU->CPU offload) and consumer (CPU->GPU reload) scheduler-side
logic directly for the ECCPUConnector.
"""

import threading
from math import ceil
from typing import TYPE_CHECKING

from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    ECCPUConnectorMetadata,
    ECRegionContext,
    setup_ec_region,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
    AllocationError,
    ECSharedRegion,
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

        self._memory_context: ECRegionContext = setup_ec_region(vllm_config)

        # Offload cache: an ec_both instance reuses its own offloaded
        # encodings by serving reloads from _local_encodings / _blocks.
        # _shared_lock guards every access to these two dicts (and the region
        # pin/free calls made against their blocks), so a future NIXL subclass
        # can touch them from a background transfer thread.
        self._local_encodings: dict[str, None] = {}
        self._blocks: dict[str, list[int]] = {}
        self._shared_lock = threading.Lock()

        # Per-step working sets, built and drained on the scheduler thread
        # within a single build_connector_meta; not shared, so not locked.
        # mm_hash -> size_bytes for pending GPU->CPU saves not yet allocated.
        self._encodings_pending_offload: dict[str, int] = {}
        # Locally cached mm_hashes pinned for CPU->GPU re-copy this step.
        self._pending_reload: set[str] = set()

    def has_cache_item(self, identifier: str) -> bool:
        if not self._is_consumer:
            return False
        return identifier in self._local_encodings

    def ensure_cache_available(
        self, request: "Request", num_computed_tokens: int
    ) -> bool:
        return True  # CPU Offloading never blocks.

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        feature = request.mm_features[index]
        mm_hash = feature.identifier
        if self._is_producer:
            self._try_offload(mm_hash, feature.mm_position.length)
        if self._is_consumer:
            self._try_reload(mm_hash)

    def _try_offload(self, mm_hash: str, feature_size: int) -> None:
        if mm_hash in self._encodings_pending_offload:
            return
        with self._shared_lock:
            if mm_hash in self._local_encodings:
                return
        size_bytes = (
            feature_size
            * self._memory_context.hidden_dim
            * self._memory_context.element_size
        )
        self._encodings_pending_offload[mm_hash] = size_bytes
        logger.debug("EC: save scheduled mm_hash=%s size_bytes=%d", mm_hash, size_bytes)

    def _try_reload(self, mm_hash: str) -> None:
        with self._shared_lock:
            if mm_hash in self._local_encodings:
                if mm_hash not in self._pending_reload:
                    self._memory_context.region.pin(self._blocks[mm_hash])
                self._pending_reload.add(mm_hash)

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> ECCPUConnectorMetadata:
        meta = ECCPUConnectorMetadata()
        try:
            if self._is_producer:
                meta.saves.update(self._build_saves())
            if self._is_consumer:
                meta.loads.update(self._build_loads())
        except Exception:
            # Drop this step's reload pins so a failure mid-build does not
            # leak them (and block their eviction) until shutdown.
            if self._is_consumer:
                self._drop_reload_pins()
            raise
        return meta

    def shutdown(self) -> None:
        if self._is_consumer:
            self._drop_reload_pins()
        try:
            self._memory_context.region.cleanup()
        except Exception:
            logger.debug("ec: region cleanup failed", exc_info=True)

    def _drop_reload_pins(self) -> None:
        """Unpin and forget every block pinned for this step's reloads."""
        with self._shared_lock:
            pending = list(self._pending_reload)
            self._pending_reload = set()
        # Pinned entries are stable; unpin outside the lock (region.unpin is
        # independently thread-safe via the region's own lock).
        for mm_hash in pending:
            blocks = self._blocks.get(mm_hash)
            if blocks is not None:
                self._memory_context.region.unpin(blocks)

    def _fifo_alloc(self, n_blocks: int) -> list[int]:
        try:
            return self._memory_context.region.alloc(n_blocks)
        except AllocationError:
            pass
        with self._shared_lock:
            result = _evict_and_alloc(
                n_blocks,
                self._local_encodings,
                self._blocks,
                self._memory_context.region,
                skip_pinned=True,
            )
        if result is not None:
            return result
        raise AllocationError(
            f"ECSharedRegion exhausted: cannot satisfy {n_blocks} blocks"
        )

    def _build_loads(self) -> dict[str, list[int]]:
        """Re-serve reloads and drop this step's pins."""
        with self._shared_lock:
            pending = list(self._pending_reload)
        # Each mm_hash here is pinned this step, so its cache entry cannot be
        # evicted concurrently — the reads below are safe outside the lock.
        loads: dict[str, list[int]] = {}
        for mm_hash in pending:
            if mm_hash in self._local_encodings:
                loads[mm_hash] = self._blocks[mm_hash]
                logger.debug("EC: local mmap re-serve mm_hash=%s", mm_hash)
        self._drop_reload_pins()
        return loads

    def _build_saves(self) -> dict[str, list[int]]:
        """Allocate blocks for pending encodings and promote them.

        Saves are best-effort: if the region cannot accommodate an encoding
        (even after FIFO eviction), the save is skipped and the encoding is
        simply not offloaded. It remains available in the engine's encoder
        cache for this request; a future request with the same mm_hash will
        schedule a fresh save attempt.
        """
        pending_offload = list(self._encodings_pending_offload.items())
        self._encodings_pending_offload = {}
        saves: dict[str, list[int]] = {}
        # Blocks allocated and promoted earlier in this loop are pinned until
        # the loop ends. Each save's _fifo_alloc may fall back to FIFO
        # eviction, which frees existing _local_encodings entries to make
        # room. Without the pin, a later save could evict and reuse the blocks
        # of an earlier save promoted in this same pass, so the worker would
        # copy two encodings into the same blocks (corruption) and the first
        # save would be silently dropped from the cache. Pinning makes those
        # blocks un-evictable (try_free skips them) for the pass; they are
        # unpinned at the end because the guard is only needed against sibling
        # saves here — afterward they are normal evictable cache entries.
        pinned_this_step: list[list[int]] = []
        try:
            for mm_hash, size_bytes in pending_offload:
                with self._shared_lock:
                    if mm_hash in self._local_encodings:
                        continue  # This mm_hash is already offloaded
                n_blocks = max(
                    1, ceil(size_bytes / self._memory_context.block_size_bytes)
                )
                try:
                    indices = self._fifo_alloc(n_blocks)
                except AllocationError:
                    logger.debug(
                        "EC: region full; skipping offload of mm_hash=%s "
                        "(%d blocks needed). Encoding is computed normally "
                        "and not cached.",
                        mm_hash,
                        n_blocks,
                    )
                    continue
                with self._shared_lock:
                    # Assume the worker's offload succeeds so the encoding is
                    # readable next step. Promote and pin atomically: pinning
                    # in the same critical section stops a concurrent evictor
                    # from reclaiming these blocks before they are pinned.
                    self._blocks[mm_hash] = indices
                    self._local_encodings[mm_hash] = None
                    self._memory_context.region.pin(indices)
                pinned_this_step.append(indices)
                saves[mm_hash] = indices
                logger.debug(
                    "EC: save allocated+promoted mm_hash=%s n_blocks=%d",
                    mm_hash,
                    n_blocks,
                )
        finally:
            # Drop the pass-scoped pins, whether the loop finished or raised:
            # the anti-eviction guard is no longer needed once allocation for
            # the step is done, and leaving blocks pinned would make them
            # permanently un-evictable (a region-space leak). The blocks stay
            # allocated and cached — unpin only clears the do-not-evict flag.
            # pinned_this_step is local and region.unpin is independently
            # thread-safe, so no _shared_lock is needed here.
            for indices in pinned_this_step:
                self._memory_context.region.unpin(indices)
        return saves


def _evict_and_alloc(
    n_blocks: int,
    cache: dict[str, None],
    blocks: dict[str, list[int]],
    region: ECSharedRegion,
    *,
    skip_pinned: bool = False,
) -> list[int] | None:
    """Evict `cache` entries in insertion order until `alloc` succeeds.

    ``cache`` is the ordered set of cached mm_hashes (``dict[str, None]``);
    ``blocks`` maps each mm_hash to its allocated block indices.

    ``skip_pinned=True`` uses ``try_free`` so that blocks held by an active
    NIXL READ pin or a ``_pending_reload`` pin are transparently skipped.
    Caller must hold the shared lock when calling this function.
    Returns allocated block list, or None if all candidates were exhausted.
    """
    for mm_hash in list(cache.keys()):
        indices = blocks[mm_hash]
        if skip_pinned:
            if not region.try_free(indices):
                continue
        else:
            region.free(indices)
        del cache[mm_hash]
        del blocks[mm_hash]
        try:
            return region.alloc(n_blocks)
        except AllocationError:
            continue
    return None
