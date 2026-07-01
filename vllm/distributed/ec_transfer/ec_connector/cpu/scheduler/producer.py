# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Producer role for the ECCPUConnector — encoding/offload pipeline.

Reserves region space for new encodings, allocates blocks in build_saves(),
and promotes them into the shared local cache so the consumer side can reload
them from the mmap instead of recomputing.
"""

import threading
from math import ceil
from typing import TYPE_CHECKING

from vllm.distributed.ec_transfer.ec_connector.cpu.common import ECRegionContext
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.common import (
    evict_and_alloc,
)
from vllm.distributed.ec_transfer.ec_connector.ec_shared_region import AllocationError
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


class ECCPUProducer:
    """Encoding/offload pipeline for the producer side.

    Runs on the scheduler thread. Shared state (_local_encodings, _blocks) is
    guarded by _lock so a later NIXL subclass can add a background session
    thread without changing these methods.
    """

    def __init__(
        self,
        memory_context: ECRegionContext,
        local_encodings: dict[str, None],
        blocks: dict[str, list[int]],
        lock: threading.Lock,
    ) -> None:
        self._memory_context = memory_context
        self._lock = lock
        self._local_encodings = local_encodings
        self._blocks = blocks
        # mm_hash → size_bytes for pending saves not yet block-allocated.
        self._pending_new_encodings: dict[str, int] = {}
        # mm_hashes whose blocks are allocated but not yet in _local_encodings.
        self._pending_save: set[str] = set()

    def ensure_cache_available(
        self, request: "Request", num_computed_tokens: int
    ) -> bool:
        """Return False if the region cannot accommodate this request's saves.

        Performs a speculative alloc-then-free for each unseen feature. The
        actual allocation happens in build_saves().
        """
        for feature in request.mm_features:
            pos = feature.mm_position
            if pos.offset + pos.length <= num_computed_tokens:
                continue
            mm_hash = feature.mm_hash or feature.identifier
            if mm_hash in self._pending_save or mm_hash in self._pending_new_encodings:
                continue
            with self._lock:
                if mm_hash in self._local_encodings:
                    continue
            size_bytes = (
                pos.length
                * self._memory_context.hidden_dim
                * self._memory_context.element_size
            )
            n_blocks = max(1, ceil(size_bytes / self._memory_context.block_size_bytes))
            try:
                indices = self._fifo_alloc(n_blocks)
                self._memory_context.region.free(indices)
            except AllocationError:
                logger.debug(
                    "EC: producer cannot reserve %d blocks for mm_hash=%s; deferring",
                    n_blocks,
                    mm_hash,
                )
                return False
        return True

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        feature = request.mm_features[index]
        mm_hash = feature.mm_hash or feature.identifier
        if mm_hash in self._pending_save or mm_hash in self._pending_new_encodings:
            return
        with self._lock:
            if mm_hash in self._local_encodings:
                return
        size_bytes = (
            feature.mm_position.length
            * self._memory_context.hidden_dim
            * self._memory_context.element_size
        )
        self._pending_new_encodings[mm_hash] = size_bytes
        logger.debug("EC: save scheduled mm_hash=%s size_bytes=%d", mm_hash, size_bytes)

    def build_saves(self) -> dict[str, list[int]]:
        """Advance the offload pipeline. Returns {mm_hash: block_indices}."""
        to_promote = self._pending_save
        self._pending_save = set()
        with self._lock:
            for mm_hash in to_promote:
                self._local_encodings[mm_hash] = None

        pending_new = list(self._pending_new_encodings.items())
        self._pending_new_encodings = {}
        saves: dict[str, list[int]] = {}
        for mm_hash, size_bytes in pending_new:
            if mm_hash in self._pending_save:
                continue
            with self._lock:
                if mm_hash in self._local_encodings:
                    continue
            n_blocks = max(1, ceil(size_bytes / self._memory_context.block_size_bytes))
            indices = self._fifo_alloc(n_blocks)
            self._blocks[mm_hash] = indices
            self._pending_save.add(mm_hash)
            saves[mm_hash] = indices
            logger.debug("EC: save allocated mm_hash=%s n_blocks=%d", mm_hash, n_blocks)
        return saves

    def shutdown(self) -> None:
        pass

    def _fifo_alloc(self, n_blocks: int) -> list[int]:
        try:
            return self._memory_context.region.alloc(n_blocks)
        except AllocationError:
            pass
        with self._lock:
            result = evict_and_alloc(
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
