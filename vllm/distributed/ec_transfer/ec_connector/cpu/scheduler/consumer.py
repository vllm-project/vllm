# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consumer role for the ECCPUConnector — local mmap reload.

On a cache hit the consumer pins the producer-allocated blocks and re-serves
them to the worker, which copies mmap→GPU. It never allocates region blocks
itself; the producer owns allocation.
"""

import threading
from typing import TYPE_CHECKING

from vllm.distributed.ec_transfer.ec_connector.cpu.common import ECRegionContext
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


class ECCPUConsumer:
    """Serves encoder cache from the local mmap region.

    _local_encodings and _blocks are shared with the producer (ec_both) and
    guarded by _lock so a later NIXL subclass can add a background thread
    without changing these methods.
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
        # Locally cached mm_hashes pinned for mmap->GPU re-copy this step.
        self._pending_reload: set[str] = set()

    def has_cache_item(self, identifier: str) -> bool:
        return identifier in self._local_encodings

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        feature = request.mm_features[index]
        # Key on identifier — the encoder-output cache key vLLM uses
        # everywhere (has_cache_item, GPU encoder_cache, worker save/load).
        mm_hash = feature.identifier
        with self._lock:
            if mm_hash in self._local_encodings:
                if mm_hash not in self._pending_reload:
                    self._memory_context.region.pin(self._blocks[mm_hash])
                self._pending_reload.add(mm_hash)

    def ensure_cache_available(
        self, request: "Request", num_computed_tokens: int
    ) -> bool:
        """CPU offload never blocks: cached items are pinned in
        update_state_after_alloc; uncached items are encoded normally."""
        return True

    def build_loads(self) -> dict[str, list[int]]:
        """Re-serve reloads and drop this step's pins."""
        loads: dict[str, list[int]] = {}
        for mm_hash in self._pending_reload:
            if mm_hash in self._local_encodings:
                loads[mm_hash] = self._blocks[mm_hash]
                logger.debug("EC: local mmap re-serve mm_hash=%s", mm_hash)
            self._memory_context.region.unpin(self._blocks[mm_hash])
        self._pending_reload = set()
        return loads

    def shutdown(self) -> None:
        for mm_hash in self._pending_reload:
            blocks = self._blocks.get(mm_hash)
            if blocks is not None:
                self._memory_context.region.unpin(blocks)
        self._pending_reload = set()
