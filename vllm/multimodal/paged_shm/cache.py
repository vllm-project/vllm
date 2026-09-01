# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Paged shared memory cache for multi-modal processing.

**Workflow:**
- **Sender**: Encodes item into MessagePack chunks. If multiple chunks (large tensor),
  allocates SHM blocks and writes asynchronously. Returns `(None, updates)` to
  indicate data is now in cache.
- **Receiver**: On cache miss (`mm_item=None`), reads and decodes from SHM;
  on hit, returns item directly.

**Classes:**
- PagedShmCache: Base class handling client, encoder/decoder, and common ops.
- PagedShmSenderCache: Implements BaseMultiModalProcessorCache for writing.
- PagedShmReceiverCache: Implements BaseMultiModalReceiverCache for reading.
"""

import threading
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.multimodal.cache import (
    BaseMultiModalProcessorCache,
    BaseMultiModalReceiverCache,
    MultiModalCacheMissError,
    MultiModalProcessorCacheInItem,
    MultiModalProcessorCacheOutItem,
)
from vllm.multimodal.inputs import MultiModalKwargsItem
from vllm.multimodal.paged_shm.client import PagedShmClient
from vllm.multimodal.paged_shm.serial_utils import (
    encode_item,
    read_decoded_from_blocks,
    write_encoded_to_blocks,
)
from vllm.multimodal.paged_shm.types import PagedShmCacheOutItem, ShmWriteRequest
from vllm.utils.cache import CacheInfo
from vllm.utils.torch_utils import DeviceLikeType
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

logger = init_logger(__name__)


@dataclass
class CacheStats:
    """
    Thread-safe statistics collector for cache operations.
    Maintains hit count, total accesses, and supports incremental reporting.
    """

    _hits: int = 0
    _total: int = 0
    _last_info: CacheInfo = CacheInfo(hits=0, total=0)
    _lock: threading.Lock = threading.Lock()

    def record_access(self, is_hit: bool) -> None:
        """Record one cache access; increment total and optionally hits."""
        with self._lock:
            self._total += 1
            if is_hit:
                self._hits += 1

    def reset(self) -> None:
        """Reset all counters to zero."""
        with self._lock:
            self._hits = 0
            self._total = 0
            self._last_info = CacheInfo(hits=0, total=0)

    def make_stats(self, delta: bool = False) -> CacheInfo:
        """
        Return current statistics. If delta=True, return the increment since the
        last call to make_stats(delta=True) (or since last reset).
        """
        with self._lock:
            current = CacheInfo(hits=self._hits, total=self._total)
            if delta:
                diff = current - self._last_info
                self._last_info = current
                return diff
            return current


class PagedShmCache:
    """Base class for SHM‑backed multi‑modal caches."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        open_write_timeout: float = 5.0,
        pool_workers: int = 1,
        pin: bool = False,
        device: DeviceLikeType = "cpu",
    ) -> None:
        mm_config = vllm_config.model_config.get_multimodal_config()
        address = mm_config.paged_shm_server_address
        if not address:
            raise RuntimeError(
                "PagedShmCache requires a valid paged_shm_server_address "
                "in the model's multimodal config."
            )

        self._client = PagedShmClient(
            address=address, pin=pin, pool_workers=pool_workers
        )
        self._storage = self._client._storage
        self._executor = self._client._executor

        self.block_size = mm_config.paged_shm_block_size
        self.open_write_timeout = open_write_timeout
        self.device = device

        self._encoder = MsgpackEncoder(
            size_threshold=self.block_size, save_raw_tensor=True
        )
        self._decoder = MsgpackDecoder(PagedShmCacheOutItem)
        self.stream: torch.Stream = nullcontext() if not pin else torch.cuda.Stream()

    def is_cached_item(self, mm_hash: str) -> bool:
        """Check if an item exists in the shared memory cache."""
        try:
            self._client.get_info(mm_hash)
            return True
        except RuntimeError:
            return False

    def invalidate(self, mm_hash: str) -> None:
        """Delete the item from shared memory."""
        try:
            self._client.delete(mm_hash)
        except Exception as e:
            logger.debug("Failed to invalidate %s: %s", mm_hash, e)

    def create_item(
        self,
        mm_item: MultiModalProcessorCacheInItem,
        mm_hash: str,
    ) -> MultiModalProcessorCacheOutItem:
        """
        Encode and write the item to shared memory.

        Steps:
        1. Encode the item into chunks;
            if only one chunk, skip SHM and return original.
        2. Calculate total physical blocks needed
            (each logical chunk starts at a new block).
        3. Request block allocation from the SHM server.
        4. Submit an async task to write chunks and close the write handle.
        5. Immediately return `(None, updates)` indicating the item is now cached.
        """
        encoded = encode_item(mm_item, self._encoder)

        if encoded is None:
            # No tensor larger than block_size, so no SHM transfer needed.
            return mm_item

        chunks, lengths = encoded

        # Each chunk starts at a new block boundary,
        # so total blocks is sum of per‑chunk needs.
        total_blocks = sum(
            (length + self.block_size - 1) // self.block_size for length in lengths
        )

        req = ShmWriteRequest(
            uuid=mm_hash,
            size=total_blocks * self.block_size,
            use_cache=True,
        )

        try:
            alloc = self._client.open_write([req], timeout=self.open_write_timeout)[0]
        except RuntimeError as e:
            logger.error("PagedShm `open_write` failed: %s", e)
            return mm_item

        self._executor.submit(self._async_write_task, alloc.uuid, chunks, alloc.blocks)
        # Signal that the item is stored in SHM (cache hit for future reads)
        return None, mm_item[1]

    def _async_write_task(
        self,
        uuid: str,
        chunks: Sequence[bytes | torch.Tensor],
        blocks: list[int],
    ) -> None:
        """Background task to write chunks and finalize the write handle."""
        try:
            write_encoded_to_blocks(self._storage, chunks, blocks)
            self._client.close_write(uuid)
        except Exception:
            # Rollback: delete the item to free blocks
            try:
                self._client.delete(uuid)
            except Exception as e:
                logger.error(
                    "Failed to clean up blocks for async write uuid %s: %s", uuid, e
                )
            raise  # re-raise original exception

    def get_item(self, mm_hash: str) -> MultiModalProcessorCacheInItem:
        """
        Read and decode an item from shared memory.

        Steps:
        1. Open a read handle for the given hash, retrieving the allocated blocks.
        2. Within a stream context (if CUDA), decode the blocks into Python objects.
        3. Synchronize the stream if it's a CUDA stream to ensure data is ready.
        4. Return the reconstructed item; finally close the read handle.
        """
        try:
            alloc = self._client.open_read(mm_hash, timeout=self.open_write_timeout)
            try:
                with self.stream:
                    mm_item = read_decoded_from_blocks(
                        self._storage,
                        alloc.blocks,
                        self.block_size,
                        self._decoder,
                        self.device,
                    )
                if isinstance(self.stream, torch.cuda.Stream):
                    self.stream.synchronize()
                return mm_item
            finally:
                assert alloc.read_token is not None
                self._client.close_read(alloc.read_token)
        except Exception:
            # Any error (item missing, timeout, etc.) is treated as a miss.
            raise MultiModalCacheMissError([mm_hash]) from None

    def close(self) -> None:
        """Close the underlying client and release resources."""
        self._client.close()


class PagedShmSenderCache(PagedShmCache, BaseMultiModalProcessorCache):
    """Sender (P0) cache."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        open_write_timeout: float = 5.0,
        pool_workers: int = 1,
    ) -> None:
        PagedShmCache.__init__(
            self,
            vllm_config,
            open_write_timeout=open_write_timeout,
            pool_workers=pool_workers,
        )
        self._stats = CacheStats()

    def get_and_update_item(
        self,
        mm_item: MultiModalProcessorCacheInItem,
        mm_hash: str,
    ) -> MultiModalProcessorCacheOutItem:
        """
        Update the cache with the given item and return the cached representation.

        If `mm_item` is provided, we treat it as a cache hit, record stats, and
        store it. Otherwise, we treat it as a miss and read from SHM (should not
        happen for sender).
        """
        if mm_item is not None:
            self._stats.record_access(is_hit=True)
            return self.create_item(mm_item, mm_hash)
        else:
            self._stats.record_access(is_hit=False)
            return self.get_item(mm_hash)

    def touch_sender_cache_item(self, mm_hash: str) -> None:
        """No‑op for sender; items are already in SHM when created."""
        pass

    def clear_cache(self) -> None:
        """Reset statistics; actual SHM data is managed by the server."""
        self._stats.reset()

    def make_stats(self, *, delta: bool = False) -> CacheInfo:
        """Return cache statistics (hits/total)."""
        return self._stats.make_stats(delta=delta)


class PagedShmReceiverCache(PagedShmCache, BaseMultiModalReceiverCache):
    """Receiver (P1) cache."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        pool_workers: int = 1,
        pin: bool = True,
        device: DeviceLikeType = "cuda",
    ) -> None:
        PagedShmCache.__init__(
            self, vllm_config, pin=pin, pool_workers=pool_workers, device=device
        )

    def get_and_update_item(
        self,
        mm_item: MultiModalKwargsItem | None,
        mm_hash: str,
    ) -> MultiModalKwargsItem:
        """
        Return the cached item.

        If `mm_item` is not None, return it directly (cache hit).
        Otherwise, read from SHM and return the reconstructed kwargs item.
        """
        if mm_item is not None:
            return mm_item
        kwargs_item, _ = self.get_item(mm_hash)
        return kwargs_item

    def touch_receiver_cache_item(
        self,
        mm_hash: str,
        mm_item: MultiModalKwargsItem | None = None,
    ) -> None:
        """No‑op for receiver; items are already valid when read."""
        pass

    def clear_cache(self) -> None:
        """No‑op; receiver does not manage cache data."""
        pass
