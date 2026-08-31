# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Paged shared memory cache for multi-modal processing.

This module provides sender (P0) and receiver (P1) cache implementations
that use a shared memory server managed by PagedShmServer.
"""

from typing import override

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
from vllm.multimodal.paged_shm.types import ShmItem, ShmWriteRequest
from vllm.utils.cache import CacheInfo
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

logger = init_logger(__name__)


class PagedShmSenderCache(BaseMultiModalProcessorCache):
    """
    Sender (P0) cache that stores multi-modal items in paged shared memory.

    All data (including prompt_updates) is stored in shared memory.
    On a cache hit, it reads the entire ShmItem from SHM, returns None for
    the item (so P0 doesn't send data to P1) along with the cached prompt_updates.
    On a miss, it writes the item to SHM and returns the original item
    for transmission to P1.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        open_write_timeout: float = 5.0,
    ) -> None:
        super().__init__()

        mm_config = vllm_config.model_config.get_multimodal_config()
        address = mm_config.paged_shm_server_address
        if not address:
            raise RuntimeError(
                "PagedShmSenderCache requires a valid paged_shm_server_address "
                "in the model's multimodal config."
            )

        self._client = PagedShmClient(address=address, pin=False, pool_workers=4)
        self._storage = self._client._storage
        self.block_size = mm_config.paged_shm_block_size
        self.open_write_timeout = open_write_timeout
        self._encoder = MsgpackEncoder(size_threshold=self.block_size)
        self._decoder = MsgpackDecoder(ShmItem)

        self._hits = 0
        self._total = 0
        self._last_info = CacheInfo(hits=0, total=0)

    def _stat(self, *, delta: bool = False) -> CacheInfo:
        info = CacheInfo(hits=self._hits, total=self._total)
        if delta:
            info_delta = info - self._last_info
            self._last_info = info
            info = info_delta
        return info

    @override
    def is_cached_item(self, mm_hash: str) -> bool:
        """Check if an item exists in the shared memory cache."""
        try:
            self._client.get_info(mm_hash)
            return True
        except RuntimeError:
            return False

    @override
    def get_and_update_item(
        self,
        mm_item: MultiModalProcessorCacheInItem,
        mm_hash: str,
    ) -> MultiModalProcessorCacheOutItem:
        self._total += 1

        # Try to read from SHM directly (no prior is_cached check)
        try:
            alloc = self._client.open_read(mm_hash, timeout=self.open_write_timeout)
            try:
                kwargs_item, prompt_updates = read_decoded_from_blocks(
                    self._storage,
                    alloc.blocks,
                    self.block_size,
                    self._decoder,
                )
                self._hits += 1
                return None, prompt_updates
            finally:
                self._client.close_read(alloc.read_token)
        except Exception:
            # Any error (item missing, timeout, etc.) is treated as a miss.
            pass

        # Cache miss: if no data provided, raise miss error
        if mm_item is None:
            raise MultiModalCacheMissError([mm_hash])

        kwargs_item, prompt_updates = mm_item

        # Encode and write to SHM
        item = ShmItem(kwargs_item=kwargs_item, prompt_updates=prompt_updates)
        meta_block_data, chunks, _ = encode_item(item, self.block_size, self._encoder)

        total_blocks = 1 + sum(
            (len(c) + self.block_size - 1) // self.block_size for c in chunks
        )
        req = ShmWriteRequest(
            uuid=mm_hash,
            size=total_blocks * self.block_size,
            use_cache=True,
            generate_read_token=True,
        )

        try:
            alloc = self._client.open_write([req], timeout=self.open_write_timeout)[0]
            write_encoded_to_blocks(
                self._storage, meta_block_data, chunks, alloc.blocks
            )
            self._client.close_write(mm_hash)
        except (MemoryError, RuntimeError) as e:
            logger.debug("Failed to cache item %s: %s", mm_hash, e)

        # Return the original item (will be sent to P1)
        return kwargs_item, prompt_updates

    @override
    def touch_sender_cache_item(self, mm_hash: str) -> None:
        """Update the cache eviction order for a multi-modal item."""
        try:
            alloc = self._client.open_read(mm_hash, timeout=0.1)
            self._client.close_read(alloc.read_token)
        except Exception:
            pass

    @override
    def clear_cache(self) -> None:
        self._hits = 0
        self._total = 0
        self._last_info = CacheInfo(hits=0, total=0)

    @override
    def invalidate(self, mm_hash: str) -> None:
        """Delete the item from SHM."""
        try:
            self._client.delete(mm_hash)
        except Exception as e:
            logger.debug("Failed to invalidate %s: %s", mm_hash, e)

    @override
    def make_stats(self, *, delta: bool = False) -> CacheInfo:
        return self._stat(delta=delta)

    @override
    def close(self) -> None:
        self._client.close()


class PagedShmReceiverCache(BaseMultiModalReceiverCache):
    """
    Receiver (P1) cache that reads multi-modal items from paged shared memory.

    It only reads data; it does not write. If mm_item is None, it reads from SHM.
    If mm_item is not None, it returns mm_item as is (no caching performed).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        open_read_timeout: float = 5.0,
    ) -> None:
        super().__init__()

        mm_config = vllm_config.model_config.get_multimodal_config()
        address = mm_config.paged_shm_server_address
        if not address:
            raise RuntimeError(
                "PagedShmReceiverCache requires a valid paged_shm_server_address "
                "in the model's multimodal config."
            )

        self._client = PagedShmClient(address=address, pin=False, pool_workers=1)
        self._storage = self._client._storage
        self.block_size = mm_config.paged_shm_block_size
        self.open_read_timeout = open_read_timeout
        self._decoder = MsgpackDecoder(ShmItem)

    @override
    def get_and_update_item(
        self,
        mm_item: MultiModalKwargsItem | None,
        mm_hash: str,
    ) -> MultiModalKwargsItem:
        # If data is provided, return it as-is (no caching)
        if mm_item is not None:
            return mm_item

        # mm_item is None: we must read from SHM
        try:
            alloc = self._client.open_read(mm_hash, timeout=self.open_read_timeout)
            try:
                kwargs_item, _ = read_decoded_from_blocks(
                    self._storage,
                    alloc.blocks,
                    self.block_size,
                    self._decoder,
                )
                return kwargs_item
            finally:
                self._client.close_read(alloc.read_token)
        except Exception as e:
            raise MultiModalCacheMissError([mm_hash]) from e

    @override
    def touch_receiver_cache_item(
        self,
        mm_hash: str,
        mm_item: MultiModalKwargsItem | None = None,
    ) -> None:
        """Update the cache eviction order for a multi-modal item."""
        try:
            alloc = self._client.open_read(mm_hash, timeout=0.1)
            self._client.close_read(alloc.read_token)
        except Exception:
            pass

    @override
    def clear_cache(self) -> None:
        pass
