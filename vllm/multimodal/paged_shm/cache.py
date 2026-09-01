# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Paged shared memory cache for multi-modal processing.

This module provides sender (P0) and receiver (P1) cache implementations
that use a shared memory server managed by PagedShmServer.
"""

from collections.abc import Sequence

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
from vllm.multimodal.paged_shm.types import ShmItem, ShmWriteRequest
from vllm.multimodal.processing.processor import ResolvedPromptUpdate
from vllm.utils.cache import CacheInfo
from vllm.utils.torch_utils import DeviceLikeType
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

logger = init_logger(__name__)


class PagedShmCache:
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
                "PagedShmSenderCache requires a valid paged_shm_server_address "
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
        self._decoder = MsgpackDecoder(ShmItem)

    def is_cached_item(self, mm_hash: str) -> bool:
        """Check if an item exists in the shared memory cache."""
        try:
            self._client.get_info(mm_hash)
            return True
        except RuntimeError:
            return False

    def invalidate(self, mm_hash: str) -> None:
        """Delete the item from SHM."""
        try:
            self._client.delete(mm_hash)
        except Exception as e:
            logger.debug("Failed to invalidate %s: %s", mm_hash, e)

    def get_item(self, mm_hash: str):
        try:
            alloc = self._client.open_read(mm_hash, timeout=self.open_write_timeout)
            try:
                kwargs_item, prompt_updates = read_decoded_from_blocks(
                    self._storage,
                    alloc.blocks,
                    self.block_size,
                    self._decoder,
                    self.device,
                )
                return kwargs_item, prompt_updates
            finally:
                assert alloc.read_token is not None
                self._client.close_read(alloc.read_token)
        except Exception:
            # Any error (item missing, timeout, etc.) is treated as a miss.
            raise MultiModalCacheMissError([mm_hash]) from None

    def create_item(
        self,
        kwargs_item: MultiModalKwargsItem,
        prompt_updates: Sequence[ResolvedPromptUpdate],
        mm_hash: str,
    ) -> MultiModalProcessorCacheOutItem:
        encoded = encode_item(kwargs_item, prompt_updates, self._encoder)

        if encoded is None:
            # No tensor larger than block_size, so no SHM transfer needed.
            # Return the original item directly.
            return kwargs_item, prompt_updates

        chunks, lengths = encoded

        # Calculate total blocks needed (each chunk starts at a new block)
        total_blocks = sum(
            (length + self.block_size - 1) // self.block_size for length in lengths
        )

        req = ShmWriteRequest(
            uuid=mm_hash,
            size=total_blocks * self.block_size,
            use_cache=True,
            generate_read_token=True,
        )

        try:
            alloc = self._client.open_write([req], timeout=self.open_write_timeout)[0]
        except RuntimeError as e:
            logger.error("PagedShm `open_write` failed: %s", e)
            return kwargs_item, prompt_updates

        self._executor.submit(self._async_write_task, alloc.uuid, chunks, alloc.blocks)
        # Signal that the item is stored in SHM (cache hit for future reads)
        return None, prompt_updates

    def _async_write_task(
        self,
        uuid: str,
        chunks: Sequence[bytes | torch.Tensor],
        blocks: list[int],
    ) -> None:
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

    def close(self) -> None:
        self._client.close()


class PagedShmSenderCache(PagedShmCache, BaseMultiModalProcessorCache):
    """Sender (P0) cache that stores multi-modal items in paged shared memory."""

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

    def get_and_update_item(
        self,
        mm_item: MultiModalProcessorCacheInItem,
        mm_hash: str,
    ) -> MultiModalProcessorCacheOutItem:
        self._total += 1
        if mm_item is not None:
            self._hits += 1
            return self.create_item(mm_item[0], mm_item[1], mm_hash)
        return self.get_item(mm_hash)

    def touch_sender_cache_item(self, mm_hash: str) -> None:
        pass

    def clear_cache(self) -> None:
        self._hits = 0
        self._total = 0
        self._last_info = CacheInfo(hits=0, total=0)

    def make_stats(self, *, delta: bool = False) -> CacheInfo:
        return self._stat(delta=delta)


class PagedShmReceiverCache(PagedShmCache, BaseMultiModalReceiverCache):
    """
    Receiver (P1) cache that reads multi-modal items from paged shared memory.

    It only reads data; it does not write. If mm_item is None, it reads from SHM.
    If mm_item is not None, it returns mm_item as is (no caching performed).
    """

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
        # If data is provided, return it as-is (no caching)
        if mm_item is not None:
            return mm_item
        kwargs_item, _ = self.get_item(mm_hash)
        return kwargs_item

    def touch_receiver_cache_item(
        self,
        mm_hash: str,
        mm_item: MultiModalKwargsItem | None = None,
    ) -> None:
        pass

    def clear_cache(self) -> None:
        pass
