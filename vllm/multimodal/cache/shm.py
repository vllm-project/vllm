# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Implementation of Shared Memory Cache (see docs/configuration/optimization.md)."""

from collections.abc import Sequence
from multiprocessing.synchronize import Lock as LockType
from typing import TYPE_CHECKING, cast

from typing_extensions import override

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.device_communicators.shm_object_storage import (
    MsgpackSerde,
    SingleWriterShmObjectStorage,
    SingleWriterShmRingBuffer,
)
from vllm.logger import init_logger
from vllm.utils.cache import CacheInfo
from vllm.utils.mem_constants import GiB_bytes, MiB_bytes

from ..inputs import (
    MultiModalBatchedField,
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
)
from .base import (
    BaseMultiModalProcessorCache,
    BaseMultiModalReceiverCache,
    MultiModalProcessorCacheInItem,
    MultiModalProcessorCacheOutItem,
)

if TYPE_CHECKING:
    from ..processing.processor import ResolvedPromptUpdate

logger = init_logger(__name__)


class ShmObjectStoreSenderCache(BaseMultiModalProcessorCache):
    """The cache which is used on P0 when SHM caching is enabled.

    How to update each item:

    - If the item is already in the cache, clear the input to avoid
      unnecessary IPC.

    - If the item is not in the cache, store the data in shared memory.
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        super().__init__()

        self.world_size = vllm_config.parallel_config.world_size
        mm_config = vllm_config.model_config.get_multimodal_config()

        ring_buffer = SingleWriterShmRingBuffer(
            data_buffer_size=int(mm_config.mm_processor_cache_gb * GiB_bytes),
            name=envs.VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME,
            create=True,  # sender is the writer
        )
        self._shm_cache = SingleWriterShmObjectStorage(
            max_object_size=mm_config.mm_shm_cache_max_object_size_mb * MiB_bytes,
            n_readers=self.world_size,
            ring_buffer=ring_buffer,
            serde_class=MsgpackSerde,
        )
        # cache prompt_updates for P0 only
        self._p0_cache: dict[str, Sequence[ResolvedPromptUpdate]] = {}

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
        return self._shm_cache.is_cached(mm_hash)

    @override
    def get_and_update_item(
        self,
        mm_item: MultiModalProcessorCacheInItem,
        mm_hash: str,
    ) -> MultiModalProcessorCacheOutItem:
        if self._shm_cache.is_cached(mm_hash):
            self._hits += 1
            self._total += 1

            address, monotonic_id = self._shm_cache.get_cached(mm_hash)
            prompt_updates = self._p0_cache[mm_hash]
            return self.address_as_item(address, monotonic_id), prompt_updates

        assert mm_item is not None, f"Expected a cached item for {mm_hash=}"
        item, prompt_updates = mm_item

        self._total += 1

        try:
            address, monotonic_id = self._shm_cache.put(mm_hash, item)
            # Try to remove dangling items if p0 cache is too large.
            if len(self._p0_cache) >= 2 * len(self._shm_cache.key_index):
                self.remove_dangling_items()

            self._p0_cache[mm_hash] = prompt_updates
            return self.address_as_item(address, monotonic_id), prompt_updates
        except ValueError as e:
            # `put` raises ValueError either for an oversize item or for a
            # duplicate key (concurrent insert); the latter is benign so we
            # only warn on the oversize case. Subsequent UUID-only requests
            # for an oversize item will fail with a cache miss.
            if "already exists" not in str(e):
                logger.warning_once(
                    "mm_input %s too large to cache; "
                    "raise --mm-shm-cache-max-object-size-mb. (%s)",
                    mm_hash,
                    str(e),
                )
            return mm_item
        except MemoryError as e:
            # Cache full and protected items prevent eviction.
            logger.debug(
                "mm_input %s not cached; shm cache full, "
                "consider raising --mm-processor-cache-gb. (%s)",
                mm_hash,
                str(e),
            )
            return mm_item

    @override
    def touch_sender_cache_item(self, mm_hash: str) -> None:
        """Touch the item in shared memory cache to prevent eviction.
        Increments writer_flag on sender side."""
        self._shm_cache.touch(mm_hash)

    @override
    def clear_cache(self) -> None:
        self._shm_cache.clear()
        self._p0_cache.clear()

        self._hits = 0
        self._total = 0
        self._last_info = CacheInfo(hits=0, total=0)

    @override
    def make_stats(self, *, delta: bool = False) -> CacheInfo:
        return self._stat(delta=delta)

    @override
    def close(self) -> None:
        self._shm_cache.close()

    def remove_dangling_items(self) -> None:
        """Remove items that are no longer in the shared memory cache."""
        cached_hashes = self._shm_cache.key_index.keys()
        dangling_hashes = set(self._p0_cache.keys()) - cached_hashes
        for mm_hash in dangling_hashes:
            del self._p0_cache[mm_hash]

    def address_as_item(
        self,
        address: int,
        monotonic_id: int,
    ) -> MultiModalKwargsItem:
        addr_elem = MultiModalFieldElem(
            data=address,
            field=MultiModalBatchedField(),
        )
        id_elem = MultiModalFieldElem(
            data=monotonic_id,
            field=MultiModalBatchedField(),
        )

        return MultiModalKwargsItem({"address": addr_elem, "monotonic_id": id_elem})


class ShmObjectStoreReceiverCache(BaseMultiModalReceiverCache):
    """The cache which is used on P1 Worker Process when SHM caching is enabled.

    How to update each item:

    - If the item has an address, replace the input with the cached item.
    - If not, return the input.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        shared_worker_lock: LockType,
    ) -> None:
        super().__init__()

        self.world_size = vllm_config.parallel_config.world_size
        mm_config = vllm_config.model_config.get_multimodal_config()

        ring_buffer = SingleWriterShmRingBuffer(
            data_buffer_size=int(mm_config.mm_processor_cache_gb * GiB_bytes),
            name=envs.VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME,
            create=False,  # Server is a reader
        )
        self._shm_cache = SingleWriterShmObjectStorage(
            max_object_size=mm_config.mm_shm_cache_max_object_size_mb * MiB_bytes,
            n_readers=self.world_size,
            ring_buffer=ring_buffer,
            serde_class=MsgpackSerde,
            reader_lock=shared_worker_lock,
        )

    @override
    def get_and_update_features(
        self, mm_features: list["MultiModalFeatureSpec"]
    ) -> list["MultiModalFeatureSpec"]:
        # strip_covered_mm_data preserves address items, so None here represents
        # a stripped uncached payload and has no SHM reference to acknowledge.
        features_with_data = [
            feature for feature in mm_features if feature.data is not None
        ]
        super().get_and_update_features(features_with_data)
        return mm_features

    @override
    def get_and_update_item(
        self,
        mm_item: MultiModalKwargsItem | None,
        mm_hash: str,
    ) -> MultiModalKwargsItem:
        assert mm_item is not None, f"Expected an address item for {mm_hash=}"
        if "address" in mm_item:
            address = cast(int, mm_item["address"].data)
            monotonic_id = cast(int, mm_item["monotonic_id"].data)
            return self._shm_cache.get(address, monotonic_id)

        return mm_item

    @override
    def touch_receiver_cache_item(
        self,
        mm_hash: str,
        mm_item: MultiModalKwargsItem | None = None,
    ) -> None:
        """Touch the item in shared memory cache to prevent eviction.
        Increments reader_count on receiver side."""
        assert mm_item is not None
        if "address" in mm_item:
            address = cast(int, mm_item["address"].data)
            monotonic_id = cast(int, mm_item["monotonic_id"].data)
            self._shm_cache.touch(mm_hash, address=address, monotonic_id=monotonic_id)

    @override
    def clear_cache(self) -> None:
        self._shm_cache.clear()
