# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing_extensions import override

from vllm.config import ModelConfig
from vllm.utils.cache import CacheInfo

from ..inputs import MultiModalKwargsItem
from .base import (
    BaseMultiModalProcessorCache,
    BaseMultiModalReceiverCache,
    MultiModalCache,
    MultiModalCacheMissError,
    MultiModalProcessorCacheInItem,
    MultiModalProcessorCacheItemMetadata,
    MultiModalProcessorCacheOutItem,
)


class LruKeyReplicatedSenderCache(BaseMultiModalProcessorCache):
    """The cache which is used on P0 when LRU caching is enabled.

    How to update each item:

    - If the item is already in the cache, clear the input to avoid
      unnecessary IPC.

    - If the item is not in the cache, store the metadata of that item so
      that the eviction policy remains the same as the cache on P1,
      and return the input.
      By only storing the metadata, we avoid keeping the data itself in
      memory inside P0.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()

        mm_config = model_config.get_multimodal_config()

        self._cache = MultiModalCache.get_lru_cache(
            mm_config.mm_processor_cache_gb,
            MultiModalProcessorCacheItemMetadata,
        )

    @override
    def is_cached_item(self, mm_hash: str) -> bool:
        return mm_hash in self._cache

    @override
    def get_and_update_item(
        self,
        mm_item: MultiModalProcessorCacheInItem,
        mm_hash: str,
    ) -> MultiModalProcessorCacheOutItem:
        if (cached_item := self._cache.get(mm_hash)) is not None:
            return None, cached_item.prompt_updates

        assert mm_item is not None, f"Expected a cached item for {mm_hash=}"

        self.cache_if_fits(
            self._cache, mm_hash, MultiModalProcessorCacheItemMetadata(*mm_item)
        )
        return mm_item

    @override
    def touch_sender_cache_item(self, mm_hash: str) -> None:
        self._cache.touch(mm_hash)

    @override
    def clear_cache(self) -> None:
        self._cache.clear()

    @override
    def make_stats(self, *, delta: bool = False) -> CacheInfo:
        return self._cache.stat(delta=delta)

    @override
    def invalidate(self, mm_hash: str) -> None:
        # Drop our stale shadow entry so the next request for this hash re-sends the
        # data and repopulates P1 (see MultiModalCacheMissError).
        self._cache.pop(mm_hash, None)


class LruKeyReplicatedReceiverCache(BaseMultiModalReceiverCache):
    """The cache which is used on P1 when LRU caching is enabled.

    How to update each item:

    - If the item is in the cache, replace the input with the cached item.
    - If the item is not in the cache, store that item (which includes tensor
      data) into the cache, and return the input.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()

        mm_config = model_config.get_multimodal_config()

        self._cache = MultiModalCache.get_lru_cache(
            mm_config.mm_processor_cache_gb,
            MultiModalKwargsItem,
        )

    @override
    def get_and_update_item(
        self,
        mm_item: MultiModalKwargsItem | None,
        mm_hash: str,
    ) -> MultiModalKwargsItem:
        if (cached_item := self._cache.get(mm_hash)) is not None:
            return cached_item

        # No data and not cached here: P0 sent data=None trusting its shadow, but
        # the P0/P1 caches have drifted. Raise a retryable error (not assert) so the
        # engine can have P0 drop the stale entry and the client resend the data.
        if mm_item is None:
            raise MultiModalCacheMissError([mm_hash])

        self.cache_if_fits(self._cache, mm_hash, mm_item)
        return mm_item

    @override
    def touch_receiver_cache_item(
        self,
        mm_hash: str,
        mm_item: MultiModalKwargsItem | None = None,
    ) -> None:
        self._cache.touch(mm_hash)

    @override
    def clear_cache(self) -> None:
        self._cache.clear()
