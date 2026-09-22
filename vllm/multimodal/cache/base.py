# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import operator
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Generic, TypeAlias, TypeVar

import torch
from typing_extensions import override

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.utils.cache import CacheInfo, LRUCache
from vllm.utils.jsontree import json_count_leaves, json_map_leaves, json_reduce_leaves
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.mem_utils import format_gib

from ..inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    NestedTensors,
)

if TYPE_CHECKING:
    from ..processing.processor import ResolvedPromptUpdate

logger = init_logger(__name__)


class MultiModalCacheMissError(RuntimeError):
    """Raised by the P1 receiver cache when items are requested with no data and
    are not cached.

    P0 (frontend) keeps a metadata-only shadow of P1 (engine) and sends
    ``data=None`` on a shadow hit. The two caches are updated in different orders
    across processes, so they can drift -- leaving P0 referencing items P1 has
    evicted. Raising (instead of asserting) lets the engine return a retryable
    response and have P0 drop the stale entries
    (``BaseMultiModalProcessorCache.invalidate``) so the client resends the data.
    Carries every drifted ``mm_hash`` in the request so P0 can drop them all in one
    pass -- one retry then recovers the whole request, not one item per retry.
    """

    def __init__(self, mm_hashes: list[str]) -> None:
        super().__init__(
            f"Multi-modal items {mm_hashes} are not in the receiver (P1) cache and "
            "no data was provided to recompute them (P0/P1 cache drift); the request "
            "should be retried with the multi-modal data attached."
        )
        self.mm_hashes = mm_hashes


class MultiModalProcessorCacheItem:
    """The data to store inside `MultiModalProcessorOnlyCache`.

    Args:
        item: The processed tensor data corresponding to a multi-modal item.
        prompt_updates: The prompt updates corresponding to `item`.

    """

    def __init__(
        self,
        item: MultiModalKwargsItem,
        prompt_updates: Sequence["ResolvedPromptUpdate"],
    ) -> None:
        super().__init__()

        self.item = item
        self.prompt_updates = prompt_updates


class MultiModalProcessorCacheItemMetadata:
    """The metadata to store inside `MultiModalProcessorSenderCache`.

    Args:
        item: The processed tensor data corresponding to a multi-modal item.
            Since P1 already stores the tensor data, we only store its size
            metadata in P0 to reduce memory usage. The size metadata is still
            needed to keep the same cache eviction policy as P0.
        prompt_updates: The prompt updates corresponding to `item`.
            This needs to stay on P0 because for some models, they are
            dependent on the processed tensor data (cached on P1).

    """

    def __init__(
        self,
        item: MultiModalKwargsItem,
        prompt_updates: Sequence["ResolvedPromptUpdate"],
    ) -> None:
        super().__init__()

        self.item_size = MultiModalCache.get_item_size(item)
        self.prompt_updates = prompt_updates


MultiModalCacheValue: TypeAlias = (
    MultiModalProcessorCacheItem
    | MultiModalProcessorCacheItemMetadata
    | MultiModalKwargsItems
    | MultiModalKwargsItem
    | Mapping[str, NestedTensors]
)

_V = TypeVar("_V", bound=MultiModalCacheValue)


class MultiModalCache:
    @classmethod
    def get_leaf_size(cls, leaf: object) -> int:
        if isinstance(leaf, MultiModalProcessorCacheItem):
            return cls.get_leaf_size(leaf.item)
        if isinstance(leaf, MultiModalProcessorCacheItemMetadata):
            return leaf.item_size

        # These are not subclasses of dict
        if isinstance(
            leaf,
            (MultiModalKwargsItems, MultiModalKwargsItem, MultiModalFieldElem),
        ):
            return cls.get_item_size(leaf.data)  # type: ignore

        # sys.getsizeof doesn't work for tensors
        if isinstance(leaf, torch.Tensor):
            return leaf.nbytes

        return sys.getsizeof(leaf)

    @classmethod
    def get_item_size(
        cls,
        value: MultiModalCacheValue,
        *,
        debug: bool = False,
    ) -> int:
        size = json_reduce_leaves(
            operator.add, json_map_leaves(cls.get_leaf_size, value)
        )

        if debug:
            leaf_count = json_count_leaves(value)
            logger.debug(
                "Calculated size of %s to be %s GiB (%d leaves)",
                type(value),
                format_gib(size),
                leaf_count,
            )

        return size

    @classmethod
    def get_item_complexity(cls, value: MultiModalCacheValue) -> int:
        """Get the number of leaf elements in a multi-modal cache value.

        This provides a measure of structural complexity that can be useful
        for debugging cache performance and understanding data patterns.

        Args:
            value: The multi-modal cache value to analyze.

        Returns:
            The number of leaf elements in the nested structure.

        """
        return json_count_leaves(value)

    @classmethod
    def get_lru_cache(
        cls,
        capacity_gb: float,
        value_type: type[_V],
        *,
        debug: bool = False,
    ) -> LRUCache[str, _V]:
        return LRUCache(
            GiB_bytes * capacity_gb,
            getsizeof=lambda x: cls.get_item_size(x, debug=debug),
        )


_I = TypeVar("_I", contravariant=True)
_O = TypeVar("_O", covariant=True)


class BaseMultiModalCache(ABC, Generic[_I, _O]):
    """Abstract base class to read/write multi-modal items from cache.

    The idea of multi-modal caching is based on having a client and server
    where the client executes in the frontend process (=P0) and
    the server in the core process (=P1). The data flow is as follows:

    ```
                  is_cached() x N    get_and_update()
    P0: From API -----------------> -----------------> To P1

                 get_and_update()
    P1: From P0 -----------------> To model
    ```

    `is_cached()` can be called any number of times in P0. However,
    `get_and_update()` must be called in P0 and P1 one after another
    so that their cache eviction order remains the same.

    This ensures that the keys in P0 and P1 caches are mirrored,
    allowing us to determine whether a key is cached in P1 by looking
    up the P0 cache, without having to communicate with P1.
    """

    @abstractmethod
    def get_and_update_item(
        self,
        mm_item: _I,
        mm_hash: str,
    ) -> _O:
        """Possibly update a multi-modal item based on whether it is
        in the underlying cache.

        This update is done out-of-place and updates the cache eviction order.

        Args:
            mm_item: The multi-modal item to update.
            mm_hash: The hash of `mm_item`.

        Returns:
            The update multi-modal item.

        """
        raise NotImplementedError

    def get_and_update(
        self,
        mm_items: Sequence[_I],
        mm_hashes: list[str],
    ) -> list[_O]:
        """Possibly update a sequence of multi-modal items based on whether they
        are in the underlying cache.

        This update is done out-of-place and updates the cache eviction order.

        Args:
            mm_items: The multi-modal items to update.
            mm_hashes: The hash of each item in `mm_items`.

        Returns:
            A new list of updated multi-modal items.

        """
        assert len(mm_items) == len(mm_hashes)

        return [
            self.get_and_update_item(mm_item, mm_hash)
            for mm_item, mm_hash in zip(mm_items, mm_hashes)
        ]

    def cache_if_fits(
        self,
        cache: LRUCache[str, _V],
        key: str,
        value: _V,
    ) -> bool:
        """Insert `value` if it fits in `cache`.

        `cachetools.Cache` raises `ValueError("value too large")` when a
        single item exceeds `maxsize`. An item bigger than the whole
        processor cache can never be a hit, so skip the insert and serve it
        uncached instead of aborting engine startup.

        LRU P0/P1 caches call this so they stay mirrored. SHM subclasses do
        not use it; they already skip oversize items in `put()`.

        Args:
            cache: The LRU cache to update.
            key: Cache key (typically the multi-modal item hash).
            value: Value to insert.

        Returns:
            `True` if the item was cached, otherwise `False`.

        """
        if cache.put_if_fits(key, value):
            return True
        logger.warning_once(
            "Skipping multi-modal processor cache insert for an item of "
            "%s GiB because it exceeds --mm-processor-cache-gb=%s. "
            "The item will be processed uncached; increase "
            "--mm-processor-cache-gb to cache items of this size.",
            format_gib(int(cache.getsizeof(value))),
            format_gib(int(cache.maxsize)),
        )
        return False

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear the underlying cache."""
        raise NotImplementedError


MultiModalProcessorCacheInItem: TypeAlias = (
    tuple[MultiModalKwargsItem, Sequence["ResolvedPromptUpdate"]] | None
)


MultiModalProcessorCacheOutItem: TypeAlias = tuple[
    MultiModalKwargsItem | None, Sequence["ResolvedPromptUpdate"]
]


class BaseMultiModalProcessorCache(
    BaseMultiModalCache[MultiModalProcessorCacheInItem, MultiModalProcessorCacheOutItem]
):
    """The required interface for caches on P0."""

    @abstractmethod
    def is_cached_item(self, mm_hash: str) -> bool:
        """Check whether a multi-modal item is
        in the underlying cache.

        This **DOES NOT** update the cache eviction order.

        Args:
            mm_hash: The hash of the item to check.

        Returns:
            `True` if the item is cached, otherwise `False`.

        """
        raise NotImplementedError

    def is_cached(self, mm_hashes: list[str]) -> list[bool]:
        """Check whether a sequence of multi-modal items are
        in the underlying cache.

        This **DOES NOT** update the cache eviction order.

        Args:
            mm_hashes: The hash of each item to check.

        Returns:
            For each item, `True` if the item is cached, otherwise `False`.

        """
        return [self.is_cached_item(mm_hash) for mm_hash in mm_hashes]

    def invalidate(self, mm_hash: str) -> None:
        """Drop ``mm_hash`` from this P0 shadow cache to recover from P0/P1 drift.

        No-op by default; shadow caches that can drift from P1 override this.
        """
        pass

    def close(self) -> None:
        """Close the underlying cache, if needed."""
        pass

    @abstractmethod
    def touch_sender_cache_item(self, mm_hash: str) -> None:
        """Update the cache eviction order for a multi-modal item.

        This is used to touch the item in the cache without changing
        its value.

        Args:
            mm_hash: The hash of the multi-modal item.

        """
        raise NotImplementedError

    @abstractmethod
    def make_stats(self, *, delta: bool = False) -> CacheInfo:
        """Get (and reset) the multi-modal cache stats.

        Returns:
            The current multi-modal caching stats.

        """
        raise NotImplementedError


class MultiModalProcessorOnlyCache(BaseMultiModalProcessorCache):
    """The cache which is used on P0 when IPC caching is disabled.

    How to update each item:

    - If the item is in the cache, replace the input with the cached item.
    - If the item is not in the cache, store that item (which includes
      tensor data and metadata) into the cache, and return the input.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()

        mm_config = model_config.get_multimodal_config()

        self._cache = MultiModalCache.get_lru_cache(
            mm_config.mm_processor_cache_gb,
            MultiModalProcessorCacheItem,
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
            return cached_item.item, cached_item.prompt_updates

        assert mm_item is not None, f"Expected a cached item for {mm_hash=}"

        self.cache_if_fits(self._cache, mm_hash, MultiModalProcessorCacheItem(*mm_item))
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


class BaseMultiModalReceiverCache(
    BaseMultiModalCache[MultiModalKwargsItem | None, MultiModalKwargsItem]
):
    """The required interface for caches on P1."""

    def get_and_update_features(
        self,
        mm_features: list["MultiModalFeatureSpec"],
    ) -> list["MultiModalFeatureSpec"]:
        """Update multimodal features with cached encoder outputs.
        Touch all identifier at first before update to avoid
        item in updated list evict during update.

        Uses mm_hash for cache key to share across LoRAs (falls back to
        identifier for backward compatibility).
        """
        for feature in mm_features:
            cache_key = feature.mm_hash or feature.identifier
            self.touch_receiver_cache_item(cache_key, feature.data)

        missing_mm_hashes: list[str] = []
        for feature in mm_features:
            cache_key = feature.mm_hash or feature.identifier
            try:
                feature.data = self.get_and_update_item(feature.data, cache_key)
            except MultiModalCacheMissError as e:
                # Collect every drifted hash in this request before raising, so the
                # engine can have P0 invalidate them all at once -- otherwise a
                # request with k drifted items needs k client retries (each resend
                # only un-shadows the one reported hash).
                missing_mm_hashes.extend(e.mm_hashes)
        if missing_mm_hashes:
            raise MultiModalCacheMissError(missing_mm_hashes)
        return mm_features

    @abstractmethod
    def touch_receiver_cache_item(
        self,
        mm_hash: str,
        mm_item: MultiModalKwargsItem | None = None,
    ) -> None:
        """Update the cache eviction order for a multi-modal item.

        This is used to touch the item in the cache without changing
        its value.

        Args:
            mm_hash: The hash of the multi-modal item.
            mm_item: The multi-modal item itself. This is optional and
                may not be needed by some cache implementations.

        """
        raise NotImplementedError
