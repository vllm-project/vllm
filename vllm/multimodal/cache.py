# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import operator
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from multiprocessing.synchronize import Lock as LockType
from typing import TYPE_CHECKING, Generic, TypeAlias, TypeVar, cast

import torch
from typing_extensions import override

import vllm.envs as envs
from vllm.distributed.device_communicators.shm_object_storage import (
    MsgpackSerde,
    SingleWriterShmObjectStorage,
    SingleWriterShmRingBuffer,
)
from vllm.logger import init_logger
from vllm.utils.cache import CacheInfo, LRUCache
from vllm.utils.jsontree import json_count_leaves, json_map_leaves, json_reduce_leaves
from vllm.utils.mem_constants import GiB_bytes, MiB_bytes

from .inputs import (
    MultiModalBatchedField,
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargs,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    NestedTensors,
)

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

    from .processing import ResolvedPromptUpdate
    from .registry import MultiModalRegistry

logger = init_logger(__name__)


class MultiModalProcessorCacheItem:
    """
    The data to store inside `MultiModalProcessorOnlyCache`.

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
    """
    The metadata to store inside `MultiModalProcessorSenderCache`.

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
    | MultiModalKwargs
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
            (
                MultiModalKwargs,
                MultiModalKwargsItems,
                MultiModalKwargsItem,
                MultiModalFieldElem,
            ),
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
                "Calculated size of %s to be %.2f GiB (%d leaves)",
                type(value),
                size / GiB_bytes,
                leaf_count,
            )

        return size

    @classmethod
    def get_item_complexity(cls, value: MultiModalCacheValue) -> int:
        """
        Get the number of leaf elements in a multi-modal cache value.

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
    """
    Abstract base class to read/write multi-modal items from cache.

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
        """
        Possibly update a multi-modal item based on whether it is
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
        """
        Possibly update a sequence of multi-modal items based on whether they
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
        """
        Check whether a multi-modal item is
        in the underlying cache.

        This **DOES NOT** update the cache eviction order.

        Args:
            mm_hash: The hash of the item to check.

        Returns:
            `True` if the item is cached, otherwise `False`.
        """
        raise NotImplementedError

    def is_cached(self, mm_hashes: list[str]) -> list[bool]:
        """
        Check whether a sequence of multi-modal items are
        in the underlying cache.

        This **DOES NOT** update the cache eviction order.

        Args:
            mm_hashes: The hash of each item to check.

        Returns:
            For each item, `True` if the item is cached, otherwise `False`.
        """
        return [self.is_cached_item(mm_hash) for mm_hash in mm_hashes]

    @abstractmethod
    def make_stats(self, *, delta: bool = False) -> CacheInfo:
        """
        Get (and reset) the multi-modal cache stats.

        Returns:
            The current multi-modal caching stats.
        """
        raise NotImplementedError


class MultiModalProcessorOnlyCache(BaseMultiModalProcessorCache):
    """
    The cache which is used on P0 when IPC caching is disabled.

    How to update each item:

    - If the item is in the cache, replace the input with the cached item.
    - If the item is not in the cache, store that item (which includes
      tensor data and metadata) into the cache, and return the input.
    """

    def __init__(self, model_config: "ModelConfig") -> None:
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

        self._cache[mm_hash] = MultiModalProcessorCacheItem(*mm_item)

        return mm_item

    @override
    def clear_cache(self) -> None:
        self._cache.clear()

    @override
    def make_stats(self, *, delta: bool = False) -> CacheInfo:
        return self._cache.stat(delta=delta)


class MultiModalProcessorSenderCache(BaseMultiModalProcessorCache):
    """
    The cache which is used on P0 when IPC caching is enabled.

    How to update each item:

    - If the item is already in the cache, clear the input to avoid
      unnecessary IPC.

    - If the item is not in the cache, store the metadata of that item so
      that the eviction policy remains the same as the cache on P1,
      and return the input.
      By only storing the metadata, we avoid keeping the data itself in
      memory inside P0.
    """

    def __init__(self, model_config: "ModelConfig") -> None:
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

        self._cache[mm_hash] = MultiModalProcessorCacheItemMetadata(*mm_item)

        return mm_item

    @override
    def clear_cache(self) -> None:
        self._cache.clear()

    @override
    def make_stats(self, *, delta: bool = False) -> CacheInfo:
        return self._cache.stat(delta=delta)


class ShmObjectStoreSenderCache(BaseMultiModalProcessorCache):
    """
    The cache which is used on P0 when IPC caching is enabled.

    How to update each item:

    - If the item is already in the cache, clear the input to avoid
      unnecessary IPC.

    - If the item is not in the cache, store the data in shared memory.
    """

    def __init__(self, vllm_config: "VllmConfig") -> None:
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
        # cache (prompt_updates, modality) for P0 only
        self._p0_cache: dict[str, tuple[Sequence[ResolvedPromptUpdate], str]] = {}

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
            prompt_updates, modality = self._p0_cache[mm_hash]
            return self.address_as_item(address, monotonic_id, modality), prompt_updates

        assert mm_item is not None, f"Expected a cached item for {mm_hash=}"

        self._total += 1

        try:
            address, monotonic_id = self._shm_cache.put(mm_hash, mm_item[0])
            # Try to remove dangling items if p0 cache is too large.
            if len(self._p0_cache) >= 2 * len(self._shm_cache.key_index):
                self.remove_dangling_items()
            self._p0_cache[mm_hash] = mm_item[1], mm_item[0].modality
            address_item = self.address_as_item(
                address, monotonic_id, mm_item[0].modality
            )
            return address_item, mm_item[1]
        except (ValueError, MemoryError) as e:
            # put may fail if the object is too large or
            # the cache is full.
            # In this case we log the error and keep the original mm_input.
            logger.debug("Failed to cache mm_input with hash %s: %s", mm_hash, e)
            return mm_item

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

    def remove_dangling_items(self) -> None:
        """Remove items that are no longer in the shared memory cache."""
        cached_hashes = self._shm_cache.key_index.keys()
        dangling_hashes = set(self._p0_cache.keys()) - cached_hashes
        for mm_hash in dangling_hashes:
            del self._p0_cache[mm_hash]

    def address_as_item(
        self, address: int, monotonic_id: int, modality: str
    ) -> MultiModalKwargsItem:
        addr_elem = MultiModalFieldElem(
            modality=modality,
            key="address",
            data=address,
            field=MultiModalBatchedField(),
        )
        id_elem = MultiModalFieldElem(
            modality=modality,
            key="monotonic_id",
            data=monotonic_id,
            field=MultiModalBatchedField(),
        )
        mm_item = MultiModalKwargsItem.from_elems([addr_elem, id_elem])
        return mm_item


def _enable_processor_cache(
    model_config: "ModelConfig",
    mm_registry: "MultiModalRegistry",
) -> bool:
    if not mm_registry.supports_multimodal_inputs(model_config):
        return False

    mm_config = model_config.get_multimodal_config()
    return mm_config.mm_processor_cache_gb > 0


def _enable_ipc_cache(vllm_config: "VllmConfig") -> bool:
    parallel_config = vllm_config.parallel_config
    supports_ipc_cache = (
        parallel_config._api_process_count == 1
        and parallel_config.data_parallel_size == 1
    ) or parallel_config.data_parallel_external_lb

    return supports_ipc_cache


def _enable_mm_input_shm_cache(vllm_config: "VllmConfig") -> bool:
    """Whether the shared memory based cache should be enabled."""

    if not _enable_ipc_cache(vllm_config):
        return False

    mm_config = vllm_config.model_config.get_multimodal_config()

    return mm_config.mm_processor_cache_type == "shm"


def processor_cache_from_config(
    vllm_config: "VllmConfig",
    mm_registry: "MultiModalRegistry",
) -> BaseMultiModalProcessorCache | None:
    """Return a `BaseMultiModalProcessorCache`, if enabled."""
    model_config = vllm_config.model_config

    if not _enable_processor_cache(model_config, mm_registry):
        return None

    if not _enable_ipc_cache(vllm_config):
        return MultiModalProcessorOnlyCache(model_config)

    if not _enable_mm_input_shm_cache(vllm_config):
        return MultiModalProcessorSenderCache(model_config)
    return ShmObjectStoreSenderCache(vllm_config)


def processor_only_cache_from_config(
    model_config: "ModelConfig",
    mm_registry: "MultiModalRegistry",
):
    """Return a `MultiModalProcessorOnlyCache`, if enabled."""
    if not _enable_processor_cache(model_config, mm_registry):
        return None

    return MultiModalProcessorOnlyCache(model_config)


class BaseMultiModalReceiverCache(
    BaseMultiModalCache[MultiModalKwargsItem | None, MultiModalKwargsItem]
):
    """The required interface for caches on P1."""

    def get_and_update_features(
        self,
        mm_features: list["MultiModalFeatureSpec"],
    ) -> list["MultiModalFeatureSpec"]:
        """Update multimodal features with cached encoder outputs."""
        for feature in mm_features:
            feature.data = self.get_and_update_item(feature.data, feature.identifier)
        return mm_features


class MultiModalReceiverCache(BaseMultiModalReceiverCache):
    """
    The cache which is used on P1 when IPC caching is enabled.

    How to update each item:

    - If the item is in the cache, replace the input with the cached item.
    - If the item is not in the cache, store that item (which includes tensor
      data) into the cache, and return the input.
    """

    def __init__(self, model_config: "ModelConfig") -> None:
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

        assert mm_item is not None, f"Expected a cached item for {mm_hash=}"

        self._cache[mm_hash] = mm_item
        return mm_item

    @override
    def clear_cache(self) -> None:
        self._cache.clear()


class ShmObjectStoreReceiverCache(BaseMultiModalReceiverCache):
    """
    The cache which is used on P1 Worker Process when IPC caching is enabled.

    How to update each item:

    - If the item has an address, replace the input with the cached item.
    - If not, return the input.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
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
    def clear_cache(self) -> None:
        self._shm_cache.clear()


def engine_receiver_cache_from_config(
    vllm_config: "VllmConfig",
    mm_registry: "MultiModalRegistry",
) -> BaseMultiModalReceiverCache | None:
    """
    This is used in the engine process.
    Return a `BaseMultiModalReceiverCache` only when IPC caching is enabled and
    mm_processor_cache_type=="lru".
    """
    model_config = vllm_config.model_config

    if not _enable_processor_cache(model_config, mm_registry):
        return None

    if not _enable_ipc_cache(vllm_config):
        return None

    if not _enable_mm_input_shm_cache(vllm_config):
        return MultiModalReceiverCache(model_config)

    return None


def worker_receiver_cache_from_config(
    vllm_config: "VllmConfig",
    mm_registry: "MultiModalRegistry",
    shared_worker_lock: LockType,
) -> BaseMultiModalReceiverCache | None:
    """
    This is used in the worker process.
    Return a `BaseMultiModalReceiverCache` only when IPC caching is enabled and
    mm_processor_cache_type=="shm".
    """
    model_config = vllm_config.model_config

    if not _enable_processor_cache(model_config, mm_registry):
        return None

    if not _enable_ipc_cache(vllm_config):
        return None

    if not _enable_mm_input_shm_cache(vllm_config):
        return None

    return ShmObjectStoreReceiverCache(vllm_config, shared_worker_lock)
