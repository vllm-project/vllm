# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Optional, Union

import torch
from typing_extensions import TypeVar, override

from vllm.logger import init_logger
from vllm.utils import GiB_bytes, LRUCache
from vllm.utils.jsontree import json_map_leaves, json_reduce_leaves

from .inputs import (MultiModalFieldElem, MultiModalKwargs,
                     MultiModalKwargsItem, MultiModalKwargsItems,
                     NestedTensors)

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

    from .registry import MultiModalRegistry

logger = init_logger(__name__)


@dataclass
class MultiModalCacheItemMetadata:
    size: int

    @classmethod
    def wraps(cls, value: "MultiModalCacheValue"):
        return cls(size=MultiModalCache.get_item_size(value))


MultiModalCacheValue = Union[
    MultiModalKwargsItems,
    MultiModalKwargsItem,
    MultiModalKwargs,
    Mapping[str, NestedTensors],
    MultiModalCacheItemMetadata,
]

_V = TypeVar("_V", bound=MultiModalCacheValue)


class MultiModalCache:

    @classmethod
    def get_leaf_size(
        cls,
        leaf: object,
        *,
        debug: bool = False,
    ) -> int:
        if isinstance(leaf, MultiModalFieldElem):
            return cls.get_item_size(leaf.data)  # type: ignore

        # These are not subclasses of dict
        if isinstance(leaf, MultiModalKwargsItems):
            return cls.get_item_size(leaf.data)  # type: ignore
        if isinstance(leaf, MultiModalKwargsItem):
            return cls.get_item_size(leaf.data)  # type: ignore
        if isinstance(leaf, MultiModalKwargs):
            return cls.get_item_size(leaf.data)  # type: ignore

        # sys.getsizeof doesn't work for tensors
        if isinstance(leaf, torch.Tensor):
            return leaf.nbytes

        if isinstance(leaf, MultiModalCacheItemMetadata):
            return leaf.size

        return sys.getsizeof(leaf)

    @classmethod
    def get_item_size(
        cls,
        value: MultiModalCacheValue,
        *,
        debug: bool = False,
    ) -> int:
        size = json_reduce_leaves(
            lambda a, b: a + b,
            json_map_leaves(lambda x: cls.get_leaf_size(x, debug=debug),
                            value),
        )

        if debug:
            logger.debug("Calculated size of %s to be %.2f GiB", type(value),
                         size / GiB_bytes)

        return size

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


_O = TypeVar(
    "_O",
    MultiModalKwargsItem,
    Optional[MultiModalKwargsItem],
    covariant=True,
    default=Optional[MultiModalKwargsItem],
)


class CachedMultiModalInputExchanger(ABC, Generic[_O]):
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

    @staticmethod
    def for_p0(
        vllm_config: "VllmConfig",
        mm_registry: "MultiModalRegistry",
    ) -> "CachedMultiModalInputExchanger[Optional[MultiModalKwargsItem]]":
        model_config = vllm_config.model_config

        if not mm_registry.supports_multimodal_inputs(model_config):
            return CachedMultiModalInputDisabled()

        mm_config = model_config.get_multimodal_config()
        if mm_config.mm_processor_cache_gb == 0:
            return CachedMultiModalInputDisabled()

        parallel_config = vllm_config.parallel_config
        supports_ipc_cache = (parallel_config.data_parallel_size == 1
                              or parallel_config.data_parallel_external_lb)
        if not supports_ipc_cache:
            return CachedMultiModalInputReceiver(model_config)

        if mm_registry.create_processor(model_config).requires_out_mm_kwargs:
            # The processed data must be cached inside P0
            return CachedMultiModalInputReceiver(model_config)

        return CachedMultiModalInputSender(model_config)

    @staticmethod
    def for_p1(
        vllm_config: "VllmConfig",
        mm_registry: "MultiModalRegistry",
    ) -> "CachedMultiModalInputExchanger[MultiModalKwargsItem]":
        model_config = vllm_config.model_config

        if not mm_registry.supports_multimodal_inputs(model_config):
            return CachedMultiModalInputDisabled()

        mm_config = model_config.get_multimodal_config()
        if mm_config.mm_processor_cache_gb == 0:
            return CachedMultiModalInputDisabled()

        parallel_config = vllm_config.parallel_config
        supports_ipc_cache = (parallel_config.data_parallel_size == 1
                              or parallel_config.data_parallel_external_lb)
        if not supports_ipc_cache:
            return CachedMultiModalInputDisabled()

        return CachedMultiModalInputReceiver(model_config)

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
    def get_and_update_item(
        self,
        mm_item: Optional[MultiModalKwargsItem],
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
        mm_items: Sequence[Optional[MultiModalKwargsItem]],
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


class CachedMultiModalInputSender(
        CachedMultiModalInputExchanger[Optional[MultiModalKwargsItem]]):
    """
    How to update each item:

    - If the item is already in the cache, clear the data in that item to avoid
      unnecessary IPC.

    - If the item is not in the cache, store the size metadata of that item so
      that the eviction policy remains the same as the cache on P1.
      By only storing the size metadata, we avoid keeping the data itself in
      memory inside P0.
    """

    def __init__(self, model_config: "ModelConfig") -> None:
        super().__init__()

        mm_config = model_config.get_multimodal_config()

        self._cache = MultiModalCache.get_lru_cache(
            mm_config.mm_processor_cache_gb,
            MultiModalCacheItemMetadata,
        )

    @override
    def is_cached_item(self, mm_hash: str) -> bool:
        return mm_hash in self._cache

    @override
    def get_and_update_item(
        self,
        mm_item: Optional[MultiModalKwargsItem],
        mm_hash: str,
    ) -> Optional[MultiModalKwargsItem]:
        if self._cache.get(mm_hash) is not None or mm_item is None:
            return None

        self._cache[mm_hash] = MultiModalCacheItemMetadata.wraps(mm_item)
        return mm_item

    @override
    def clear_cache(self) -> None:
        self._cache.clear()


class CachedMultiModalInputReceiver(
        CachedMultiModalInputExchanger[MultiModalKwargsItem]):
    """
    How to update each item:

    - If the item is in the cache, put the cached data into the item.
    - If the item is not in the cache, store the data into the cache.
    """

    def __init__(self, model_config: "ModelConfig") -> None:
        super().__init__()

        mm_config = model_config.get_multimodal_config()

        self._cache = MultiModalCache.get_lru_cache(
            mm_config.mm_processor_cache_gb,
            MultiModalKwargsItem,
        )

    @override
    def is_cached_item(self, mm_hash: str) -> bool:
        return mm_hash in self._cache

    @override
    def get_and_update_item(
        self,
        mm_item: Optional[MultiModalKwargsItem],
        mm_hash: str,
    ) -> MultiModalKwargsItem:
        if (cached_item := self._cache.get(mm_hash)) is not None:
            return cached_item

        assert mm_item is not None, f"Expected an item from P0 for {mm_hash=}"

        self._cache[mm_hash] = mm_item
        return mm_item

    @override
    def clear_cache(self) -> None:
        self._cache.clear()


class CachedMultiModalInputDisabled(
        CachedMultiModalInputExchanger[MultiModalKwargsItem]):
    """Return the passed items without applying any caching."""

    @override
    def is_cached_item(self, mm_hash: str) -> bool:
        return False

    @override
    def get_and_update_item(
        self,
        mm_item: Optional[MultiModalKwargsItem],
        mm_hash: str,
    ) -> MultiModalKwargsItem:
        assert mm_item is not None, ("CachedMultiModalInputSender should not "
                                     "be used when caching is disabled")

        return mm_item

    @override
    def clear_cache(self) -> None:
        pass
