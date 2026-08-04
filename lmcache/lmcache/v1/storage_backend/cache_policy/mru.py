# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from typing import Any

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy, KeyType

logger = init_logger(__name__)


class MRUCachePolicy(BaseCachePolicy[KeyType, OrderedDict[KeyType, Any]]):
    """
    MRU cache policy.
    """

    def __init__(self):
        logger.info("Initializing MRUCachePolicy")

    def init_mutable_mapping(self) -> OrderedDict[KeyType, Any]:
        return OrderedDict()

    def update_on_hit(
        self,
        key: KeyType,
        cache_dict: OrderedDict[KeyType, Any],
    ) -> None:
        # since MRU evicts from the back, the logic is same as LRU.
        cache_dict.move_to_end(key, last=True)

    def update_on_put(
        self,
        key: KeyType,
    ) -> None:
        # No action needed for MRU on put, as the key is already at the back.
        pass

    def update_on_force_evict(
        self,
        key: KeyType,
    ) -> None:
        pass

    # NOTE(Jiayi): We do best effort to get eviction candidates so the number
    # of returned keys might be smaller than num_candidates.
    def get_evict_candidates(
        self,
        cache_dict: OrderedDict[KeyType, Any],
        num_candidates: int = 1,
    ) -> list[KeyType]:
        evict_keys = []
        # Since the most recent object is at the end, we reverse the order here
        for key, cache in reversed(cache_dict.items()):
            if not cache.can_evict:
                continue
            evict_keys.append(key)
            if len(evict_keys) == num_candidates:
                break

        return evict_keys
