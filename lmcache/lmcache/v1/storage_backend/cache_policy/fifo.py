# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy, KeyType

logger = init_logger(__name__)


class FIFOCachePolicy(BaseCachePolicy[KeyType, dict[KeyType, Any]]):
    """
    FIFO cache policy.
    """

    def __init__(self):
        logger.info("Initializing FIFOCachePolicy")

    def init_mutable_mapping(self) -> dict[KeyType, Any]:
        # NOTE(Jiayi): python dict maintains insertion order.
        return {}

    def update_on_hit(
        self,
        key: KeyType,
        cache_dict: dict[KeyType, Any],
    ) -> None:
        pass

    def update_on_put(
        self,
        key: KeyType,
    ) -> None:
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
        cache_dict: dict[KeyType, Any],
        num_candidates: int = 1,
    ) -> list[KeyType]:
        evict_keys = []
        for key, cache in cache_dict.items():
            if not cache.can_evict:
                continue
            evict_keys.append(key)
            if len(evict_keys) == num_candidates:
                break

        return evict_keys
