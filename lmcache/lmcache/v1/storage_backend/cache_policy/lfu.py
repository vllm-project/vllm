# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any

# Third Party
from sortedcontainers import SortedDict

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy, KeyType

logger = init_logger(__name__)


class LFUCachePolicy(BaseCachePolicy[KeyType, dict[KeyType, Any]]):
    """
    LFU cache policy.
    """

    # NOTE(Jiayi): We use `sorted dict` + `bucket` to implement LFU.
    # NOTE(Jiayi): We use FIFO for entries with the same frequency.
    def __init__(self):
        # TODO(Jiayi): `SortedDict` is log(N).
        # A way to make it O(1) is to use a dict and keep track min freuency.
        # However, this requires us keep another data structures to keep track
        # of the pinned keys.
        self.freq_to_keys = SortedDict()

        # TODO(Jiayi): We can optimize this a bit by using `key_to_val_freq`
        self.key_to_freq = {}

        logger.info("Initializing LFUCachePolicy")

    def init_mutable_mapping(self) -> dict[KeyType, Any]:
        return {}

    def update_on_hit(
        self,
        key: KeyType,
        cache_dict: dict[KeyType, Any],
    ) -> None:
        curr_freq = self.key_to_freq[key]
        self.freq_to_keys[curr_freq].pop(key)
        if not self.freq_to_keys[curr_freq]:
            self.freq_to_keys.pop(curr_freq)

        curr_freq += 1
        self.key_to_freq[key] = curr_freq

        if curr_freq not in self.freq_to_keys:
            self.freq_to_keys[curr_freq] = {key: None}
        else:
            self.freq_to_keys[curr_freq][key] = None

    def update_on_put(
        self,
        key: KeyType,
    ) -> None:
        # Initialize the frequency for the new key.
        self.key_to_freq[key] = 1
        if 1 not in self.freq_to_keys:
            self.freq_to_keys[1] = {key: None}
        else:
            self.freq_to_keys[1][key] = None

    def update_on_force_evict(
        self,
        key: KeyType,
    ) -> None:
        freq = self.key_to_freq.pop(key, None)
        if not freq:
            return
        self.freq_to_keys[freq].pop(key)
        if not self.freq_to_keys[freq]:
            self.freq_to_keys.pop(freq)

    # NOTE(Jiayi): We do best effort to get eviction candidates so the number
    # of returned keys might be smaller than num_candidates.
    def get_evict_candidates(
        self,
        cache_dict: dict[KeyType, Any],
        num_candidates: int = 1,
    ) -> list[KeyType]:
        evict_keys = []
        evict_freqs = []
        for curr_min_freq, fifo_keys in self.freq_to_keys.items():
            for key in fifo_keys:
                if not cache_dict[key].can_evict:
                    continue
                evict_keys.append(key)
                evict_freqs.append(curr_min_freq)
                self.key_to_freq.pop(key)
                if len(evict_keys) == num_candidates:
                    break

            if len(evict_keys) == num_candidates:
                break

        for freq, key in zip(evict_freqs, evict_keys, strict=False):
            self.freq_to_keys[freq].pop(key)
            if not self.freq_to_keys[freq]:
                self.freq_to_keys.pop(freq)

        return evict_keys
