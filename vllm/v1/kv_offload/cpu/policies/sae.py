# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session-Aware Eviction (SAE) cache policy for CPU offload.

Ported from the out-of-tree ``sae_kv_offload`` plugin package. Two
documented semantic differences from that reference:

1. Session boundaries are reconstructed from the call sequence
   (``insert`` after ``touch``/``evict``/``remove``/``clear`` opens a
   new session; consecutive ``insert`` calls join it) rather than
   taken from a batch of block hashes handed to ``prepare_store``.
2. Per-batch position weighting on ``get`` is dropped because the
   current scheduler calls ``manager.lookup`` one key at a time.
"""

from collections import OrderedDict
from collections.abc import Iterable

from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy


class SAECachePolicy(CachePolicy):
    """Session-Aware Eviction cache policy.

    See the module docstring for the two semantic differences from
    the reference algorithm.
    """

    def __init__(
        self,
        cache_capacity: int,
        *,
        decay_interval: int = 500,
        decay_factor: float = 0.9,
        ghost_hit_weight: float = 12.0,
        ghost_miss_weight: float = 1.0,
        ghost_norm: float = 12.0,
    ) -> None:
        self._cache_capacity = cache_capacity
        self._decay_interval = decay_interval
        self._decay_factor = decay_factor
        self._ghost_hit_weight = ghost_hit_weight
        self._ghost_miss_weight = ghost_miss_weight
        self._ghost_norm = ghost_norm

        self._blocks: dict[OffloadKey, BlockStatus] = {}
        self._sid_to_keys: dict[int, list[OffloadKey]] = {}
        self._key_to_sid: dict[OffloadKey, int] = {}
        self._sid_stats: dict[int, dict[str, float | int]] = {}
        self._key_ghost: dict[OffloadKey, float] = {}
        self._evictable_keys: OrderedDict[OffloadKey, None] = OrderedDict()

        self._logical_timer: int = 0
        self._sid_counter: int = 0
        self._lookup_count: int = 0
        self._open_sid: int | None = None
        self._last_event: str = "init"

    @override
    def get(self, key: OffloadKey) -> BlockStatus | None:
        return self._blocks.get(key)

    @override
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        raise NotImplementedError

    @override
    def remove(self, key: OffloadKey) -> None:
        raise NotImplementedError

    @override
    def touch(self, keys: Iterable[OffloadKey]) -> None:
        raise NotImplementedError

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        raise NotImplementedError

    @override
    def clear(self) -> None:
        raise NotImplementedError
