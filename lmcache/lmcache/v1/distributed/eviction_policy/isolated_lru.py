# SPDX-License-Identifier: Apache-2.0
"""
IsolatedLRU (per-``cache_salt`` LRU) eviction policy.

Unlike :class:`LRUEvictionPolicy` which maintains a single global LRU
list, this policy keeps one LRU list per ``cache_salt`` and uses
``support_isolation=True`` so the L2 eviction controller can scope
eviction to a specific salt via the ``cache_salt`` argument to
``get_eviction_actions``.

Combined with :class:`QuotaManager`, this lets the controller evict
only the over-quota ``cache_salt``'s keys while leaving other salts'
cached data untouched.
"""

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict
from collections.abc import Callable
import threading

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.eviction import EvictionPolicy
from lmcache.v1.distributed.internal_api import (
    EvictionAction,
    EvictionDestination,
)


class IsolatedLRUEvictionPolicy(EvictionPolicy):
    """Per-``cache_salt`` LRU eviction policy.

    Maintains one ``OrderedDict`` per ``cache_salt`` (keyed by
    ``ObjectKey``). New and touched keys move to the end of their
    bucket's ordering; least-recently-used keys stay at the front and
    are evicted first.

    Thread-safety: every public method holds a single lock so bucket
    dicts cannot be mutated concurrently.
    """

    @property
    def support_isolation(self) -> bool:
        return True

    def __init__(
        self,
        default_destination: EvictionDestination = EvictionDestination.DISCARD,
    ):
        self._lock = threading.Lock()
        # cache_salt -> ordered {ObjectKey: None} (oldest first).
        self._per_salt_order: dict[str, OrderedDict[ObjectKey, None]] = {}
        # Registered destinations (first one wins if any are registered,
        # matching LRUEvictionPolicy semantics).
        self._destinations: list[EvictionDestination] = []
        self._default_destination = default_destination

    def register_eviction_destination(self, destination: EvictionDestination) -> None:
        with self._lock:
            if destination not in self._destinations:
                self._destinations.append(destination)

    def on_keys_created(self, keys: list[ObjectKey]) -> None:
        if not keys:
            return
        with self._lock:
            # Same prefix-match ordering rationale as ``LRUEvictionPolicy``:
            # later keys in a request should be evicted first, so we
            # insert in reverse to place the final key at the LRU head.
            for key in reversed(keys):
                order = self._per_salt_order.get(key.cache_salt)
                if order is None:
                    order = OrderedDict()
                    self._per_salt_order[key.cache_salt] = order
                if key in order:
                    order.move_to_end(key)
                else:
                    order[key] = None

    def on_keys_touched(self, keys: list[ObjectKey]) -> None:
        if not keys:
            return
        with self._lock:
            for key in reversed(keys):
                order = self._per_salt_order.get(key.cache_salt)
                if order is not None and key in order:
                    order.move_to_end(key)

    def on_keys_removed(self, keys: list[ObjectKey]) -> None:
        if not keys:
            return
        with self._lock:
            for key in keys:
                order = self._per_salt_order.get(key.cache_salt)
                if order is None:
                    continue
                order.pop(key, None)
                if not order:
                    # Drop the empty bucket so ``list``/iteration
                    # snapshots stay compact.
                    del self._per_salt_order[key.cache_salt]

    def get_eviction_actions(
        self,
        expected_ratio: float,
        key_eligible_filter: Callable[[ObjectKey], bool] | None = None,
        cache_salt: str | None = None,
    ) -> list[EvictionAction]:
        """Select victims in LRU order, scoped to a single ``cache_salt``.

        IsolatedLRU is "isolated only" by contract — callers must pass a
        concrete ``cache_salt``. The base interface keeps ``cache_salt``
        optional for compatibility with non-isolated policies; passing
        ``None`` here raises ``ValueError`` rather than silently
        falling back to a global pool.

        Args:
            expected_ratio: Fraction of the candidate pool to evict,
                clamped to ``[0.0, 1.0]``. If the ratio rounds down to
                zero and the pool is non-empty, at least one key is
                returned (matches ``LRUEvictionPolicy``).
            key_eligible_filter: Optional predicate — keys failing the
                filter (e.g. locked) are skipped.
            cache_salt: The salt whose bucket to evict from. Required.

        Returns:
            A list with at most one ``EvictionAction`` (one per
            destination — IsolatedLRU has a single destination, so the
            list is either empty or length 1). Empty when nothing is
            available to evict under the current filter / ratio.

        Raises:
            ValueError: If ``cache_salt`` is ``None``.
        """
        if cache_salt is None:
            raise ValueError(
                "IsolatedLRUEvictionPolicy.get_eviction_actions requires "
                "cache_salt; this policy is per-bucket and has no global "
                "eviction path."
            )
        with self._lock:
            order = self._per_salt_order.get(cache_salt)
            pool = list(order.keys()) if order else []

            if not pool:
                return []

            expected_ratio = max(0.0, min(1.0, expected_ratio))
            target_count = int(len(pool) * expected_ratio)
            if expected_ratio > 0 and target_count == 0:
                target_count = 1
            if target_count == 0:
                return []

            keys_to_evict: list[ObjectKey] = []
            for key in pool:
                if key_eligible_filter is not None and not key_eligible_filter(key):
                    continue
                keys_to_evict.append(key)
                if len(keys_to_evict) >= target_count:
                    break

            if not keys_to_evict:
                return []

            destination = self._default_destination
            if self._destinations:
                destination = self._destinations[0]

            return [EvictionAction(keys=keys_to_evict, destination=destination)]

    # =========================================================================
    # Methods below are NOT part of the EvictionPolicy interface.
    # They are provided for testing and debugging purposes only.
    # =========================================================================

    def get_num_tracked_keys(self, cache_salt: str | None = None) -> int:
        """Number of tracked keys, optionally scoped to one bucket."""
        with self._lock:
            if cache_salt is not None:
                order = self._per_salt_order.get(cache_salt)
                return len(order) if order is not None else 0
            return sum(len(o) for o in self._per_salt_order.values())

    def get_tracked_salts(self) -> list[str]:
        """Return the set of cache_salts with at least one tracked key."""
        with self._lock:
            return list(self._per_salt_order.keys())
