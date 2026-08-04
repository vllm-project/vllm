# SPDX-License-Identifier: Apache-2.0
"""
LRU (Least Recently Used) eviction policy implementation
"""

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


class LRUEvictionPolicy(EvictionPolicy):
    """
    LRU (Least Recently Used) eviction policy.

    This policy tracks the order of key accesses and evicts the least recently
    used keys first when eviction is needed.

    Thread Safety:
        This class is thread-safe. All operations are protected by a global lock
        to ensure eventual consistency without corrupted states.

    Attributes:
        _lock: Threading lock for thread-safe operations
        _order: OrderedDict maintaining LRU order (most recent at end)
        _destinations: List of registered eviction destinations
        _default_destination: The default destination for eviction actions
    """

    def __init__(
        self,
        default_destination: EvictionDestination = EvictionDestination.DISCARD,
    ):
        """
        Initialize the LRU eviction policy.

        Args:
            default_destination: The default destination for evicted objects.
                Defaults to DISCARD.
        """
        # Lock for thread-safe operations
        self._lock = threading.Lock()

        # OrderedDict to maintain LRU order - keys at the beginning are oldest
        self._order: OrderedDict[ObjectKey, None] = OrderedDict()

        # List of registered eviction destinations
        self._destinations: list[EvictionDestination] = []

        # Default destination for eviction
        self._default_destination = default_destination

    def register_eviction_destination(self, destination: EvictionDestination):
        """
        Register an eviction destination for the eviction policy to use.

        Args:
            destination (EvictionDestination): The eviction destination to register
        """
        with self._lock:
            if destination not in self._destinations:
                self._destinations.append(destination)

    def on_keys_created(self, keys: list[ObjectKey]):
        """
        Notify the eviction policy that new keys have been created.
        New keys are added as most recently used.

        Args:
            keys (list[ObjectKey]): The keys that have been created
        """
        if not keys:
            return
        with self._lock:
            # NOTE: for the request, the later keys should be evicted first.
            # For example, the request has (key1, key2, key3), if we first
            # evict key1, due to prefix match, key2 and key3 will not be hit.
            for key in reversed(keys):
                # If key already exists, move it to the end (most recently used)
                if key in self._order:
                    self._order.move_to_end(key)
                else:
                    # Add new key at the end (most recently used)
                    self._order[key] = None

    def on_keys_touched(self, keys: list[ObjectKey]):
        """
        Notify the eviction policy that keys have been accessed.
        Touched keys are moved to the most recently used position.

        Args:
            keys (list[ObjectKey]): The keys that have been accessed
        """
        if not keys:
            return
        with self._lock:
            # NOTE: for the request, the later keys should be evicted first.
            # The example is the same as `on_keys_created`.
            for key in reversed(keys):
                if key in self._order:
                    # Move to end (most recently used)
                    self._order.move_to_end(key)

    def on_keys_removed(self, keys: list[ObjectKey]):
        """
        Notify the eviction policy that keys have been deleted.
        Deleted keys are removed from tracking.

        Args:
            keys (list[ObjectKey]): The keys that have been deleted
        """
        if not keys:
            return
        with self._lock:
            for key in keys:
                # Remove from LRU order tracking
                if key in self._order:
                    del self._order[key]

    def get_eviction_actions(
        self,
        expected_ratio: float,
        key_eligible_filter: Callable[[ObjectKey], bool] | None = None,
        cache_salt: str | None = None,
    ) -> list[EvictionAction]:
        """
        Get the eviction actions to evict objects from L1 cache.
        Returns keys in LRU order (least recently used first).

        Args:
            expected_ratio (float): A hint indicating approximately what fraction
                of tracked keys should be evicted. Value should be in range [0.0, 1.0].
                For example, 0.1 means roughly 10% of keys should be evicted.
            key_eligible_filter: An optional callable that takes an ObjectKey
                and returns True if the key is eligible for eviction. When
                provided, keys for which the filter returns False will be
                skipped. This is useful for skipping locked keys that
                cannot be deleted.
            cache_salt: Ignored by LRU policy (not user-level).

        Returns:
            list[EvictionAction]: The eviction actions to perform. Each
                action contains the keys and one eviction destination.

        Notes:
            The eviction action may not be successfully executed, or it
            may be executed asynchronously. Therefore, the eviction policy
            should not assume that the objects are evicted immediately, but
            it should use `on_keys_deleted` to know when the objects are actually
            deleted.
        """
        with self._lock:
            if not self._order:
                return []

            # Clamp expected_ratio to valid range
            expected_ratio = max(0.0, min(1.0, expected_ratio))

            # Calculate target number of keys to evict based on ratio
            target_count = int(len(self._order) * expected_ratio)

            # Ensure at least 1 key if ratio > 0 and we have keys
            if expected_ratio > 0 and target_count == 0 and len(self._order) > 0:
                target_count = 1

            if target_count == 0:
                return []

            # Get keys in LRU order (from beginning - least recently used),
            # skipping keys that fail the filter (e.g. locked keys).
            keys_to_evict: list[ObjectKey] = []

            for key in self._order:
                if key_eligible_filter is not None and not key_eligible_filter(key):
                    # Skip keys that are not eligible for eviction
                    # (e.g. currently locked by read/write operations)
                    continue
                keys_to_evict.append(key)
                if len(keys_to_evict) >= target_count:
                    break

            if not keys_to_evict:
                return []

            # Determine the destination
            destination = self._default_destination
            if self._destinations:
                # Use the first registered destination if available
                destination = self._destinations[0]

            return [EvictionAction(keys=keys_to_evict, destination=destination)]

    # =========================================================================
    # Methods below are NOT part of the EvictionPolicy interface.
    # They are provided for testing and debugging purposes only.
    # =========================================================================

    def get_num_tracked_keys(self) -> int:
        """
        Get the number of keys currently being tracked.

        Note:
            This method is NOT part of the EvictionPolicy interface.
            It is provided for testing and debugging purposes only.

        Returns:
            int: The number of tracked keys
        """
        with self._lock:
            return len(self._order)

    def get_eviction_candidates(self, count: int) -> list[ObjectKey]:
        """
        Get a list of eviction candidates without creating eviction actions.
        Useful for querying what would be evicted.

        Note:
            This method is NOT part of the EvictionPolicy interface.
            It is provided for testing and debugging purposes only.

        Args:
            count: Maximum number of candidates to return

        Returns:
            list[ObjectKey]: List of keys that are candidates for eviction,
                in LRU order (least recently used first)
        """
        with self._lock:
            candidates: list[ObjectKey] = []

            for key in self._order:
                candidates.append(key)
                if len(candidates) >= count:
                    break

            return candidates
