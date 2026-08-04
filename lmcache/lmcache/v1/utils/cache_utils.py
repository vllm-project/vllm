# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Callable, Generic, Optional, TypeVar
import threading
import time

T = TypeVar("T")


class TTLListCache(Generic[T]):
    """
    Generic list cache with TTL (Time-To-Live) support.

    This cache stores a list of items of type T and automatically refreshes
    when the cache expires.
    """

    def __init__(self, timeout_seconds: float = 30.0):
        """
        Initialize the cache.

        Args:
            timeout_seconds: Default TTL in seconds for cache entries.
        """
        self.cache: list[T] = []
        self.cache_time: float = 0.0
        self.cache_lock = threading.Lock()
        self.timeout_seconds = timeout_seconds

    def get_cached(
        self,
        get_fresh_data: Callable[[], list[T]],
        timeout_override: Optional[float] = None,
    ) -> list[T]:
        """
        Get cached data, refreshing if expired.

        Args:
            get_fresh_data: Function to get fresh data when cache is expired
            timeout_override: Optional timeout override in seconds.
                If None, use instance timeout_seconds.
                If 0: cache immediately expires, always get fresh data and update cache.
                If > 0: use specified timeout.

        Returns:
            List of cached items, either from cache or fresh data
        """
        current_time = time.time()

        # Fast path: check cache without lock
        if not self.is_expired(timeout_override, current_time):
            return self.cache

        # Slow path: refresh cache with lock
        with self.cache_lock:
            # Double-check after acquiring lock
            if not self.is_expired(timeout_override, current_time):
                return self.cache

            # Refresh cache
            self.cache = get_fresh_data()
            self.cache_time = current_time
            return self.cache

    def clear(self) -> None:
        """Clear the cache."""
        with self.cache_lock:
            self.cache.clear()
            self.cache_time = 0.0

    def is_expired(
        self,
        timeout_override: Optional[float] = None,
        current_time: Optional[float] = None,
    ) -> bool:
        """
        Check if the cache is expired without refreshing it.

        Args:
            timeout_override: Optional timeout override in seconds.
                If None, use instance timeout_seconds.
                If 0: always considered expired.
                If > 0: use specified timeout.
            current_time: Optional current time. If None, use time.time().

        Returns:
            True if cache is expired or uninitialized, False otherwise
        """
        if self.cache_time == 0.0:
            return True

        effective_timeout = (
            timeout_override if timeout_override is not None else self.timeout_seconds
        )
        if current_time is None:
            current_time = time.time()

        return current_time - self.cache_time >= effective_timeout

    def get_cache_age(self) -> float:
        """
        Get the age of the cache in seconds.

        Returns:
            Age in seconds. Returns float('inf') if cache is uninitialized.
        """
        if self.cache_time == 0.0:
            return float("inf")
        return time.time() - self.cache_time

    def __len__(self) -> int:
        """Get the number of items in cache (without checking expiration)."""
        return len(self.cache)

    def __repr__(self) -> str:
        """String representation of the cache."""
        age = self.get_cache_age()
        if age == float("inf"):
            age_str = "uninitialized"
        else:
            age_str = f"{age:.2f}s old"
        return (
            f"TTLListCache(cache_items={len(self)}, age={age_str}, "
            f"timeout={self.timeout_seconds}s)"
        )
