# SPDX-License-Identifier: Apache-2.0
"""Per-``cache_salt`` L2 usage manager for the MP coordinator."""

# Future
from __future__ import annotations

# Standard
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey

logger = init_logger(__name__)


class L2UsageManager:
    """Thread-safe in-memory ledger of L2 byte usage per ``cache_salt``,
    plus a per-key size map so re-stores don't double-count."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bytes_by_salt: dict[str, int] = {}
        self._total_bytes: int = 0
        self._key_sizes: dict[ObjectKey, int] = {}

    def has_key(self, key: ObjectKey) -> bool:
        """Return ``True`` if ``key`` has a recorded size."""
        with self._lock:
            return key in self._key_sizes

    def get_key_size(self, key: ObjectKey) -> int | None:
        """Return the bytes tracked for ``key``, or ``None`` if unknown."""
        with self._lock:
            return self._key_sizes.get(key)

    def record_stored(self, key: ObjectKey, num_bytes: int) -> None:
        """Record that ``key`` is resident on L2 at ``num_bytes``.

        Re-storing the same key replaces its size (delta-adjusts the
        per-salt and global totals); a re-store at the same size is a
        no-op.

        Raises:
            ValueError: ``num_bytes`` is negative.
        """
        if num_bytes < 0:
            raise ValueError(f"num_bytes must be non-negative (got {num_bytes})")
        with self._lock:
            existing = self._key_sizes.get(key)
            if existing is not None:
                delta = num_bytes - existing
            else:
                delta = num_bytes
            if delta == 0:
                # Already tracked at the same size — keep the entry,
                # nothing to adjust.
                self._key_sizes[key] = num_bytes
                return
            self._key_sizes[key] = num_bytes
            salt = key.cache_salt
            new_salt_total = self._bytes_by_salt.get(salt, 0) + delta
            if new_salt_total <= 0:
                self._bytes_by_salt.pop(salt, None)
            else:
                self._bytes_by_salt[salt] = new_salt_total
            self._total_bytes = max(0, self._total_bytes + delta)

    def record_evicted(self, key: ObjectKey) -> int:
        """Drop ``key`` from the ledger and return the bytes freed
        (``0`` if ``key`` was unknown)."""
        with self._lock:
            size = self._key_sizes.pop(key, None)
            if size is None or size == 0:
                return size or 0
            salt = key.cache_salt
            current = self._bytes_by_salt.get(salt, 0)
            new_val = current - size
            if new_val < 0:
                logger.warning(
                    "Usage underflow for cache_salt=%r on evict: %d - %d = %d",
                    salt,
                    current,
                    size,
                    new_val,
                )
                new_val = 0
            if new_val == 0:
                self._bytes_by_salt.pop(salt, None)
            else:
                self._bytes_by_salt[salt] = new_val
            self._total_bytes = max(0, self._total_bytes - size)
            return size

    def get(self, cache_salt: str) -> int:
        """Return the current byte usage for ``cache_salt``."""
        with self._lock:
            return self._bytes_by_salt.get(cache_salt, 0)

    def get_all(self) -> dict[str, int]:
        """Return a snapshot copy of per-salt byte usage."""
        with self._lock:
            return dict(self._bytes_by_salt)

    def get_total(self) -> int:
        """Return total bytes tracked across all salts."""
        with self._lock:
            return self._total_bytes
