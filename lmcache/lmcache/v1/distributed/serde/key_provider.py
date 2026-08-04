# SPDX-License-Identifier: Apache-2.0
"""Key providers mapping a ``cache_salt`` to AES key bytes.

A serde selects the key material for a tenant's ``cache_salt`` (the public
tenant selector) through a ``KeyProvider``. See
``docs/design/v1/distributed/serde/aesgcm.md`` for the key and trust models.
"""

# Standard
import abc
import threading

# Third Party
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


class KeyProvider(abc.ABC):
    """Supplies the AES key for a given ``cache_salt``.

    Implementations must be thread-safe: the serde thread pool may call
    ``get_key`` concurrently.
    """

    @abc.abstractmethod
    def get_key(self, cache_salt: str) -> bytes:
        """Return the AES key bytes for ``cache_salt`` (16 or 32 bytes)."""
        raise NotImplementedError


class HkdfKeyProvider(KeyProvider):
    """Derive a per-``cache_salt`` key from one master key via HKDF-SHA256.

    Args:
        master_key: Secret input key material (any length; keep >= key_len).
        key_len: Derived key length in bytes (16 for AES-128, 32 for AES-256).
        info_prefix: HKDF ``info`` prefix for domain separation; ``cache_salt``
            is appended to it. Callers with different schemes must pass distinct
            prefixes so their derived keys never collide.
    """

    def __init__(self, master_key: bytes, key_len: int, info_prefix: bytes) -> None:
        if not master_key:
            raise ValueError("master_key must be non-empty")
        self._master_key = master_key
        self._key_len = key_len
        self._info_prefix = info_prefix
        self._cache: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def get_key(self, cache_salt: str) -> bytes:
        """Return the derived key for ``cache_salt``, computing once and caching.

        Thread-safe: the cache is guarded by a lock.
        """
        with self._lock:
            key = self._cache.get(cache_salt)
            if key is None:
                key = HKDF(
                    algorithm=SHA256(),
                    length=self._key_len,
                    salt=None,
                    info=self._info_prefix + cache_salt.encode("utf-8"),
                ).derive(self._master_key)
                self._cache[cache_salt] = key
            return key
