# SPDX-License-Identifier: Apache-2.0
"""
Per-cache_salt quota registry.

Holds the authoritative map from ``cache_salt`` to a byte budget.
Consumed by the L2 eviction controller each cycle to decide which
users are over their quota, and by the HTTP API for runtime
administration (CRUD endpoints at ``/quota/...``).

Salts without an explicit entry resolve through two different lenses:

- :meth:`get_limit_bytes` — legacy allowlist semantics: unregistered
  salts resolve to ``0`` bytes, so their data is evicted next cycle.
  Used by the MP server's local eviction controller.
- :meth:`effective_limit_bytes` — default-quota semantics: unregistered
  salts resolve to the registry's *default limit*, which starts as
  ``None`` (exempt from eviction). Used by the coordinator's eviction
  manager, so a freshly booted coordinator whose quota table has not
  been re-synced by the external quota controller cannot mass-evict
  unknown tenants. The controller arms allowlist enforcement by setting
  the default to ``0`` (``PUT /quota/config``) after syncing quotas.
"""

# Future
from __future__ import annotations

# Standard
import threading

# First Party
from lmcache.v1.distributed.internal_api import QuotaEntry


class QuotaManager:
    """Thread-safe registry of byte quotas keyed by ``cache_salt``.

    Quotas are dynamic — CRUD operations (``set_quota``, ``delete_quota``)
    are cheap, and any change takes effect on the very next eviction
    cycle (no restart needed).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # cache_salt -> limit in bytes
        self._limits: dict[str, int] = {}
        # Limit applied by ``effective_limit_bytes`` to salts with no
        # explicit entry. ``None`` (the boot default) means "exempt":
        # consumers of the default-quota semantics skip eviction for
        # unregistered salts until an operator sets a concrete value.
        self._default_limit_bytes: int | None = None

    def set_quota(self, cache_salt: str, limit_bytes: int) -> None:
        """Create or update the quota for ``cache_salt``.

        Raises ``ValueError`` for a negative limit. A zero limit is
        accepted and behaves identically to having no entry at all
        (keys under this salt will be evicted next cycle) — the
        difference is purely bookkeeping: the entry shows up in
        ``list_quotas()`` until removed.
        """
        if limit_bytes < 0:
            raise ValueError(f"limit_bytes must be non-negative (got {limit_bytes})")
        with self._lock:
            self._limits[cache_salt] = limit_bytes

    def delete_quota(self, cache_salt: str) -> bool:
        """Remove the quota entry for ``cache_salt``.

        Returns ``True`` if an entry was removed, ``False`` if the salt
        had no registration. After deletion the effective limit drops
        to ``0``, so any existing bytes under that salt will be evicted
        at the next eviction cycle.
        """
        with self._lock:
            return self._limits.pop(cache_salt, None) is not None

    def get_limit_bytes(self, cache_salt: str) -> int:
        """Return the effective limit for ``cache_salt``.

        Unregistered salts resolve to ``0`` (allowlist semantics) —
        callers use this value to drive eviction decisions, so the
        zero default deliberately triggers eviction of any bytes
        accumulated under an unknown salt.
        """
        with self._lock:
            return self._limits.get(cache_salt, 0)

    def set_default_limit_bytes(self, limit_bytes: int | None) -> None:
        """Set the limit applied to salts with no explicit entry.

        Args:
            limit_bytes: ``None`` exempts unregistered salts from
                eviction (the boot default). ``0`` activates strict
                allowlist enforcement — all bytes under unregistered
                salts become evictable next cycle. A positive value
                gives every unregistered salt that byte budget.

        Raises:
            ValueError: If ``limit_bytes`` is negative.
        """
        if limit_bytes is not None and limit_bytes < 0:
            raise ValueError(f"limit_bytes must be non-negative (got {limit_bytes})")
        with self._lock:
            self._default_limit_bytes = limit_bytes

    def get_default_limit_bytes(self) -> int | None:
        """Return the default limit for unregistered salts.

        ``None`` means unregistered salts are exempt from eviction (no
        default has been configured since startup).
        """
        with self._lock:
            return self._default_limit_bytes

    def effective_limit_bytes(self, cache_salt: str) -> int | None:
        """Return the limit for ``cache_salt`` under default-quota semantics.

        Explicitly registered salts return their registered limit.
        Unregistered salts return the default limit — ``None`` (exempt
        from eviction) until :meth:`set_default_limit_bytes` configures a
        concrete value. See the module docstring for how this differs
        from :meth:`get_limit_bytes`.
        """
        with self._lock:
            explicit = self._limits.get(cache_salt)
            if explicit is not None:
                return explicit
            return self._default_limit_bytes

    def has_quota(self, cache_salt: str) -> bool:
        """Whether ``cache_salt`` has an explicit registration.

        Useful for distinguishing "zero limit by default" from
        "explicitly registered with limit=0" in status reports.
        """
        with self._lock:
            return cache_salt in self._limits

    def list_quotas(self) -> list[QuotaEntry]:
        """Return a snapshot of all registered quotas.

        The returned list is a detached copy; mutating it does not
        affect the registry.
        """
        with self._lock:
            return [
                QuotaEntry(cache_salt=salt, limit_bytes=limit)
                for salt, limit in self._limits.items()
            ]
