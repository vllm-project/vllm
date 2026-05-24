# SPDX-License-Identifier: Apache-2.0
"""P41 Redis-backed response cache — cross-process + restart-survivable.

Design rationale
----------------
The in-memory `ResponseCacheLRU` is per-process. For the Genesis
deployment on VM 100 we have:

  - vllm-server (2 TP workers + 1 API server)
  - cliproxyapi proxy (independent process, port 8330)
  - genesis-aggregator (independent process)
  - LibreChat / OpenWebUI / lobechat (independent processes)

Without a shared backing store, each process would keep its own LRU —
miss rate across the stack would be unnecessarily high, and a restart
wipes everything. Redis (already running as `agg-redis` on the server,
see `project_genesis_infrastructure` memory) sidesteps both issues.

What this backend does
----------------------
- Same public API as `ResponseCacheLRU` — callers can swap between
  backends transparently via env `GENESIS_P41_BACKEND=redis` vs
  `GENESIS_P41_BACKEND=memory`.
- TTL is enforced by Redis `SETEX` — no client-side timestamp math.
- Hit/miss counters are stored as Redis `INCR` counters at
  `<prefix>:stats:hits` / `:misses` so they survive across processes
  and restarts; `stats()` returns a snapshot via `MGET`.
- Graceful degradation: if Redis is down at `get()` time, log a
  warning and return None (treat as miss). At `store()` time, same.
  No exception propagation to the request path → no user-visible fault.

Scope (explicit non-goals)
--------------------------
- **NO semantic similarity.** Exact-match only, matching v7.6 P41
  contract. Semantic cache (P41b) is explicitly DROPPED per user
  direction — hallucination risk unacceptable.
- **NO LRU semantics here** — Redis is the LRU itself via `maxmemory-
  policy allkeys-lru` (operator responsibility, set on the Redis side).
  We pass `max_entries` to the constructor for stats-display parity
  with `ResponseCacheLRU`, but enforcement is on Redis.
- **NO hit-weighted eviction.** That requires scanning candidates and
  is cheap in-process but expensive over the Redis wire. Redis LRU
  is good enough for cross-process use.

Dependencies
------------
- `redis` Python package (NOT a hard Genesis dep — imported lazily
  inside this module; import failure surfaces at backend-selection
  time and triggers fallback to in-memory LRU via `get_default_cache`).

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
Status: v7.7 implementation (opt-in via env)
"""
from __future__ import annotations

import json
import logging
from typing import Any, Mapping, Optional

from vllm._genesis.cache.response_cache import _stable_key

log = logging.getLogger("genesis.cache.redis_backend")

_KEY_PREFIX = "genesis:p41:"
_STATS_HITS = _KEY_PREFIX + "stats:hits"
_STATS_MISSES = _KEY_PREFIX + "stats:misses"
_STATS_STORES = _KEY_PREFIX + "stats:stores"


class RedisResponseCache:
    """Redis-backed exact-match response cache.

    Public API matches `ResponseCacheLRU`: `get`, `store`, `invalidate`,
    `clear`, `stats`. Drop-in replacement via `GENESIS_P41_BACKEND=redis`.

    Thread-safety: relies on the underlying `redis-py` client which is
    connection-pool-based and safe for concurrent use.

    Resilience: never raises on Redis transport errors. A failed
    `get()` returns None; a failed `store()` is a logged warning. The
    request path is never blocked on cache I/O failures.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/1",
        ttl_seconds: int = 3600,
        max_entries: int = 0,
        connect_timeout: float = 1.0,
        socket_timeout: float = 0.5,
    ):
        """Lazy-import `redis`. Raises ImportError if unavailable —
        `get_default_cache` catches that and falls back to LRU."""
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be > 0, got {ttl_seconds}")
        try:
            import redis  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "redis package not installed. Install `pip install redis` "
                "to use P41 Redis backend. Falling back to in-memory LRU."
            ) from e

        self._ttl_seconds = int(ttl_seconds)
        self._max_entries = int(max_entries)  # informational only
        self._url = redis_url
        self._client = redis.from_url(
            redis_url,
            socket_connect_timeout=connect_timeout,
            socket_timeout=socket_timeout,
            decode_responses=False,  # we handle bytes ourselves
            health_check_interval=30,
        )

    # ── public api (mirrors ResponseCacheLRU) ───────────────────────

    def get(
        self,
        prompt: str,
        model: str,
        sampling_params: Mapping[str, Any],
    ) -> Optional[Any]:
        key = _KEY_PREFIX + _stable_key(prompt, model, sampling_params)
        try:
            raw = self._client.get(key)
            if raw is None:
                self._incr_safe(_STATS_MISSES)
                return None
            self._incr_safe(_STATS_HITS)
            # Value is JSON-serialised response payload.
            return json.loads(raw.decode("utf-8", errors="replace"))
        except Exception as e:
            # Transport or decode error — treat as miss, don't block caller.
            log.warning(
                "[P41 Redis] get failed, treating as miss: %s",
                type(e).__name__,
            )
            return None

    def store(
        self,
        prompt: str,
        model: str,
        sampling_params: Mapping[str, Any],
        response: Any,
    ) -> None:
        key = _KEY_PREFIX + _stable_key(prompt, model, sampling_params)
        try:
            payload = json.dumps(
                response, ensure_ascii=False, default=str,
            ).encode("utf-8")
        except (TypeError, ValueError) as e:
            log.warning(
                "[P41 Redis] response not JSON-serialisable (%s); "
                "skipping store for key=%s...", type(e).__name__, key[:40],
            )
            return
        try:
            self._client.setex(key, self._ttl_seconds, payload)
            self._incr_safe(_STATS_STORES)
        except Exception as e:
            log.warning(
                "[P41 Redis] store failed: %s", type(e).__name__,
            )

    def invalidate(
        self,
        prompt: str,
        model: str,
        sampling_params: Mapping[str, Any],
    ) -> bool:
        key = _KEY_PREFIX + _stable_key(prompt, model, sampling_params)
        try:
            return bool(self._client.delete(key))
        except Exception as e:
            log.warning(
                "[P41 Redis] invalidate failed: %s", type(e).__name__,
            )
            return False

    def clear(self) -> None:
        """Drop all entries + stats. Uses SCAN+DEL so it's safe on
        large datasets (does NOT issue `FLUSHDB`)."""
        try:
            pipe = self._client.pipeline(transaction=False)
            count = 0
            for k in self._client.scan_iter(match=_KEY_PREFIX + "*"):
                pipe.delete(k)
                count += 1
                if count % 1000 == 0:
                    pipe.execute()
                    pipe = self._client.pipeline(transaction=False)
            pipe.execute()
        except Exception as e:
            log.warning(
                "[P41 Redis] clear failed: %s", type(e).__name__,
            )

    # ── introspection ────────────────────────────────────────────────

    def stats(self) -> dict:
        """Snapshot of cross-process hit/miss/store counters + current
        key count via DBSIZE-equivalent scan.

        NB: Key count is approximate under concurrent load (Redis scan
        is lock-free). For precise accounting use Redis's native
        `INFO keyspace` on the operator side.
        """
        out: dict[str, Any] = {
            "backend": "redis",
            "url": self._url,
            "ttl_seconds": self._ttl_seconds,
            "max_entries": self._max_entries,
        }
        try:
            hits, misses, stores = self._client.mget(
                _STATS_HITS, _STATS_MISSES, _STATS_STORES,
            )
            hits_i = int(hits or 0)
            misses_i = int(misses or 0)
            stores_i = int(stores or 0)
            total = hits_i + misses_i
            out["hits"] = hits_i
            out["misses"] = misses_i
            out["stores"] = stores_i
            out["hit_rate"] = (hits_i / total) if total else 0.0
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"
            return out

        # Approximate size via scan (cheap for our scale — 1k entries max)
        #
        # G-012 audit fix (2026-05-02): exclude the stats keys by NAME
        # rather than subtracting a fixed constant 3. If only some of
        # the stats keys exist (e.g. fresh cache with only one store
        # call so misses/stores haven't been incremented yet), the
        # `size - 3` arithmetic underflowed and `max(0, ...)` masked
        # legitimate entries — a 1-entry cache with 1 stats key would
        # report `size=0` instead of `size=1`.
        _STATS_KEY_SET = frozenset((
            _STATS_HITS, _STATS_MISSES, _STATS_STORES,
        ))
        try:
            size = sum(
                1 for k in self._client.scan_iter(
                    match=_KEY_PREFIX + "*", count=500,
                )
                if (
                    k.decode("utf-8", errors="replace")
                    if isinstance(k, (bytes, bytearray))
                    else k
                ) not in _STATS_KEY_SET
            )
            out["size"] = size
        except Exception:
            out["size"] = None
        return out

    def __len__(self) -> int:
        try:
            return self.stats().get("size") or 0
        except Exception:
            return 0

    # ── internals ───────────────────────────────────────────────────

    def _incr_safe(self, key: str) -> None:
        try:
            self._client.incr(key)
        except Exception:
            # Stats are best-effort; never block on failure.
            pass
