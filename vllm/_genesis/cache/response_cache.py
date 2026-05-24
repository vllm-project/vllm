# SPDX-License-Identifier: Apache-2.0
"""P41 — Genesis response cache for exact-match identical queries.

Problem
-------
vLLM's built-in `--enable-prefix-caching` reuses **KV blocks** across
requests that share a prompt prefix. That removes prefill cost for
cached tokens but still pays for the decode pass — every request
even for a byte-for-byte identical prompt re-runs the full generation
loop.

For FAQ-style or agent/tool workflows where the SAME prompt +
sampling params repeat (retrieval-augmented, health checks, reproducing
a canned response), short-circuiting to a fully-cached completion
saves:

- 100% of decode cost for the identical-request case
- The entire end-to-end latency — no GPU queue, no detokenizer

Implementation
--------------
An in-process LRU cache keyed by a stable SHA-256 of
`(prompt, model, sampling_params_tuple)`. Entries store the full
OpenAI-compatible response JSON. Eviction is O(1) via `collections.
OrderedDict.move_to_end`. Configurable max entries + per-entry TTL.

Scope (explicit non-goals for v7.6)
-----------------------------------
- **No semantic similarity.** Keys are exact — byte-identical prompt
  AND byte-identical sampling params, or cache miss. v7.7 roadmap item
  `P41b` will add embedding-based near-duplicate matching via a
  lightweight sentence-transformer sidecar; kept OUT of v7.6 because
  semantic caching can return SUBTLY WRONG answers (quality regression)
  without careful threshold tuning + quality gating.
- **No "learning" / fine-tune.** Cache is read/write only; no
  gradient/update path. "Learning" in the user brief is scheduled for
  v7.7 as optional per-key quality tracking + automatic eviction of
  entries whose hit count is low.
- **No cross-process persistence.** LRU is per-process only. Restart
  wipes the cache. Durable disk cache is v7.7 item — design-wise a
  simple SQLite / LMDB sidecar; out of scope for this commit.

Opt-in gate
-----------
`GENESIS_ENABLE_P41_RESPONSE_CACHE=1`. Default OFF. Even when enabled
callers must EXPLICITLY probe + store — there is no middleware hook
installed automatically. Integration lives at the deployment layer
(aggregator proxy / FastAPI dep / your choice).

Invariants
----------
- Cache is **never consulted** for requests with `stream=True` since
  users of streaming explicitly opt into partial delivery semantics
  that cache can't replicate.
- Cache is **never consulted** for requests with `temperature > 0` by
  default — non-deterministic sampling would return STALE random
  outputs. Use `allow_sampled=True` kwarg to override at your own risk.
- Cache key includes model ID to avoid cross-model collisions.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
Status: v7.6 implementation (opt-in, exact-match only)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Mapping, Optional

log = logging.getLogger("genesis.cache.response_cache")

_ENV_ENABLE = "GENESIS_ENABLE_P41_RESPONSE_CACHE"
_ENV_MAX_ENTRIES = "GENESIS_P41_MAX_ENTRIES"
_ENV_TTL_SECONDS = "GENESIS_P41_TTL_SECONDS"
_ENV_HIT_WEIGHTED = "GENESIS_P41_HIT_WEIGHTED"
_ENV_HIT_ALPHA = "GENESIS_P41_HIT_ALPHA"
_ENV_BACKEND = "GENESIS_P41_BACKEND"          # "memory" (default) | "redis"
_ENV_REDIS_URL = "GENESIS_P41_REDIS_URL"      # e.g. "redis://192.168.1.10:6379/1"


def _read_bool_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _read_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
        if 0.0 <= v <= 1.0:
            return v
    except ValueError:
        pass
    return default


def is_p41_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _resolve_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return default


def _stable_key(
    prompt: str,
    model: str,
    sampling_params: Mapping[str, Any],
) -> str:
    """Build a stable SHA-256 cache key.

    Uses `json.dumps(sort_keys=True)` on sampling_params so the order of
    the caller-supplied dict doesn't change the key. Params with
    `None` value are dropped from the key so `{"stop": None}` and
    missing `stop` collide intentionally.
    """
    cleaned = {k: v for k, v in sampling_params.items() if v is not None}
    serialized = json.dumps(cleaned, sort_keys=True, ensure_ascii=False)
    blob = f"model={model}\x00prompt={prompt}\x00params={serialized}".encode(
        "utf-8", errors="replace",
    )
    return hashlib.sha256(blob).hexdigest()


class ResponseCacheLRU:
    """In-process LRU with TTL eviction for exact-match response cache.

    Thread-safe (single mutex around `get/store/evict`). O(1) insert
    + lookup; O(k) smart-eviction where k = eviction_scan_size (default
    8 — constant, so still amortised O(1) per insert).

    Eviction policy
    ---------------
    Default mode: **pure LRU** — oldest access wins eviction.

    Opt-in "hit-weighted" mode (v7.7): combine access-age (recency)
    with per-entry hit count so frequently-referenced entries survive
    longer than a pure LRU would let them. When enabled, `store()`
    scans up to `eviction_scan_size` oldest entries and evicts the
    one with the LOWEST score:

        score(entry) = alpha * recency_norm + (1 - alpha) * log1p(hits)

    - `recency_norm`: 0.0 for most-recently-accessed, 1.0 for oldest
    - `alpha`: 0.3 by default — 70% weight on popularity, 30% on recency

    This approximates "learning" per user's v7.7 brief: entries that
    historically got more hits are treated as more valuable and stay
    in cache longer than a naive LRU would allow. Low-hit (1-2 times)
    entries get evicted first, leaving high-hit (20+ times) entries
    protected against a burst of one-off requests.

    Rationale for the opt-in gate: hit-weighted mode ONLY helps when
    the workload has meaningful hit-count variance (FAQ / agent tool /
    canned response). For fully-unique-request workloads it's equivalent
    to pure LRU. Default off, enable via `GENESIS_P41_HIT_WEIGHTED=1`.

    Hit/miss counters are exposed for `/metrics` integration.
    """

    def __init__(
        self,
        max_entries: int = 1024,
        ttl_seconds: int = 3600,
        hit_weighted_eviction: bool = False,
        hit_weight_alpha: float = 0.3,
        eviction_scan_size: int = 8,
    ):
        if max_entries <= 0:
            raise ValueError(f"max_entries must be > 0, got {max_entries}")
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be > 0, got {ttl_seconds}")
        if not 0.0 <= hit_weight_alpha <= 1.0:
            raise ValueError(
                f"hit_weight_alpha must be in [0, 1], got {hit_weight_alpha}"
            )
        if eviction_scan_size < 1:
            raise ValueError(
                f"eviction_scan_size must be >= 1, got {eviction_scan_size}"
            )
        self._max_entries = int(max_entries)
        self._ttl_seconds = int(ttl_seconds)
        self._hit_weighted = bool(hit_weighted_eviction)
        self._hit_weight_alpha = float(hit_weight_alpha)
        self._eviction_scan_size = int(eviction_scan_size)
        # Entry format: (stored_at: float, payload: Any, hit_count: int)
        self._store: "OrderedDict[str, tuple[float, Any, int]]" = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._ttl_evictions = 0
        self._hit_weighted_evictions = 0

    # ── public api ───────────────────────────────────────────────

    def get(
        self,
        prompt: str,
        model: str,
        sampling_params: Mapping[str, Any],
    ) -> Optional[Any]:
        """Return cached response or None on miss / stale / disabled.

        On hit: bumps entry to MRU + increments per-entry hit_count
        (drives hit-weighted eviction) + global hits counter.
        Eagerly evicts stale entries (TTL enforcement).
        """
        key = _stable_key(prompt, model, sampling_params)
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            stored_at, payload, hit_count = entry
            if now - stored_at > self._ttl_seconds:
                # Stale — evict and report as miss.
                del self._store[key]
                self._ttl_evictions += 1
                self._misses += 1
                return None
            # Hit — increment per-entry counter, move to MRU.
            self._store[key] = (stored_at, payload, hit_count + 1)
            self._store.move_to_end(key, last=True)
            self._hits += 1
            return payload

    def store(
        self,
        prompt: str,
        model: str,
        sampling_params: Mapping[str, Any],
        response: Any,
    ) -> None:
        """Insert or refresh a cache entry.

        Refresh preserves hit_count — the entry stays "popular" even
        if its payload was updated (operator invalidation semantics).
        On insert-at-capacity, eviction picks the least-valuable entry
        using the configured policy (pure LRU or hit-weighted).
        """
        key = _stable_key(prompt, model, sampling_params)
        now = time.monotonic()
        with self._lock:
            if key in self._store:
                # Refresh — update timestamp + payload, preserve hit_count.
                _, _, hit_count = self._store[key]
                self._store.move_to_end(key, last=True)
                self._store[key] = (now, response, hit_count)
                return
            self._store[key] = (now, response, 0)
            self._store.move_to_end(key, last=True)
            # Evict until at size.
            while len(self._store) > self._max_entries:
                victim_key = self._pick_eviction_victim()
                del self._store[victim_key]
                self._evictions += 1

    # ── eviction-policy internals ───────────────────────────────────

    def _pick_eviction_victim(self) -> str:
        """Return the key of the entry to evict.

        Pure LRU mode: the oldest-accessed entry (first in OrderedDict).
        Hit-weighted mode: scan the N oldest entries and pick the one
        with the LOWEST weighted score (least valuable). O(scan_size).
        """
        if not self._hit_weighted:
            # Pure LRU — oldest access wins eviction. O(1).
            return next(iter(self._store))

        # Hit-weighted: scan up to `eviction_scan_size` oldest entries
        # and pick the one minimising `alpha*recency + (1-alpha)*log1p(hits)`.
        # `recency` here is just rank-in-oldest (0 = oldest, 1 = next-
        # oldest …) normalised over the scan window — keeps scoring cheap.
        import math
        scan_size = min(self._eviction_scan_size, len(self._store))
        alpha = self._hit_weight_alpha
        candidates: list[tuple[str, int]] = []
        for i, (k, (_, _, hits)) in enumerate(self._store.items()):
            if i >= scan_size:
                break
            candidates.append((k, hits))
        # Normalise recency rank into [0,1] over scan window
        best_key = candidates[0][0]
        best_score = float("inf")
        n = max(1, scan_size - 1)
        for rank, (k, hits) in enumerate(candidates):
            recency = rank / n  # 0=oldest, approaches 1 for recent
            # Lower score = more evictable.
            # Oldest entries (high recency value) contribute high
            # `alpha * recency`; few hits contribute low
            # `(1-alpha) * log1p(hits)`. We want to evict the entry
            # whose combined "recency-is-old + few-hits" is LOWEST,
            # i.e. evict old+unpopular, protect recent-or-popular.
            # Flip sign on recency so oldest==most-evictable (lowest score).
            score = alpha * (1.0 - recency) + (1.0 - alpha) * math.log1p(hits)
            if score < best_score:
                best_score = score
                best_key = k
        self._hit_weighted_evictions += 1
        return best_key

    def invalidate(
        self,
        prompt: str,
        model: str,
        sampling_params: Mapping[str, Any],
    ) -> bool:
        """Drop a specific key. Returns True if it was present."""
        key = _stable_key(prompt, model, sampling_params)
        with self._lock:
            return self._store.pop(key, None) is not None

    def clear(self) -> None:
        """Drop ALL entries. Counters preserved for diagnostics."""
        with self._lock:
            self._store.clear()

    # ── introspection ─────────────────────────────────────────────

    def stats(self) -> dict:
        """Current cache stats — thread-safe snapshot."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total) if total else 0.0
            # Top-N most-hit keys (for diagnostic insight into what's
            # getting re-queried most).
            top_n = sorted(
                (
                    (hits, k) for k, (_, _, hits) in self._store.items()
                ),
                reverse=True,
            )[:5]
            return {
                "size": len(self._store),
                "max_entries": self._max_entries,
                "ttl_seconds": self._ttl_seconds,
                "hit_weighted_eviction": self._hit_weighted,
                "hit_weight_alpha": self._hit_weight_alpha,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
                "ttl_evictions": self._ttl_evictions,
                "hit_weighted_evictions": self._hit_weighted_evictions,
                "top_hit_counts": [hc for hc, _ in top_n],
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# ── default singleton ───────────────────────────────────────────
_DEFAULT_CACHE: Optional[ResponseCacheLRU] = None
_DEFAULT_CACHE_LOCK = threading.Lock()


def get_default_cache() -> Optional[ResponseCacheLRU]:
    """Return the process-wide default cache, or None if P41 is disabled.

    Lazily constructed on first call; subsequent calls return the same
    instance. Re-reads env on each call-when-None so tests can
    monkey-patch `GENESIS_ENABLE_P41_RESPONSE_CACHE` after process
    start without reimport gymnastics.
    """
    global _DEFAULT_CACHE
    if not is_p41_enabled():
        return None
    if _DEFAULT_CACHE is not None:
        return _DEFAULT_CACHE
    with _DEFAULT_CACHE_LOCK:
        if _DEFAULT_CACHE is not None:  # double-check
            return _DEFAULT_CACHE
        max_entries = _resolve_int_env(_ENV_MAX_ENTRIES, 1024)
        ttl = _resolve_int_env(_ENV_TTL_SECONDS, 3600)
        hit_weighted = _read_bool_env(_ENV_HIT_WEIGHTED)
        alpha = _read_float_env(_ENV_HIT_ALPHA, 0.3)
        backend = os.environ.get(_ENV_BACKEND, "memory").strip().lower()

        if backend == "redis":
            # Redis backend: shared across processes + restart-survivable.
            # Import lazily — `redis` is not a hard dep.
            try:
                from vllm._genesis.cache.redis_backend import (
                    RedisResponseCache,
                )
                url = os.environ.get(
                    _ENV_REDIS_URL, "redis://localhost:6379/1",
                )
                _DEFAULT_CACHE = RedisResponseCache(
                    redis_url=url, ttl_seconds=ttl,
                )
                log.info(
                    "[P41] Redis-backed response cache initialised: "
                    "url=%s ttl=%ds", url, ttl,
                )
                return _DEFAULT_CACHE
            except Exception as e:
                log.warning(
                    "[P41] Redis backend init failed (%s); falling back to "
                    "in-memory LRU", e,
                )
                # Fall through to in-memory
        _DEFAULT_CACHE = ResponseCacheLRU(
            max_entries=max_entries, ttl_seconds=ttl,
            hit_weighted_eviction=hit_weighted,
            hit_weight_alpha=alpha,
        )
        log.info(
            "[P41] default response cache initialised: max_entries=%d "
            "ttl=%ds hit_weighted=%s alpha=%.2f",
            max_entries, ttl, hit_weighted, alpha,
        )
    return _DEFAULT_CACHE


def reset_default_cache_for_tests() -> None:
    """Drop the singleton so the next `get_default_cache()` re-reads env.

    TESTS ONLY. Calling at runtime invalidates existing hit-rate stats
    but does not affect correctness."""
    global _DEFAULT_CACHE
    with _DEFAULT_CACHE_LOCK:
        _DEFAULT_CACHE = None
