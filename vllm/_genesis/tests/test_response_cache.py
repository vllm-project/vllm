# SPDX-License-Identifier: Apache-2.0
"""TDD tests for P41 — Genesis response cache.

Covers:
- Key stability: same (prompt, model, params) → same hash regardless
  of dict insertion order
- `None`-valued params drop from key (collision with missing-key)
- `ResponseCacheLRU.get/store` round-trip correctness
- TTL eviction
- LRU eviction when exceeding max_entries
- Thread-safety: concurrent `get`/`store` doesn't corrupt counters
- `stats()` counter semantics
- `invalidate()` / `clear()` behaviour
- `is_p41_enabled()` respects truthy env values
- `get_default_cache()` singleton and reset_for_tests
- Never mutates the caller's params dict

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import threading

import pytest


@pytest.fixture(autouse=True)
def _reset_env_and_cache(monkeypatch):
    from vllm._genesis.cache.response_cache import reset_default_cache_for_tests
    monkeypatch.delenv("GENESIS_ENABLE_P41_RESPONSE_CACHE", raising=False)
    monkeypatch.delenv("GENESIS_P41_MAX_ENTRIES", raising=False)
    monkeypatch.delenv("GENESIS_P41_TTL_SECONDS", raising=False)
    reset_default_cache_for_tests()
    yield
    reset_default_cache_for_tests()


class TestKeyStability:
    def test_same_key_across_dict_orderings(self):
        from vllm._genesis.cache.response_cache import _stable_key
        p = "hello world"
        m = "qwen-test"
        k1 = _stable_key(p, m, {"temperature": 0.0, "max_tokens": 64, "top_p": 1.0})
        k2 = _stable_key(p, m, {"max_tokens": 64, "top_p": 1.0, "temperature": 0.0})
        assert k1 == k2

    def test_none_values_collide_with_missing(self):
        from vllm._genesis.cache.response_cache import _stable_key
        k_none = _stable_key("p", "m", {"stop": None, "n": 1})
        k_missing = _stable_key("p", "m", {"n": 1})
        assert k_none == k_missing

    def test_different_prompts_different_keys(self):
        from vllm._genesis.cache.response_cache import _stable_key
        k1 = _stable_key("hello", "m", {"t": 0.0})
        k2 = _stable_key("world", "m", {"t": 0.0})
        assert k1 != k2

    def test_different_models_different_keys(self):
        from vllm._genesis.cache.response_cache import _stable_key
        k1 = _stable_key("p", "qwen-a", {"t": 0.0})
        k2 = _stable_key("p", "qwen-b", {"t": 0.0})
        assert k1 != k2


class TestRoundTrip:
    def test_get_miss_returns_none(self):
        from vllm._genesis.cache.response_cache import ResponseCacheLRU
        c = ResponseCacheLRU(max_entries=4, ttl_seconds=60)
        assert c.get("p", "m", {"t": 0.0}) is None
        assert c.stats()["misses"] == 1
        assert c.stats()["hits"] == 0

    def test_store_then_get_returns_stored(self):
        from vllm._genesis.cache.response_cache import ResponseCacheLRU
        c = ResponseCacheLRU(max_entries=4, ttl_seconds=60)
        resp = {"choices": [{"text": "hi"}], "usage": {"total_tokens": 5}}
        c.store("p", "m", {"t": 0.0}, resp)
        out = c.get("p", "m", {"t": 0.0})
        assert out == resp
        assert c.stats()["hits"] == 1

    def test_refresh_same_key_overwrites(self):
        from vllm._genesis.cache.response_cache import ResponseCacheLRU
        c = ResponseCacheLRU(max_entries=4, ttl_seconds=60)
        c.store("p", "m", {"t": 0.0}, {"v": 1})
        c.store("p", "m", {"t": 0.0}, {"v": 2})
        assert c.get("p", "m", {"t": 0.0}) == {"v": 2}
        assert len(c) == 1


class TestTTLEviction:
    def test_expired_entry_returns_none_and_drops(self, monkeypatch):
        from vllm._genesis.cache import response_cache as rc
        c = rc.ResponseCacheLRU(max_entries=4, ttl_seconds=60)
        base = 1000.0
        monkeypatch.setattr(rc.time, "monotonic", lambda: base)
        c.store("p", "m", {}, {"v": 1})
        # Jump 61 seconds ahead
        monkeypatch.setattr(rc.time, "monotonic", lambda: base + 61.0)
        assert c.get("p", "m", {}) is None
        assert c.stats()["ttl_evictions"] == 1
        assert len(c) == 0

    def test_fresh_entry_returns(self, monkeypatch):
        from vllm._genesis.cache import response_cache as rc
        c = rc.ResponseCacheLRU(max_entries=4, ttl_seconds=60)
        base = 1000.0
        monkeypatch.setattr(rc.time, "monotonic", lambda: base)
        c.store("p", "m", {}, {"v": 1})
        monkeypatch.setattr(rc.time, "monotonic", lambda: base + 30.0)
        assert c.get("p", "m", {}) == {"v": 1}


class TestLRUEviction:
    def test_exceeds_capacity_evicts_oldest(self):
        from vllm._genesis.cache.response_cache import ResponseCacheLRU
        c = ResponseCacheLRU(max_entries=3, ttl_seconds=3600)
        for i in range(5):
            c.store(f"p{i}", "m", {}, {"v": i})
        assert len(c) == 3
        # 0 and 1 should be evicted (oldest insertions)
        assert c.get("p0", "m", {}) is None
        assert c.get("p1", "m", {}) is None
        assert c.get("p2", "m", {}) == {"v": 2}
        assert c.get("p4", "m", {}) == {"v": 4}
        assert c.stats()["evictions"] == 2

    def test_access_refreshes_lru_order(self):
        from vllm._genesis.cache.response_cache import ResponseCacheLRU
        c = ResponseCacheLRU(max_entries=2, ttl_seconds=3600)
        c.store("a", "m", {}, {"v": "a"})
        c.store("b", "m", {}, {"v": "b"})
        # Touch "a" — makes it MRU
        _ = c.get("a", "m", {})
        # Insert "c" — should evict "b" (LRU), not "a"
        c.store("c", "m", {}, {"v": "c"})
        assert c.get("a", "m", {}) == {"v": "a"}
        assert c.get("b", "m", {}) is None
        assert c.get("c", "m", {}) == {"v": "c"}


class TestInvalidateClear:
    def test_invalidate_present(self):
        from vllm._genesis.cache.response_cache import ResponseCacheLRU
        c = ResponseCacheLRU()
        c.store("p", "m", {}, {"v": 1})
        assert c.invalidate("p", "m", {}) is True
        assert c.get("p", "m", {}) is None

    def test_invalidate_absent(self):
        from vllm._genesis.cache.response_cache import ResponseCacheLRU
        c = ResponseCacheLRU()
        assert c.invalidate("p", "m", {}) is False

    def test_clear_empties(self):
        from vllm._genesis.cache.response_cache import ResponseCacheLRU
        c = ResponseCacheLRU()
        for i in range(5):
            c.store(f"p{i}", "m", {}, {"v": i})
        assert len(c) == 5
        c.clear()
        assert len(c) == 0
        # Counters preserved
        stats = c.stats()
        assert "hits" in stats
        assert "misses" in stats


class TestThreadSafety:
    def test_concurrent_get_and_store_no_counter_corruption(self):
        from vllm._genesis.cache.response_cache import ResponseCacheLRU
        c = ResponseCacheLRU(max_entries=200, ttl_seconds=3600)
        # Prime 50 entries
        for i in range(50):
            c.store(f"p{i}", "m", {}, {"v": i})

        errors = []

        def worker():
            try:
                for i in range(100):
                    # Mix: 90% get (50% hit), 10% store
                    if i % 10 == 0:
                        c.store(f"new_{threading.get_ident()}_{i}", "m", {}, {"v": i})
                    else:
                        c.get(f"p{i % 60}", "m", {})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        stats = c.stats()
        assert stats["hits"] + stats["misses"] > 0


class TestIsEnabled:
    def test_default_off(self):
        from vllm._genesis.cache.response_cache import is_p41_enabled
        assert is_p41_enabled() is False

    def test_truthy_values_on(self, monkeypatch):
        from vllm._genesis.cache.response_cache import is_p41_enabled
        for v in ("1", "true", "TRUE", "yes", "Yes", "on"):
            monkeypatch.setenv("GENESIS_ENABLE_P41_RESPONSE_CACHE", v)
            assert is_p41_enabled() is True, f"Should accept {v!r}"

    def test_falsy_values_off(self, monkeypatch):
        from vllm._genesis.cache.response_cache import is_p41_enabled
        for v in ("0", "false", "no", "off", ""):
            monkeypatch.setenv("GENESIS_ENABLE_P41_RESPONSE_CACHE", v)
            assert is_p41_enabled() is False, f"Should reject {v!r}"


class TestDefaultSingleton:
    def test_none_when_disabled(self):
        from vllm._genesis.cache.response_cache import get_default_cache
        assert get_default_cache() is None

    def test_singleton_when_enabled(self, monkeypatch):
        from vllm._genesis.cache.response_cache import (
            get_default_cache, ResponseCacheLRU,
        )
        monkeypatch.setenv("GENESIS_ENABLE_P41_RESPONSE_CACHE", "1")
        c1 = get_default_cache()
        c2 = get_default_cache()
        assert c1 is not None
        assert c1 is c2
        assert isinstance(c1, ResponseCacheLRU)

    def test_env_overrides_max_entries_and_ttl(self, monkeypatch):
        from vllm._genesis.cache.response_cache import get_default_cache
        monkeypatch.setenv("GENESIS_ENABLE_P41_RESPONSE_CACHE", "1")
        monkeypatch.setenv("GENESIS_P41_MAX_ENTRIES", "50")
        monkeypatch.setenv("GENESIS_P41_TTL_SECONDS", "120")
        c = get_default_cache()
        assert c is not None
        assert c.stats()["max_entries"] == 50
        assert c.stats()["ttl_seconds"] == 120


class TestParamsImmutability:
    def test_get_doesnt_mutate_params_dict(self):
        from vllm._genesis.cache.response_cache import ResponseCacheLRU
        c = ResponseCacheLRU()
        params = {"t": 0.0, "n": 1}
        snapshot = dict(params)
        c.get("p", "m", params)
        assert params == snapshot

    def test_store_doesnt_mutate_params_dict(self):
        from vllm._genesis.cache.response_cache import ResponseCacheLRU
        c = ResponseCacheLRU()
        params = {"t": 0.0, "n": 1, "stop": None}
        snapshot = dict(params)
        c.store("p", "m", params, {"v": 1})
        assert params == snapshot


class TestConstructorValidation:
    def test_rejects_zero_max(self):
        from vllm._genesis.cache.response_cache import ResponseCacheLRU
        with pytest.raises(ValueError):
            ResponseCacheLRU(max_entries=0, ttl_seconds=60)

    def test_rejects_negative_ttl(self):
        from vllm._genesis.cache.response_cache import ResponseCacheLRU
        with pytest.raises(ValueError):
            ResponseCacheLRU(max_entries=4, ttl_seconds=0)

    def test_rejects_alpha_out_of_range(self):
        from vllm._genesis.cache.response_cache import ResponseCacheLRU
        with pytest.raises(ValueError):
            ResponseCacheLRU(hit_weight_alpha=1.5)
        with pytest.raises(ValueError):
            ResponseCacheLRU(hit_weight_alpha=-0.1)

    def test_rejects_zero_scan_size(self):
        from vllm._genesis.cache.response_cache import ResponseCacheLRU
        with pytest.raises(ValueError):
            ResponseCacheLRU(eviction_scan_size=0)


class TestHitWeightedEviction:
    """Regression tests for v7.7 hit-rate-based eviction extension."""

    def test_default_is_pure_lru(self):
        """hit_weighted_eviction=False → oldest always wins."""
        from vllm._genesis.cache.response_cache import ResponseCacheLRU
        c = ResponseCacheLRU(max_entries=3, ttl_seconds=3600)
        c.store("a", "m", {}, 1)
        c.store("b", "m", {}, 2)
        c.store("c", "m", {}, 3)
        # Hit "a" many times — it's already oldest, but under pure LRU
        # hit count doesn't matter.
        for _ in range(10):
            assert c.get("a", "m", {}) == 1
        # "a" was hit last → became MRU. Insert "d" → evict "b" (now oldest)
        c.store("d", "m", {}, 4)
        assert c.get("a", "m", {}) == 1  # hit
        assert c.get("b", "m", {}) is None  # evicted
        assert c.get("c", "m", {}) == 3
        assert c.get("d", "m", {}) == 4

    def test_hit_weighted_protects_popular(self):
        """With hit_weighted_eviction=True, an older-but-popular entry
        survives while newer-but-cold entries can be evicted."""
        from vllm._genesis.cache.response_cache import ResponseCacheLRU
        c = ResponseCacheLRU(
            max_entries=3, ttl_seconds=3600,
            hit_weighted_eviction=True, hit_weight_alpha=0.3,
            eviction_scan_size=3,
        )
        # Populate 3 entries (a = oldest, c = newest)
        c.store("a", "m", {}, 1)
        c.store("b", "m", {}, 2)
        c.store("c", "m", {}, 3)
        # Bump "a" hit count 20× without refreshing its store time.
        # Pure LRU would evict "a" on next insert (it's oldest by store).
        # Hit-weighted must protect it (high hit_count outweighs old recency).
        for _ in range(20):
            _ = c.get("a", "m", {})
        c.store("d", "m", {}, 4)
        # Under hit-weighted: "a" survives, some less-hit entry evicted.
        assert c.get("a", "m", {}) == 1, (
            "hit-weighted should protect the 20-hit entry against "
            "pure-LRU eviction"
        )
        # Exactly one of {b, c} should be evicted (d has 0 hits, "a"
        # has many — so the unpopular oldest among b/c goes).
        surviving = sum(
            c.get(k, "m", {}) is not None for k in ("b", "c", "d")
        )
        assert surviving == 2, (
            f"expected 2 surviving from (b,c,d) under hit-weighted; "
            f"got {surviving}"
        )

    def test_hit_weighted_counter_preserved_on_refresh(self):
        """store() on existing key must NOT reset hit_count."""
        from vllm._genesis.cache.response_cache import ResponseCacheLRU
        c = ResponseCacheLRU(
            max_entries=4, ttl_seconds=3600, hit_weighted_eviction=True,
        )
        c.store("a", "m", {}, 1)
        for _ in range(5):
            _ = c.get("a", "m", {})
        # Refresh payload — hit_count should remain 5
        c.store("a", "m", {}, 999)
        stats = c.stats()
        assert stats["top_hit_counts"][0] == 5

    def test_stats_reports_hit_counts_top_n(self):
        from vllm._genesis.cache.response_cache import ResponseCacheLRU
        c = ResponseCacheLRU(max_entries=10, ttl_seconds=3600)
        for i in range(5):
            c.store(f"p{i}", "m", {}, i)
            for _ in range(i * 3):
                _ = c.get(f"p{i}", "m", {})
        stats = c.stats()
        assert "top_hit_counts" in stats
        # Sorted descending
        tops = stats["top_hit_counts"]
        assert tops == sorted(tops, reverse=True)
        assert tops[0] == 12  # i=4 → 4*3


class TestRedisBackendImportGuard:
    """Redis backend must import cleanly and gracefully degrade when
    redis package or server is unavailable. Test without actual Redis
    — we just verify the import surface + error paths."""

    def test_import_module_succeeds(self):
        """Module import on CPU-only doesn't require redis package."""
        from vllm._genesis.cache import redis_backend
        assert hasattr(redis_backend, "RedisResponseCache")

    def test_missing_redis_raises_importerror(self, monkeypatch):
        """Constructing RedisResponseCache without redis pkg raises
        ImportError so `get_default_cache` can catch + fallback."""
        import sys
        # Hide `redis` module if present
        monkeypatch.setitem(sys.modules, "redis", None)
        from vllm._genesis.cache.redis_backend import RedisResponseCache
        with pytest.raises(ImportError):
            RedisResponseCache(redis_url="redis://localhost:0/0")

    def test_default_cache_falls_back_on_redis_failure(self, monkeypatch):
        """If backend=redis but Redis unreachable, get_default_cache
        falls back to in-memory LRU without raising."""
        from vllm._genesis.cache.response_cache import (
            get_default_cache, ResponseCacheLRU,
            reset_default_cache_for_tests,
        )
        monkeypatch.setenv("GENESIS_ENABLE_P41_RESPONSE_CACHE", "1")
        monkeypatch.setenv("GENESIS_P41_BACKEND", "redis")
        monkeypatch.setenv("GENESIS_P41_REDIS_URL", "redis://localhost:1/0")
        # Hide redis to simulate "package missing"
        import sys
        monkeypatch.setitem(sys.modules, "redis", None)
        reset_default_cache_for_tests()
        c = get_default_cache()
        assert c is not None
        # Must have fallen back to in-memory
        assert isinstance(c, ResponseCacheLRU)
