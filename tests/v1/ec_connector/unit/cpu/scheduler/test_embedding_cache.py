# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for EmbeddingCache.

Pure integer bookkeeping — no mmap, torch, or CUDA — so these run anywhere.
"""

import threading

import pytest

from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.embedding_cache import (
    EmbeddingCache,
)


def _cache(num_blocks: int = 8) -> EmbeddingCache:
    return EmbeddingCache(num_blocks)


# ── alloc ────────────────────────────────────────────────────────────────────


def test_alloc_returns_unique_block_ids():
    cache = _cache()
    entry = cache.alloc("a", 3)
    assert entry is not None
    assert len(entry.block_ids) == 3
    assert len(set(entry.block_ids)) == 3
    assert all(0 <= i < 8 for i in entry.block_ids)


def test_alloc_returns_tuple():
    cache = _cache()
    entry = cache.alloc("a", 2)
    assert isinstance(entry.block_ids, tuple)


def test_alloc_returns_none_when_full():
    cache = _cache(num_blocks=4)
    cache.alloc("a", 4)
    assert cache.alloc("b", 1) is None


def test_alloc_asserts_duplicate_key():
    cache = _cache()
    cache.alloc("a", 2)
    with pytest.raises(AssertionError, match="duplicate alloc"):
        cache.alloc("a", 2)


def test_alloc_asserts_n_blocks_exceeds_capacity():
    cache = _cache(num_blocks=4)
    with pytest.raises(AssertionError, match="capacity"):
        cache.alloc("a", 5)


def test_alloc_zero_blocks():
    cache = _cache()
    entry = cache.alloc("a", 0)
    assert entry is not None
    assert entry.block_ids == ()


def test_alloc_starts_not_ready():
    cache = _cache()
    entry = cache.alloc("a", 2)
    assert not entry.ready
    assert not entry.evictable


# ── get ──────────────────────────────────────────────────────────────────────


def test_get_missing_returns_none():
    cache = _cache()
    assert cache.get("a") is None


def test_get_returns_entry():
    cache = _cache()
    alloc_entry = cache.alloc("a", 2)
    got = cache.get("a")
    assert got is alloc_entry


def test_get_returns_not_ready_entry():
    cache = _cache()
    cache.alloc("a", 2)
    entry = cache.get("a")
    assert entry is not None
    assert not entry.ready


# ── mark_ready ───────────────────────────────────────────────────────────────


def test_mark_ready_transitions():
    cache = _cache()
    cache.alloc("a", 2)
    entry = cache.get("a")
    assert not entry.ready
    cache.mark_ready("a")
    assert entry.ready
    assert entry.evictable


def test_mark_ready_asserts_already_ready():
    cache = _cache()
    cache.alloc("a", 2)
    cache.mark_ready("a")
    with pytest.raises(AssertionError, match="already-ready"):
        cache.mark_ready("a")


def test_mark_ready_asserts_missing_key():
    cache = _cache()
    with pytest.raises(KeyError):
        cache.mark_ready("nonexistent")


# ── pin / unpin ──────────────────────────────────────────────────────────────


def test_pin_asserts_not_ready():
    cache = _cache()
    cache.alloc("a", 2)
    with pytest.raises(AssertionError, match="not-ready"):
        cache.pin("a")


def test_pin_unpin_basic():
    cache = _cache()
    cache.alloc("a", 2)
    cache.mark_ready("a")
    cache.pin("a")
    entry = cache.get("a")
    assert not entry.evictable
    cache.unpin("a")
    assert entry.evictable


def test_nested_pin():
    cache = _cache()
    cache.alloc("a", 2)
    cache.mark_ready("a")
    cache.pin("a")
    cache.pin("a")
    cache.unpin("a")
    entry = cache.get("a")
    assert not entry.evictable  # still pinned (count=1)
    cache.unpin("a")
    assert entry.evictable


def test_unpin_asserts_on_unpinned():
    cache = _cache()
    cache.alloc("a", 2)
    cache.mark_ready("a")
    with pytest.raises(AssertionError, match="unpinned"):
        cache.unpin("a")


# ── eviction ─────────────────────────────────────────────────────────────────


def test_eviction_frees_ready_unpinned_fifo():
    cache = _cache(num_blocks=4)
    cache.alloc("a", 2)
    cache.mark_ready("a")
    cache.alloc("b", 2)
    cache.mark_ready("b")
    # Pool full. Alloc should evict "a" (oldest).
    entry_c = cache.alloc("c", 2)
    assert entry_c is not None
    assert cache.get("a") is None  # evicted
    assert cache.get("b") is not None  # still there


def test_eviction_skips_pinned():
    cache = _cache(num_blocks=4)
    cache.alloc("a", 2)
    cache.mark_ready("a")
    cache.pin("a")
    cache.alloc("b", 2)
    cache.mark_ready("b")
    # "a" is pinned, so "b" should be evicted instead.
    entry_c = cache.alloc("c", 2)
    assert entry_c is not None
    assert cache.get("a") is not None  # pinned, survived
    assert cache.get("b") is None  # evicted
    cache.unpin("a")


def test_eviction_skips_not_ready():
    cache = _cache(num_blocks=4)
    cache.alloc("a", 2)  # not ready
    cache.alloc("b", 2)
    cache.mark_ready("b")
    # "a" is not ready (non-evictable), "b" is evictable.
    entry_c = cache.alloc("c", 2)
    assert entry_c is not None
    assert cache.get("a") is not None  # not ready, survived
    assert cache.get("b") is None  # evicted
    cache.mark_ready("a")  # clean up


def test_eviction_returns_none_if_all_non_evictable():
    cache = _cache(num_blocks=4)
    cache.alloc("a", 2)
    cache.mark_ready("a")
    cache.pin("a")
    cache.alloc("b", 2)  # not ready
    # Both entries are non-evictable (a=pinned, b=not-ready).
    assert cache.alloc("c", 1) is None
    # Fail-fast: nothing was evicted.
    assert cache.get("a") is not None
    assert cache.get("b") is not None
    cache.unpin("a")


def test_fail_fast_no_wasted_eviction():
    cache = _cache(num_blocks=6)
    cache.alloc("a", 2)
    cache.mark_ready("a")  # evictable: 2 blocks
    cache.alloc("b", 2)  # not ready
    cache.alloc("c", 2)  # not ready
    # free=0, evictable=2, need 3 -> cannot succeed
    assert cache.alloc("d", 3) is None
    # "a" was NOT evicted despite being evictable
    assert cache.get("a") is not None


def test_eviction_multiple_entries():
    cache = _cache(num_blocks=6)
    cache.alloc("a", 2)
    cache.mark_ready("a")
    cache.alloc("b", 2)
    cache.mark_ready("b")
    cache.alloc("c", 2)
    cache.mark_ready("c")
    # Need 4 blocks: must evict both "a" and "b".
    entry_d = cache.alloc("d", 4)
    assert entry_d is not None
    assert cache.get("a") is None
    assert cache.get("b") is None
    assert cache.get("c") is not None


# ── thread safety ────────────────────────────────────────────────────────────


def test_concurrent_pin_unpin():
    cache = _cache(num_blocks=8)
    cache.alloc("a", 4)
    cache.mark_ready("a")
    errors: list[Exception] = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        try:
            for _ in range(500):
                cache.pin("a")
                cache.unpin("a")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    entry = cache.get("a")
    assert entry.evictable  # all pins balanced


def test_concurrent_alloc_evict():
    cache = _cache(num_blocks=4)
    errors: list[Exception] = []
    barrier = threading.Barrier(4)
    counter = {"i": 0}
    counter_lock = threading.Lock()

    def worker():
        barrier.wait()
        try:
            for _ in range(200):
                with counter_lock:
                    key = f"k{counter['i']}"
                    counter["i"] += 1
                entry = cache.alloc(key, 1)
                if entry is not None:
                    cache.mark_ready(key)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
