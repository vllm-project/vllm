# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator L2UsageManager."""

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.mp_coordinator.cache_control.usage_manager import L2UsageManager


def _key(chunk: int = 0, salt: str = "a") -> ObjectKey:
    return ObjectKey(
        chunk_hash=chunk.to_bytes(4, "big"),
        model_name="m",
        kv_rank=0,
        cache_salt=salt,
    )


# =============================================================================
# record_stored — basic accumulation
# =============================================================================


def test_record_stored_basic():
    t = L2UsageManager()
    t.record_stored(_key(0, salt="a"), 100)
    assert t.get("a") == 100
    assert t.get_total() == 100
    assert t.has_key(_key(0, salt="a"))


def test_record_stored_distinct_keys_accumulate():
    t = L2UsageManager()
    t.record_stored(_key(0, salt="a"), 100)
    t.record_stored(_key(1, salt="a"), 200)
    assert t.get("a") == 300
    assert t.get_total() == 300


# =============================================================================
# record_stored — replace-on-existing (the new contract)
# =============================================================================


def test_record_stored_same_key_same_size_is_idempotent():
    t = L2UsageManager()
    k = _key(0, salt="a")
    t.record_stored(k, 100)
    t.record_stored(k, 100)
    # Stored twice, but the per-salt total stays at 100 — no
    # double-counting.
    assert t.get("a") == 100
    assert t.get_total() == 100


def test_record_stored_same_key_new_size_replaces():
    t = L2UsageManager()
    k = _key(0, salt="a")
    t.record_stored(k, 100)
    t.record_stored(k, 250)
    # New size replaces the old one; delta (+150) lands on the total.
    assert t.get_key_size(k) == 250
    assert t.get("a") == 250
    assert t.get_total() == 250


def test_record_stored_shrink_adjusts_down():
    t = L2UsageManager()
    k = _key(0, salt="a")
    t.record_stored(k, 100)
    t.record_stored(k, 40)
    assert t.get("a") == 40
    assert t.get_total() == 40


# =============================================================================
# record_evicted — key-aware
# =============================================================================


def test_record_evicted_returns_freed_bytes():
    t = L2UsageManager()
    k = _key(0, salt="a")
    t.record_stored(k, 100)
    assert t.record_evicted(k) == 100
    assert t.get("a") == 0
    assert t.get_total() == 0
    assert not t.has_key(k)


def test_record_evicted_unknown_key_returns_zero():
    t = L2UsageManager()
    assert t.record_evicted(_key(99, salt="z")) == 0


def test_record_evicted_removes_zero_entry():
    t = L2UsageManager()
    k = _key(0, salt="a")
    t.record_stored(k, 100)
    t.record_evicted(k)
    assert t.get_all() == {}


def test_record_evicted_partial_when_multiple_keys():
    t = L2UsageManager()
    t.record_stored(_key(0, salt="a"), 100)
    t.record_stored(_key(1, salt="a"), 200)
    t.record_evicted(_key(0, salt="a"))
    # Bucket only loses the evicted key's bytes.
    assert t.get("a") == 200
    assert t.get_total() == 200


# =============================================================================
# Existence query
# =============================================================================


def test_has_key_returns_false_for_unknown():
    t = L2UsageManager()
    assert not t.has_key(_key(0))


def test_get_key_size_returns_none_for_unknown():
    t = L2UsageManager()
    assert t.get_key_size(_key(0)) is None


# =============================================================================
# Multi-salt + zero / negative
# =============================================================================


def test_multiple_salts():
    t = L2UsageManager()
    t.record_stored(_key(0, salt="a"), 100)
    t.record_stored(_key(0, salt="b"), 200)
    assert t.get("a") == 100
    assert t.get("b") == 200
    assert t.get_total() == 300


def test_get_unknown_returns_zero():
    t = L2UsageManager()
    assert t.get("unknown") == 0


def test_get_all():
    t = L2UsageManager()
    t.record_stored(_key(0, salt="a"), 100)
    t.record_stored(_key(0, salt="b"), 200)
    assert t.get_all() == {"a": 100, "b": 200}


def test_get_all_empty():
    t = L2UsageManager()
    assert t.get_all() == {}


def test_zero_bytes_stores_entry_without_changing_total():
    t = L2UsageManager()
    k = _key(0, salt="a")
    t.record_stored(k, 0)
    # The key is now tracked at size 0; the per-salt total doesn't move.
    assert t.has_key(k)
    assert t.get("a") == 0
    assert t.get_total() == 0


def test_negative_store_raises():
    t = L2UsageManager()
    with pytest.raises(ValueError, match="non-negative"):
        t.record_stored(_key(0), -1)
