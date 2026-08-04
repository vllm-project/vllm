# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the L2 adapter base class accounting + AdapterUsage.

The base class owns ``_total_bytes_used`` / ``_bytes_by_cache_salt`` and
exposes them through ``get_usage()``. Adapters drive accounting by passing
``sizes`` to ``_notify_keys_stored`` / ``_notify_keys_deleted``.
"""

# Standard
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L2AdapterListener, L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import (
    AdapterUsage,
    L2AdapterInterface,
    L2TaskId,
)


def _make_key(chunk_id: int, salt: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="m",
        kv_rank=0,
        cache_salt=salt,
    )


class _StubAdapter(L2AdapterInterface):
    """Minimal adapter that satisfies the abstract surface so we can
    exercise base-class behavior in isolation."""

    def get_store_event_fd(self) -> int:
        return -1

    def get_lookup_and_lock_event_fd(self) -> int:
        return -1

    def get_load_event_fd(self) -> int:
        return -1

    def submit_store_task(self, keys, objects):
        return 0

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        return {}

    def submit_lookup_and_lock_task(self, keys, layout_desc: MemoryLayoutDesc):
        return 0

    def query_lookup_and_lock_result(self, task_id):
        return None

    def submit_unlock(self, keys):
        return None

    def submit_load_task(self, keys, objects):
        return 0

    def query_load_result(self, task_id):
        return None

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# AdapterUsage
# ---------------------------------------------------------------------------


class TestAdapterUsage:
    def test_usage_fraction_in_range(self):
        u = AdapterUsage(total_bytes_used=50, total_capacity_bytes=100)
        assert u.usage_fraction == 0.5

    def test_usage_fraction_zero_capacity_returns_minus_one(self):
        # ``-1`` is the legacy "no eviction signal" sentinel — kept so
        # eviction-controller callers can keep using ``< 0`` shortcuts.
        u = AdapterUsage(total_bytes_used=0, total_capacity_bytes=0)
        assert u.usage_fraction == -1.0

    def test_usage_fraction_negative_capacity_returns_minus_one(self):
        u = AdapterUsage(total_bytes_used=10, total_capacity_bytes=-1)
        assert u.usage_fraction == -1.0

    def test_bytes_by_cache_salt_default_empty(self):
        u = AdapterUsage(total_bytes_used=0, total_capacity_bytes=100)
        assert u.bytes_by_cache_salt == {}

    def test_frozen(self):
        # Standard
        from dataclasses import FrozenInstanceError

        u = AdapterUsage(total_bytes_used=1, total_capacity_bytes=2)
        with pytest.raises(FrozenInstanceError):
            u.total_bytes_used = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# supports_global_eviction
# ---------------------------------------------------------------------------


class TestSupportsGlobalEviction:
    def test_default_no_capacity_means_no_eviction(self):
        a = _StubAdapter()
        assert a.supports_global_eviction is False
        assert a.get_usage().usage_fraction == -1.0

    def test_positive_capacity_supports_global_eviction(self):
        a = _StubAdapter(max_capacity_bytes=1024)
        assert a.supports_global_eviction is True

    def test_zero_capacity_explicit_means_no_eviction(self):
        a = _StubAdapter(max_capacity_bytes=0)
        assert a.supports_global_eviction is False


# ---------------------------------------------------------------------------
# Base-class byte accounting
# ---------------------------------------------------------------------------


class TestBaseAccounting:
    def test_store_increments_aggregate_and_by_cache_salt(self):
        a = _StubAdapter(max_capacity_bytes=10_000)
        k_alice = _make_key(1, salt="alice")
        k_bob = _make_key(2, salt="bob")
        k_alice2 = _make_key(3, salt="alice")
        a._notify_keys_stored([k_alice, k_bob, k_alice2], [100, 200, 50])
        u = a.get_usage()
        assert u.total_bytes_used == 350
        assert u.bytes_by_cache_salt == {"alice": 150, "bob": 200}

    def test_delete_decrements_aggregate_and_by_cache_salt(self):
        a = _StubAdapter(max_capacity_bytes=10_000)
        k_alice = _make_key(1, salt="alice")
        k_bob = _make_key(2, salt="bob")
        a._notify_keys_stored([k_alice, k_bob], [100, 200])
        a._notify_keys_deleted([k_alice], [100])
        u = a.get_usage()
        assert u.total_bytes_used == 200
        assert u.bytes_by_cache_salt == {"bob": 200}

    def test_cache_salt_bucket_dropped_when_zero(self):
        """Per-``cache_salt`` bookkeeping should not retain stale ``0``
        entries — it keeps the snapshot compact and avoids memory
        growth across many short-lived salts."""
        a = _StubAdapter(max_capacity_bytes=10_000)
        k = _make_key(1, salt="alice")
        a._notify_keys_stored([k], [100])
        a._notify_keys_deleted([k], [100])
        u = a.get_usage()
        assert u.total_bytes_used == 0
        assert "alice" not in u.bytes_by_cache_salt
        assert u.bytes_by_cache_salt == {}

    def test_empty_salt_is_a_real_bucket(self):
        """Un-salted traffic accumulates under the empty-string key.
        ``bytes_by_cache_salt`` must reflect that so legacy/unisolated traffic
        is observable too."""
        a = _StubAdapter(max_capacity_bytes=10_000)
        k = _make_key(1, salt="")
        a._notify_keys_stored([k], [100])
        u = a.get_usage()
        assert u.bytes_by_cache_salt == {"": 100}

    def test_get_usage_filters_zero_buckets(self):
        """Snapshot only includes positive-byte buckets."""
        a = _StubAdapter(max_capacity_bytes=10_000)
        # Manually inject a ``0`` entry as if accounting drift left one
        # behind.
        a._bytes_by_cache_salt["ghost"] = 0
        a._bytes_by_cache_salt["alice"] = 100
        a._total_bytes_used = 100
        u = a.get_usage()
        assert u.bytes_by_cache_salt == {"alice": 100}

    def test_size_list_length_mismatch_raises(self):
        """``zip(strict=True)`` catches caller bugs where keys/sizes drift
        out of sync — that would silently corrupt accounting otherwise."""
        a = _StubAdapter(max_capacity_bytes=10_000)
        with pytest.raises(ValueError):
            a._notify_keys_stored([_make_key(1)], [100, 200])

    def test_underflow_clamps_to_zero(self):
        """A delete that would drive ``_total_bytes_used`` negative
        clamps to 0 (and logs a warning, observable via stderr).
        Without the clamp the sentinel ``usage_fraction == -1`` would
        silently disable eviction across the whole adapter."""
        a = _StubAdapter(max_capacity_bytes=10_000)
        k = _make_key(1, salt="alice")
        a._notify_keys_stored([k], [100])
        # Delete reports a larger size than was stored — caller bug.
        a._notify_keys_deleted([k], [500])
        u = a.get_usage()
        assert u.total_bytes_used == 0  # clamped, not -400
        assert u.usage_fraction == 0.0  # not the -1 sentinel

    def test_get_usage_returns_immutable_by_cache_salt_view(self):
        """``bytes_by_cache_salt`` is a read-only ``Mapping`` so a caller
        cannot mutate the snapshot or the adapter's live state."""
        a = _StubAdapter(max_capacity_bytes=10_000)
        k = _make_key(1, salt="alice")
        a._notify_keys_stored([k], [100])
        u = a.get_usage()
        # MappingProxyType raises TypeError on mutation attempts.
        with pytest.raises(TypeError):
            u.bytes_by_cache_salt["bob"] = 999  # type: ignore[index]
        # Original snapshot still intact.
        assert dict(u.bytes_by_cache_salt) == {"alice": 100}

    def test_get_usage_snapshot_is_detached(self):
        """A held ``AdapterUsage`` reference does not see later
        accounting changes — each call returns a fresh snapshot."""
        a = _StubAdapter(max_capacity_bytes=10_000)
        k1 = _make_key(1, salt="alice")
        a._notify_keys_stored([k1], [100])
        u1 = a.get_usage()
        a._notify_keys_stored([_make_key(2, salt="bob")], [200])
        # u1 is the snapshot at the earlier moment.
        assert u1.total_bytes_used == 100
        assert dict(u1.bytes_by_cache_salt) == {"alice": 100}
        # Live state is now 300 / {alice, bob}.
        u2 = a.get_usage()
        assert u2.total_bytes_used == 300
        assert dict(u2.bytes_by_cache_salt) == {"alice": 100, "bob": 200}

    def test_zero_size_notify_is_listener_only(self):
        """Calling ``_notify_keys_stored`` with size=0 fires the listener
        but leaves byte counters untouched. Used by the native connector
        to bump LRU on a re-store without double-counting bytes."""
        a = _StubAdapter(max_capacity_bytes=10_000)
        lst = _RecordingListener()
        a.register_listener(lst)
        k = _make_key(1, salt="alice")
        a._notify_keys_stored([k], [100])
        # Re-store with size=0 — listener fires, bytes unchanged.
        a._notify_keys_stored([k], [0])
        u = a.get_usage()
        assert u.total_bytes_used == 100
        assert dict(u.bytes_by_cache_salt) == {"alice": 100}
        # Listener fired twice (LRU policy can move_to_end on second).
        assert lst.stored == [[k], [k]]

    def test_concurrent_stores_thread_safe(self):
        """Accounting is held under ``_usage_lock``; concurrent notifies
        from many threads must not lose updates."""
        a = _StubAdapter(max_capacity_bytes=10_000_000)
        n_threads = 8
        per_thread = 100

        def worker(idx: int):
            keys = [_make_key(i + idx * per_thread) for i in range(per_thread)]
            sizes = [10] * per_thread
            a._notify_keys_stored(keys, sizes)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert a.get_usage().total_bytes_used == n_threads * per_thread * 10


# ---------------------------------------------------------------------------
# Listener notifications still fire (regression guard for the refactor)
# ---------------------------------------------------------------------------


class _RecordingListener(L2AdapterListener):
    """Concrete ``L2AdapterListener`` that captures notify calls so tests
    can assert listener invocation."""

    def __init__(self) -> None:
        self.stored: list[list[ObjectKey]] = []
        self.accessed: list[list[ObjectKey]] = []
        self.deleted: list[list[ObjectKey]] = []

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        self.stored.append(list(keys))

    def on_l2_keys_accessed(self, keys: list[ObjectKey]) -> None:
        self.accessed.append(list(keys))

    def on_l2_keys_deleted(self, keys: list[ObjectKey]) -> None:
        self.deleted.append(list(keys))


class TestListenersStillFire:
    def test_store_notify_fires_listener(self):
        a = _StubAdapter(max_capacity_bytes=1000)
        lst = _RecordingListener()
        a.register_listener(lst)
        k = _make_key(1, salt="alice")
        a._notify_keys_stored([k], [100])
        assert lst.stored == [[k]]

    def test_delete_notify_fires_listener(self):
        a = _StubAdapter(max_capacity_bytes=1000)
        lst = _RecordingListener()
        a.register_listener(lst)
        k = _make_key(1, salt="alice")
        a._notify_keys_stored([k], [100])
        a._notify_keys_deleted([k], [100])
        assert lst.deleted == [[k]]

    def test_accessed_notify_unchanged_signature(self):
        """``_notify_keys_accessed`` does not affect bytes — its signature
        is unchanged from before the refactor."""
        a = _StubAdapter()
        lst = _RecordingListener()
        a.register_listener(lst)
        k = _make_key(1)
        a._notify_keys_accessed([k])
        assert lst.accessed == [[k]]
