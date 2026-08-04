# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for :class:`QuotaManager`.
"""

# Standard
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.quota_manager import QuotaEntry, QuotaManager


class TestQuotaManagerBasics:
    def test_unregistered_salt_limit_is_zero(self):
        """Legacy allowlist semantics — anything unregistered has limit 0."""
        qm = QuotaManager()
        assert qm.get_limit_bytes("alice") == 0
        assert not qm.has_quota("alice")

    def test_set_and_get(self):
        qm = QuotaManager()
        qm.set_quota("alice", 1024)
        assert qm.get_limit_bytes("alice") == 1024
        assert qm.has_quota("alice")

    def test_set_overwrites(self):
        qm = QuotaManager()
        qm.set_quota("alice", 1024)
        qm.set_quota("alice", 2048)
        assert qm.get_limit_bytes("alice") == 2048

    def test_delete_returns_true_when_present(self):
        qm = QuotaManager()
        qm.set_quota("alice", 1024)
        assert qm.delete_quota("alice") is True
        assert not qm.has_quota("alice")
        assert qm.get_limit_bytes("alice") == 0

    def test_delete_returns_false_when_absent(self):
        qm = QuotaManager()
        assert qm.delete_quota("alice") is False

    def test_zero_limit_is_registered_and_distinguishable(self):
        """A zero limit IS a registration — ``has_quota`` returns True,
        so the entry shows up in ``list_quotas``. Evaluating the quota
        still yields 0, so the salt behaves like an unregistered one
        for eviction purposes."""
        qm = QuotaManager()
        qm.set_quota("alice", 0)
        assert qm.has_quota("alice")
        assert qm.get_limit_bytes("alice") == 0
        entries = qm.list_quotas()
        assert QuotaEntry(cache_salt="alice", limit_bytes=0) in entries

    def test_reject_negative_limit(self):
        qm = QuotaManager()
        with pytest.raises(ValueError, match="non-negative"):
            qm.set_quota("alice", -1)

    def test_empty_salt_is_a_valid_key(self):
        """``cache_salt=""`` is a real bucket (anonymous traffic) — the
        registry treats it like any other key."""
        qm = QuotaManager()
        qm.set_quota("", 512)
        assert qm.has_quota("")
        assert qm.get_limit_bytes("") == 512


class TestQuotaManagerList:
    def test_list_returns_snapshot(self):
        qm = QuotaManager()
        qm.set_quota("alice", 1)
        entries = qm.list_quotas()
        # Mutating the snapshot must not affect the registry.
        entries.clear()
        assert qm.has_quota("alice")

    def test_list_is_complete(self):
        qm = QuotaManager()
        qm.set_quota("alice", 100)
        qm.set_quota("bob", 200)
        qm.set_quota("", 50)
        got = {(e.cache_salt, e.limit_bytes) for e in qm.list_quotas()}
        assert got == {("alice", 100), ("bob", 200), ("", 50)}


class TestQuotaManagerThreadSafety:
    def test_concurrent_writes_do_not_corrupt(self):
        """Every write must land intact under contention."""
        qm = QuotaManager()
        n_writers = 8
        per_writer = 50
        barrier = threading.Barrier(n_writers)

        def writer(worker_id: int) -> None:
            barrier.wait()
            for i in range(per_writer):
                qm.set_quota(f"user-{worker_id}-{i}", worker_id * 1000 + i)

        threads = [threading.Thread(target=writer, args=(w,)) for w in range(n_writers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(qm.list_quotas()) == n_writers * per_writer
        # Spot-check a few values.
        assert qm.get_limit_bytes("user-3-7") == 3007
        assert qm.get_limit_bytes("user-0-0") == 0

    def test_concurrent_read_write(self):
        """Readers must never see a corrupted value."""
        qm = QuotaManager()
        qm.set_quota("alice", 1024)
        stop = threading.Event()
        errors: list[str] = []

        def reader():
            while not stop.is_set():
                v = qm.get_limit_bytes("alice")
                if v not in (1024, 2048):
                    errors.append(f"unexpected {v}")
                    return

        def writer():
            for _ in range(1000):
                qm.set_quota("alice", 2048)
                qm.set_quota("alice", 1024)

        t_r = threading.Thread(target=reader)
        t_w = threading.Thread(target=writer)
        t_r.start()
        t_w.start()
        t_w.join()
        stop.set()
        t_r.join()
        assert not errors, errors

    def test_concurrent_set_and_delete_on_same_key(self):
        """A set racing with a delete must leave the registry in a
        self-consistent state — either set wins (entry present) or
        delete wins (entry absent). Neither path should crash or leak
        a half-applied write."""
        qm = QuotaManager()
        iters = 500

        def setter():
            for _ in range(iters):
                qm.set_quota("alice", 1024)

        def deleter():
            for _ in range(iters):
                qm.delete_quota("alice")

        t1 = threading.Thread(target=setter)
        t2 = threading.Thread(target=deleter)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        # Final state is either present-with-1024 or absent. Anything
        # else (KeyError mid-read, partial write, etc.) would have
        # surfaced during the run.
        final = qm.get_limit_bytes("alice")
        assert final in (0, 1024)


class TestQuotaManagerEdgeCases:
    def test_very_large_limit(self):
        """Well inside a 64-bit int — shouldn't overflow or round."""
        qm = QuotaManager()
        big = 2**60
        qm.set_quota("alice", big)
        assert qm.get_limit_bytes("alice") == big

    def test_delete_then_read_returns_zero(self):
        """After deletion the salt is indistinguishable from an
        unregistered one — limit is the allowlist default of 0."""
        qm = QuotaManager()
        qm.set_quota("alice", 1024)
        qm.delete_quota("alice")
        assert qm.get_limit_bytes("alice") == 0
        assert not qm.has_quota("alice")

    def test_many_salts(self):
        """Sanity: large registries don't trip a size limit."""
        qm = QuotaManager()
        for i in range(10_000):
            qm.set_quota(f"user-{i}", i)
        assert len(qm.list_quotas()) == 10_000
        assert qm.get_limit_bytes("user-9999") == 9999


class TestQuotaManagerDefaultLimit:
    """Default-quota semantics used by the coordinator's eviction manager
    (``effective_limit_bytes``); ``get_limit_bytes`` keeps the legacy
    allowlist behavior for the MP server's local eviction controller."""

    def test_default_starts_unset(self):
        qm = QuotaManager()
        assert qm.get_default_limit_bytes() is None

    def test_effective_limit_none_for_unregistered_until_default_set(self):
        """Boot state: unregistered salts are exempt (None), even though
        the legacy ``get_limit_bytes`` still reports 0."""
        qm = QuotaManager()
        assert qm.effective_limit_bytes("alice") is None
        assert qm.get_limit_bytes("alice") == 0

    def test_default_zero_applies_to_unregistered(self):
        qm = QuotaManager()
        qm.set_default_limit_bytes(0)
        assert qm.get_default_limit_bytes() == 0
        assert qm.effective_limit_bytes("alice") == 0

    def test_positive_default_applies_to_unregistered(self):
        qm = QuotaManager()
        qm.set_default_limit_bytes(4096)
        assert qm.effective_limit_bytes("alice") == 4096

    def test_explicit_quota_wins_over_default(self):
        qm = QuotaManager()
        qm.set_default_limit_bytes(4096)
        qm.set_quota("alice", 1024)
        assert qm.effective_limit_bytes("alice") == 1024

    def test_explicit_zero_quota_wins_over_unset_default(self):
        """An explicit 0 is enforceable even while the default is None."""
        qm = QuotaManager()
        qm.set_quota("alice", 0)
        assert qm.effective_limit_bytes("alice") == 0
        assert qm.effective_limit_bytes("bob") is None

    def test_default_resettable_to_none(self):
        qm = QuotaManager()
        qm.set_default_limit_bytes(0)
        qm.set_default_limit_bytes(None)
        assert qm.get_default_limit_bytes() is None
        assert qm.effective_limit_bytes("alice") is None

    def test_negative_default_rejected(self):
        qm = QuotaManager()
        with pytest.raises(ValueError):
            qm.set_default_limit_bytes(-1)
