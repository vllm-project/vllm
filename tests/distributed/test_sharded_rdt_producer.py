# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protocol tests for the sharded-RDT producer server (`_RDTProducerServer`).

The publish -> serve -> free_group -> release lifecycle is the subsystem's
concurrency core: a group whose barrier never completes stalls `end_sync`
forever, and a group freed too early drops CUDA-IPC storage while a consumer is
still reading it. Freeing is a PER-GROUP BARRIER: every live consumer signals
`free_group(gi)` at every owner of the group exactly once per sync, and the
group is released when the count reaches the live total handed to `begin_sync`
— one uniform integer, no routed per-producer targets.

These tests drive the real server. Everything here is CPU-clean: the IPC rebuild
is stubbed (`rebuild_cuda_tensor`) and the serve path is only entered as far as
the routing guard, which fires before any CUDA work.

`rdt_produce_weights_batched`'s pack/copy half needs a real GPU arena; it stays
covered by the GPU tests in `test_weight_transfer.py`.
"""

import gc
import threading
import time

import pytest
import torch

import vllm.distributed.weight_transfer.sharded_rdt_trainer as trainer_mod
from vllm.distributed.weight_transfer.sharded_rdt_trainer import (
    DEFAULT_GATHER_LOOKAHEAD,
    _RDTProducerServer,
)

# Publish/free are keyed by GROUP INDEX; the names ride the publish entries.
GI_A, GI_B = 0, 1
GROUP_A = ("model.layers.0.w", "model.layers.0.b")
GROUP_B = ("model.layers.1.w",)


@pytest.fixture
def server_factory(monkeypatch):
    """Build real `_RDTProducerServer` instances without a GPU.

    Three process-global side effects in `__init__` have to be neutralized:
    accelerator discovery (no driver here), the Ray NIXL monkeypatch (fail-soft
    already, but pointless), and `gc.freeze()` — which would otherwise leak a
    frozen GC generation into the pytest process for the rest of the session.
    """
    monkeypatch.setattr(
        torch.accelerator, "current_device_index", lambda *a, **kw: 0, raising=False
    )
    monkeypatch.setattr(gc, "freeze", lambda: None)

    # The IPC rebuild: hand back a plain CPU uint8 storage of the requested
    # size. publish_group then does the real .view(dtype) + as_strided on it, so
    # the view reconstruction under test is genuine.
    def _fake_rebuild(*args):
        return torch.arange(args[0], dtype=torch.uint8) if args else torch.empty(0)

    monkeypatch.setattr(trainer_mod, "rebuild_cuda_tensor", _fake_rebuild)

    built = []

    def _make(**kwargs):
        kwargs.setdefault("num_rdt_buffers", 2)
        kwargs.setdefault("arena_presize_gb", 0.0)
        kwargs.setdefault("pack_check", False)
        kwargs.setdefault("gather_lookahead", 2)
        server = _RDTProducerServer(**kwargs)
        built.append(server)
        return server

    yield _make
    for server in built:
        server.shutdown()


def _entries(names, *, nbytes=64, dtype="bfloat16"):
    """The `(storages, views)` payload the engine's gather loop ships: one
    storage export shared by every name, plus a per-name as_strided view spec.

    `storages` values stand in for `reduce_tensor`'s args. Only the arity
    matters here: `publish_group` overwrites index 6 with the server's own
    device index, so the tuple must be at least 7 long. The stubbed rebuild
    reads the byte count from index 0.
    """
    sid = 1
    storages = {sid: (nbytes, None, None, None, None, None, -1, None)}
    itemsize = getattr(torch, dtype).itemsize
    per = (nbytes // itemsize) // max(1, len(names))
    views = {name: (sid, dtype, [per], [1], i * per) for i, name in enumerate(names)}
    return storages, views


def _publish(server, gi, names, **kw):
    return server.publish_group(gi, _entries(names, **kw))


class TestPublishAndRebuild:
    def test_publish_makes_every_name_serveable(self, server_factory):
        server = server_factory()
        server.begin_sync(1)
        _publish(server, GI_A, GROUP_A)
        assert set(server._cache) == set(GROUP_A)

    def test_rebuilt_views_have_the_requested_dtype_and_geometry(self, server_factory):
        server = server_factory()
        server.begin_sync(1)
        _publish(server, GI_A, GROUP_A, nbytes=64, dtype="bfloat16")
        for name in GROUP_A:
            t = server._cache[name]
            assert t.dtype is torch.bfloat16
            assert tuple(t.shape) == (16,)  # 64 bytes / 2 names / 2 bytes

    def test_views_of_one_storage_do_not_overlap(self, server_factory):
        """One IPC export per storage, one as_strided view per name — the whole
        point of the storage/view split. Overlapping views would serve the same
        bytes for two different weights."""
        server = server_factory()
        server.begin_sync(1)
        _publish(server, GI_A, GROUP_A, nbytes=64)
        a, b = (server._cache[n] for n in GROUP_A)
        assert a.storage_offset() != b.storage_offset()
        assert a.data_ptr() != b.data_ptr()

    def test_publish_records_the_groups_names_for_release(self, server_factory):
        server = server_factory()
        server.begin_sync(3)
        _publish(server, GI_A, GROUP_A)
        assert server._group_names[GI_A] == list(GROUP_A)

    def test_publish_takes_a_backpressure_slot(self, server_factory):
        server = server_factory()
        server.begin_sync(1)
        _publish(server, GI_A, GROUP_A)
        assert server._inflight_groups == [GI_A]

    def test_the_default_lookahead_is_two(self):
        """Measured at 235B: lookahead=1 costs ~2.5-3s of sync wall — every
        group boundary drains the consumers' pull pipeline while the fleet-wide
        barrier closes. 2 restores one group of cross-boundary slack."""
        assert DEFAULT_GATHER_LOOKAHEAD == 2


class TestFreeBarrier:
    def test_single_consumer_signal_releases_the_group(self, server_factory):
        server = server_factory()
        server.begin_sync(1)
        _publish(server, GI_A, GROUP_A)
        server.free_group(GI_A)
        assert server._inflight_groups == []
        assert set(server._cache) == set()

    def test_group_is_held_until_the_last_live_consumer_signals(self, server_factory):
        """Every live consumer signals every owner; releasing on the first
        signal would drop storage other consumers are still reading."""
        server = server_factory()
        server.begin_sync(3)
        _publish(server, GI_A, GROUP_A)
        held_after_each_signal = []
        for _ in range(3):
            server.free_group(GI_A)
            held_after_each_signal.append(GI_A in server._inflight_groups)
        assert held_after_each_signal == [True, True, False]

    def test_intermediate_signals_do_not_release(self, server_factory):
        server = server_factory()
        server.begin_sync(3)
        _publish(server, GI_A, GROUP_A)
        server.free_group(GI_A)
        assert server._inflight_groups == [GI_A]
        assert set(server._cache) == set(GROUP_A)
        server.free_group(GI_A)
        assert server._inflight_groups == [GI_A]
        server.free_group(GI_A)
        assert server._inflight_groups == []

    def test_a_signal_that_beats_its_publish_is_completed_at_publish(
        self, server_factory
    ):
        """A consumer with nothing to pull from a group signals it at sync
        start, which can precede the publish. The publish must then release the
        group rather than wait for a signal that will never come again."""
        server = server_factory()
        server.begin_sync(1)
        server.free_group(GI_A)
        freed = _publish(server, GI_A, GROUP_A)
        assert server._inflight_groups == []
        assert freed == [GI_A]

    def test_early_signals_from_every_consumer_complete_at_publish(
        self, server_factory
    ):
        server = server_factory()
        server.begin_sync(3)
        for _ in range(3):
            server.free_group(GI_A)
        freed = _publish(server, GI_A, GROUP_A)
        assert freed == [GI_A]

    def test_freed_groups_are_reported_back_to_the_engine(self, server_factory):
        """The engine holds the CUDA-IPC export refs and may only drop them once
        the server says the group is gone."""
        server = server_factory()
        server.begin_sync(1)
        _publish(server, GI_A, GROUP_A)
        server.free_group(GI_A)
        assert _publish(server, GI_B, GROUP_B) == [GI_A]

    def test_a_signal_for_an_unpublished_group_releases_nothing(self, server_factory):
        server = server_factory()
        server.begin_sync(1)
        _publish(server, GI_A, GROUP_A)
        server.free_group(99)
        assert server._inflight_groups == [GI_A]
        assert set(server._cache) == set(GROUP_A)

    def test_release_drops_cache_and_count_entries(self, server_factory):
        server = server_factory()
        server.begin_sync(1)
        _publish(server, GI_A, GROUP_A)
        _publish(server, GI_B, GROUP_B)
        server.free_group(GI_A)
        assert set(server._cache) == set(GROUP_B)
        assert server._free_counts.get(GI_A) is None
        assert server._group_names.get(GI_A) is None


class TestBackpressure:
    def test_publish_blocks_once_the_lookahead_is_full(self, server_factory):
        """This is what bounds resident gathered groups on the trainer."""
        server = server_factory(gather_lookahead=1)
        server.begin_sync(1)
        _publish(server, GI_A, GROUP_A)

        done = threading.Event()

        def _second():
            _publish(server, GI_B, GROUP_B)
            done.set()

        t = threading.Thread(target=_second, daemon=True)
        t.start()
        assert not done.wait(timeout=0.3), "second publish should be blocked"

        server.free_group(GI_A)  # the consumer back-edge drains it
        assert done.wait(timeout=5), "free_group must release the publish"
        t.join(timeout=5)
        assert server._inflight_groups == [GI_B]

    def test_lookahead_one_serializes_the_pipeline_order(self, server_factory):
        """The lookahead=1 steady state: publish(N+1) is admitted only after
        group N's barrier completes, so the credit release ordering IS the
        publish ordering."""
        server = server_factory(gather_lookahead=1)
        server.begin_sync(1)
        order = []

        def _publisher():
            for gi, names in ((0, ("g0.w",)), (1, ("g1.w",)), (2, ("g2.w",))):
                _publish(server, gi, names)
                order.append(f"pub{gi}")

        pub = threading.Thread(target=_publisher, daemon=True)
        pub.start()
        for gi in range(3):
            while True:
                with server._cache_cond:
                    published = gi in server._group_names
                if published:
                    break
                time.sleep(0.01)
            order.append(f"free{gi}")
            server.free_group(gi)
        pub.join(timeout=5)
        assert not pub.is_alive()
        assert order == ["pub0", "free0", "pub1", "free1", "pub2", "free2"]

    def test_lookahead_two_admits_two_groups_without_blocking(self, server_factory):
        server = server_factory(gather_lookahead=2)
        server.begin_sync(1)
        _publish(server, GI_A, GROUP_A)
        _publish(server, GI_B, GROUP_B)
        assert server._inflight_groups == [GI_A, GI_B]

    def test_a_gather_error_releases_a_blocked_publish(self, server_factory):
        """Otherwise a trainer-side failure on another rank deadlocks this one."""
        server = server_factory(gather_lookahead=1)
        server.begin_sync(1)
        _publish(server, GI_A, GROUP_A)

        done = threading.Event()
        threading.Thread(
            target=lambda: (_publish(server, GI_B, GROUP_B), done.set()), daemon=True
        ).start()
        assert not done.wait(timeout=0.3)

        server.set_gather_error("rank 3 export failed")
        assert done.wait(timeout=5), "set_gather_error must unblock the publish"


class TestEndSync:
    def test_end_sync_returns_immediately_when_everything_is_freed(
        self, server_factory
    ):
        server = server_factory()
        server.begin_sync(1)
        _publish(server, GI_A, GROUP_A)
        server.free_group(GI_A)
        assert server.end_sync() == [GI_A]

    def test_end_sync_waits_for_an_outstanding_group(self, server_factory):
        server = server_factory()
        server.begin_sync(1)
        _publish(server, GI_A, GROUP_A)

        returned = []
        done = threading.Event()

        def _end():
            returned.append(server.end_sync())
            done.set()

        t = threading.Thread(target=_end, daemon=True)
        t.start()
        assert not done.wait(timeout=0.3), "end_sync must wait for the barrier"

        server.free_group(GI_A)
        assert done.wait(timeout=5)
        t.join(timeout=5)
        assert returned == [[GI_A]]

    def test_a_gather_error_releases_a_blocked_end_sync(self, server_factory):
        server = server_factory()
        server.begin_sync(1)
        _publish(server, GI_A, GROUP_A)

        done = threading.Event()
        threading.Thread(
            target=lambda: (server.end_sync(), done.set()), daemon=True
        ).start()
        assert not done.wait(timeout=0.3)
        server.set_gather_error("boom")
        assert done.wait(timeout=5)


class TestBeginSync:
    def test_begin_sync_sets_the_barrier_target(self, server_factory):
        server = server_factory()
        server.begin_sync(5)
        assert server._live_count == 5

    def test_the_target_is_floored_at_one(self, server_factory):
        """A bare/zero call must not make every group free instantly (or divide
        the barrier by zero)."""
        server = server_factory()
        server.begin_sync(0)
        assert server._live_count == 1

    def test_a_degraded_sync_lowers_the_target(self, server_factory):
        """FT degraded sync: the live count is the WHOLE degraded-sync
        mechanism on this side — the group releases after the live consumers
        alone."""
        server = server_factory()
        server.begin_sync(2)  # 4 provisioned, 2 alive
        _publish(server, GI_A, GROUP_A)
        server.free_group(GI_A)
        assert server._inflight_groups == [GI_A]
        server.free_group(GI_A)
        assert server._inflight_groups == []

    def test_begin_sync_clears_per_sync_state(self, server_factory):
        server = server_factory()
        server.begin_sync(2)
        _publish(server, GI_A, GROUP_A)
        server.free_group(GI_A)

        server.begin_sync(2)
        assert server._inflight_groups == []
        assert server._freed_pending == []
        assert server._free_counts == {}
        assert server._group_names == {}

    def test_a_straggler_signal_cannot_credit_the_next_sync(self, server_factory):
        """begin_sync resets the counts, so a signal that leaked past the sync
        boundary would over-credit — the reason the consumer drains its fired
        signals before finishing. Pin the reset side of that contract."""
        server = server_factory()
        server.begin_sync(2)
        _publish(server, GI_A, GROUP_A)
        server.free_group(GI_A)  # one of two consumers; sync then aborts/ends
        server.begin_sync(2)
        _publish(server, GI_A, GROUP_A)
        server.free_group(GI_A)
        assert server._inflight_groups == [GI_A], (
            "one signal after the reset must not release"
        )

    def test_begin_sync_clears_a_previous_gather_error(self, server_factory):
        server = server_factory()
        server.set_gather_error("last sync failed")
        server.begin_sync(1)
        assert server._gather_error is None

    def test_the_packed_destination_cache_survives_the_sync_boundary(
        self, server_factory
    ):
        """The packed layout repeats every sync — that is what makes caching the
        destination views worth ~5ms per 384-spec group."""
        server = server_factory()
        server._pack_dsts[("sentinel",)] = (0, [])
        server.begin_sync(1)
        assert ("sentinel",) in server._pack_dsts


class TestServedNamesGuard:
    """A producer serves only the names it holds: its stage's groups, and
    within them its own EP coordinate's experts plus the replicated names.
    Without this guard a misrouted pull would block forever in the cache wait
    for a name this rank never publishes."""

    def test_a_pull_for_an_unserved_name_fails_loudly(self, server_factory):
        server = server_factory(served_names=list(GROUP_A))
        server.begin_sync(1)
        with pytest.raises(RuntimeError, match="wrong producer"):
            server.rdt_produce_weights_batched([(GROUP_B[0], ())], consumer_id=0)

    def test_the_error_names_the_unserved_weights(self, server_factory):
        server = server_factory(served_names=list(GROUP_A))
        server.begin_sync(1)
        with pytest.raises(RuntimeError, match=GROUP_B[0]):
            server.rdt_produce_weights_batched([(GROUP_B[0], ())], consumer_id=0)

    def test_served_names_none_accepts_anything(self, server_factory):
        """Gather-to-all: every producer holds every group, so there is nothing
        to guard. The call proceeds to the cache wait (not exercised here)."""
        server = server_factory(served_names=None)
        assert server._served_names is None


class TestStallWatchdog:
    """The bound on the three waits that used to be unbounded.

    Engine-death detection is generation-driven, and no generation is in flight
    during a weight sync, so a consumer that dies INSIDE the sync window has no
    detector. It never sends its `free_group` signals, the group's barrier never
    completes, and `publish_group` / `end_sync` / the serve cache wait block
    forever — which stops this rank iterating its WeightSource, which is a
    collective, which wedges every other trainer rank in NCCL with no exception
    anywhere.

    The watchdog converts that into one real error. These use a tiny timeout;
    the production default is 300s.
    """

    def test_a_blocked_publish_fails_instead_of_hanging(self, server_factory):
        server = server_factory(gather_lookahead=1, stall_timeout_s=0.3)
        server.begin_sync(1)
        _publish(server, GI_A, GROUP_A)  # fills the only credit; nobody will signal

        done = threading.Event()
        threading.Thread(
            target=lambda: (_publish(server, GI_B, GROUP_B), done.set()), daemon=True
        ).start()

        assert done.wait(timeout=10), (
            "publish_group must give up rather than block forever"
        )
        assert isinstance(server._gather_error, RuntimeError)
        assert "RDT stall" in str(server._gather_error)

    def test_end_sync_fails_when_a_consumer_never_signals(self, server_factory):
        """The exact mid-sync death signature: the group is published, one of
        the two live consumers that owed a signal is gone."""
        server = server_factory(stall_timeout_s=0.3)
        server.begin_sync(2)
        _publish(server, GI_A, GROUP_A)
        server.free_group(GI_A)  # one of the two consumers signals; the other died

        server.end_sync()
        assert server._gather_error is not None
        assert "RDT stall" in str(server._gather_error)

    def test_the_error_reaches_every_blocked_waiter(self, server_factory):
        """The watchdog fires on the existing `set_gather_error` channel, so one
        stall unwinds the whole rank through one path rather than each waiter
        timing out separately."""
        server = server_factory(gather_lookahead=1, stall_timeout_s=0.3)
        server.begin_sync(1)
        _publish(server, GI_A, GROUP_A)

        errors: list = []

        def _pull():
            try:
                server.rdt_produce_weights_batched([(GROUP_B[0], ())], consumer_id=0)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        puller = threading.Thread(target=_pull, daemon=True)
        publisher = threading.Thread(
            target=lambda: _publish(server, GI_B, GROUP_B), daemon=True
        )
        puller.start()
        publisher.start()
        puller.join(timeout=10)
        publisher.join(timeout=10)
        assert not puller.is_alive() and not publisher.is_alive()
        assert errors and "gather errored" in str(errors[0])

    def test_progress_anywhere_keeps_a_slow_sync_alive(self, server_factory):
        """A consumer that is slow but signaling must never trip it: the stamp
        is global to the producer, so a steady trickle of signals holds the
        whole rank open. Nine sequential publishes with a 0.3s timeout each take
        longer in total than the timeout, and none may fire."""
        server = server_factory(gather_lookahead=1, stall_timeout_s=0.3)
        server.begin_sync(1)
        for gi in range(9):
            _publish(server, gi, (f"model.layers.{gi}.w",))
            time.sleep(0.1)
            server.free_group(gi)
        assert server._gather_error is None
        server.end_sync()
        assert server._gather_error is None

    def test_begin_sync_resets_the_progress_stamp(self, server_factory):
        """Syncs are minutes apart. Without the reset the first publish of sync N+1
        would measure its stall from somewhere inside sync N and fire immediately."""
        server = server_factory(gather_lookahead=1, stall_timeout_s=0.3)
        server.begin_sync(1)
        _publish(server, GI_A, GROUP_A)
        server.free_group(GI_A)
        server.end_sync()

        time.sleep(0.5)  # the idle gap between syncs, > the timeout
        server.begin_sync(1)
        _publish(server, GI_B, GROUP_B)
        assert server._gather_error is None

    def test_default_timeout_is_the_documented_one(self, server_factory):
        server = server_factory()
        assert server._stall_timeout == trainer_mod.DEFAULT_STALL_TIMEOUT_S


class TestConcurrentPublishAndFree:
    def test_the_lifecycle_survives_concurrent_publishes_and_signals(
        self, server_factory
    ):
        """Every group published must end up freed and out of the cache, with
        end_sync returning cleanly — the property whose violation is a hang."""
        server = server_factory(gather_lookahead=2)
        server.begin_sync(1)
        groups = [(gi, (f"model.layers.{gi}.w",)) for gi in range(12)]

        def _publisher():
            for gi, names in groups:
                _publish(server, gi, names)

        def _signaler():
            for gi, _names in groups:
                while True:
                    with server._cache_cond:
                        published = gi in server._group_names
                    if published:
                        break
                server.free_group(gi)

        pub = threading.Thread(target=_publisher, daemon=True)
        free = threading.Thread(target=_signaler, daemon=True)
        pub.start()
        free.start()
        pub.join(timeout=10)
        free.join(timeout=10)
        assert not pub.is_alive() and not free.is_alive()

        server.end_sync()
        assert server._inflight_groups == []
        assert server._cache == {}

    def test_concurrent_signals_of_one_group_release_it_exactly_once(
        self, server_factory
    ):
        """All live consumers signaling at once: the release must happen on the
        last signal and only once, or the engine drops IPC refs twice (or
        never)."""
        server = server_factory()
        server.begin_sync(8)
        _publish(server, GI_A, GROUP_A)

        barrier = threading.Barrier(8)

        def _signal():
            barrier.wait()
            server.free_group(GI_A)

        threads = [threading.Thread(target=_signal, daemon=True) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
            assert not t.is_alive()

        assert server._inflight_groups == []
        assert server._freed_pending == [GI_A]
        assert server._cache == {}


class TestFakeServerAgreesWithTheRealOne:
    """`test_weight_transfer.py`'s `_FakeProducerServer` is a second, independent
    model of this protocol that the engine-side tests assert on. Pin the two
    against each other so the fake cannot silently drift from the semantics the
    real server enforces."""

    @staticmethod
    def _fake():
        from tests.distributed.test_weight_transfer import _FakeProducerServer

        return _FakeProducerServer(auto_free=False)

    def test_the_barrier_and_early_signal_semantics_match(self, server_factory):
        real = server_factory()
        for server, publish in (
            (real, lambda gi, names: _publish(server, gi, names)),
            (self._fake(), lambda gi, names: server.publish_group(gi, _entries(names))),
        ):
            server.begin_sync(2)
            # early signal, then publish, then the closing signal
            server.free_group(GI_A)
            publish(GI_A, GROUP_A)
            assert server._inflight_groups == [GI_A]
            server.free_group(GI_A)
            assert server._inflight_groups == []
            assert server.end_sync() == [GI_A]
