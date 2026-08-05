# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protocol tests for the sharded-RDT producer server (`_RDTProducerServer`).

The publish -> serve -> free_gather -> release lifecycle is the subsystem's
concurrency core: a group nobody frees stalls `end_sync` forever, and a group
freed too early drops CUDA-IPC storage while a consumer is still reading it.
Until now the only in-process coverage was `_FakeProducerServer` in
`test_weight_transfer.py` — a separate reimplementation of these semantics that
omits the backpressure wait entirely — so the real class was exercised only by
two GPU tests of the packed-destination cache.

These tests drive the real server. Everything here is CPU-clean: the IPC rebuild
is stubbed (`rebuild_cuda_tensor`) and the serve path is only entered as far as
the routing guard, which fires before any CUDA work.

`rdt_produce_weights_batched`'s pack/copy half needs a real GPU arena; it stays
covered by the GPU tests in `test_weight_transfer.py`.
"""

import gc
import threading

import pytest
import torch

import vllm.distributed.weight_transfer.sharded_rdt_trainer as trainer_mod
from vllm.distributed.weight_transfer.sharded_rdt_trainer import _RDTProducerServer

# A group's key IS its name tuple, so no name->key map can go stale.
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
        kwargs.setdefault("nosync", False)
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


def _publish(server, key, free_target=1, **kw):
    return server.publish_group(key, _entries(key, **kw), free_target)


class TestPublishAndRebuild:
    def test_publish_makes_every_name_serveable(self, server_factory):
        server = server_factory()
        server.begin_sync()
        _publish(server, GROUP_A)
        assert set(server._cache) == set(GROUP_A)

    def test_rebuilt_views_have_the_requested_dtype_and_geometry(self, server_factory):
        server = server_factory()
        server.begin_sync()
        _publish(server, GROUP_A, nbytes=64, dtype="bfloat16")
        for name in GROUP_A:
            t = server._cache[name]
            assert t.dtype is torch.bfloat16
            assert tuple(t.shape) == (16,)  # 64 bytes / 2 names / 2 bytes

    def test_views_of_one_storage_do_not_overlap(self, server_factory):
        """One IPC export per storage, one as_strided view per name — the whole
        point of the storage/view split. Overlapping views would serve the same
        bytes for two different weights."""
        server = server_factory()
        server.begin_sync()
        _publish(server, GROUP_A, nbytes=64)
        a, b = (server._cache[n] for n in GROUP_A)
        assert a.storage_offset() != b.storage_offset()
        assert a.data_ptr() != b.data_ptr()

    def test_publish_arms_the_groups_free_target(self, server_factory):
        server = server_factory()
        server.begin_sync()
        _publish(server, GROUP_A, free_target=3)
        assert server._free_targets[GROUP_A] == 3

    def test_free_target_is_floored_at_one(self, server_factory):
        """A published group must be freeable, so a 0 target cannot be honoured;
        the engine is expected to skip publishing such a group entirely."""
        server = server_factory()
        server.begin_sync()
        _publish(server, GROUP_A, free_target=0)
        assert server._free_targets[GROUP_A] == 1

    def test_publish_takes_a_backpressure_slot(self, server_factory):
        server = server_factory()
        server.begin_sync()
        _publish(server, GROUP_A)
        assert server._inflight_keys == [GROUP_A]


class TestFreeRefCounting:
    def test_single_consumer_free_releases_the_group(self, server_factory):
        server = server_factory()
        server.begin_sync()
        _publish(server, GROUP_A, free_target=1)
        server.free_gather(list(GROUP_A))
        assert server._inflight_keys == []
        assert set(server._cache) == set()

    def test_group_is_held_until_the_last_routed_consumer_frees(self, server_factory):
        """Under C>P fan-in several consumers pull one group from one producer;
        releasing on the first free would drop storage still being read."""
        server = server_factory()
        server.begin_sync()
        _publish(server, GROUP_A, free_target=3)
        held_after_each_free = []
        for _ in range(3):
            server.free_gather(list(GROUP_A))
            held_after_each_free.append(GROUP_A in server._inflight_keys)
        assert held_after_each_free == [True, True, False]

    def test_intermediate_frees_do_not_release(self, server_factory):
        server = server_factory()
        server.begin_sync()
        _publish(server, GROUP_A, free_target=3)
        server.free_gather(list(GROUP_A))
        assert server._inflight_keys == [GROUP_A]
        assert set(server._cache) == set(GROUP_A)
        server.free_gather(list(GROUP_A))
        assert server._inflight_keys == [GROUP_A]
        server.free_gather(list(GROUP_A))
        assert server._inflight_keys == []

    def test_a_free_that_beats_its_publish_is_completed_at_publish(
        self, server_factory
    ):
        """A consumer with nothing to pull for a group frees it as its plan
        starts, which can precede the publish. The publish must then release the
        group rather than wait for a free that will never come again."""
        server = server_factory()
        server.begin_sync()
        server.free_gather(list(GROUP_A))
        freed = _publish(server, GROUP_A, free_target=1)
        assert server._inflight_keys == []
        assert freed == [GROUP_A]

    def test_freed_keys_are_reported_back_to_the_engine(self, server_factory):
        """The engine holds the CUDA-IPC export refs and may only drop them once
        the server says the group is gone."""
        server = server_factory()
        server.begin_sync()
        _publish(server, GROUP_A)
        server.free_gather(list(GROUP_A))
        assert _publish(server, GROUP_B) == [GROUP_A]

    def test_a_free_whose_names_match_no_group_releases_nothing(self, server_factory):
        server = server_factory()
        server.begin_sync()
        _publish(server, GROUP_A)
        server.free_gather(["some.other.name"])
        assert server._inflight_keys == [GROUP_A]
        assert set(server._cache) == set(GROUP_A)

    def test_release_drops_cache_and_event_entries(self, server_factory):
        server = server_factory()
        server.begin_sync()
        _publish(server, GROUP_A)
        _publish(server, GROUP_B)
        server.free_gather(list(GROUP_A))
        assert set(server._cache) == set(GROUP_B)
        assert server._free_counts.get(GROUP_A) is None
        assert server._free_targets.get(GROUP_A) is None


class TestBackpressure:
    def test_publish_blocks_once_the_lookahead_is_full(self, server_factory):
        """This is what bounds resident gathered groups on the trainer. The fake
        server used by the engine tests does not implement it at all."""
        server = server_factory(gather_lookahead=1)
        server.begin_sync()
        _publish(server, GROUP_A)

        done = threading.Event()

        def _second():
            _publish(server, GROUP_B)
            done.set()

        t = threading.Thread(target=_second, daemon=True)
        t.start()
        assert not done.wait(timeout=0.3), "second publish should be blocked"

        server.free_gather(list(GROUP_A))  # the consumer back-edge drains it
        assert done.wait(timeout=5), "free_gather must release the publish"
        t.join(timeout=5)
        assert server._inflight_keys == [GROUP_B]

    def test_lookahead_two_admits_two_groups_without_blocking(self, server_factory):
        server = server_factory(gather_lookahead=2)
        server.begin_sync()
        _publish(server, GROUP_A)
        _publish(server, GROUP_B)
        assert server._inflight_keys == [GROUP_A, GROUP_B]

    def test_a_gather_error_releases_a_blocked_publish(self, server_factory):
        """Otherwise a trainer-side failure on another rank deadlocks this one."""
        server = server_factory(gather_lookahead=1)
        server.begin_sync()
        _publish(server, GROUP_A)

        done = threading.Event()
        threading.Thread(
            target=lambda: (_publish(server, GROUP_B), done.set()), daemon=True
        ).start()
        assert not done.wait(timeout=0.3)

        server.set_gather_error("rank 3 export failed")
        assert done.wait(timeout=5), "set_gather_error must unblock the publish"


class TestEndSync:
    def test_end_sync_returns_immediately_when_everything_is_freed(
        self, server_factory
    ):
        server = server_factory()
        server.begin_sync()
        _publish(server, GROUP_A)
        server.free_gather(list(GROUP_A))
        assert server.end_sync() == [GROUP_A]

    def test_end_sync_waits_for_an_outstanding_group(self, server_factory):
        server = server_factory()
        server.begin_sync()
        _publish(server, GROUP_A)

        returned = []
        done = threading.Event()

        def _end():
            returned.append(server.end_sync())
            done.set()

        t = threading.Thread(target=_end, daemon=True)
        t.start()
        assert not done.wait(timeout=0.3), "end_sync must wait for the free"

        server.free_gather(list(GROUP_A))
        assert done.wait(timeout=5)
        t.join(timeout=5)
        assert returned == [[GROUP_A]]

    def test_a_gather_error_releases_a_blocked_end_sync(self, server_factory):
        server = server_factory()
        server.begin_sync()
        _publish(server, GROUP_A)

        done = threading.Event()
        threading.Thread(
            target=lambda: (server.end_sync(), done.set()), daemon=True
        ).start()
        assert not done.wait(timeout=0.3)
        server.set_gather_error("boom")
        assert done.wait(timeout=5)


class TestBeginSync:
    def test_begin_sync_clears_per_sync_state(self, server_factory):
        server = server_factory()
        server.begin_sync()
        _publish(server, GROUP_A, free_target=2)
        server.free_gather(list(GROUP_A))

        server.begin_sync()
        assert server._inflight_keys == []
        assert server._freed_pending == []
        assert server._free_counts == {}
        assert server._free_targets == {}

    def test_begin_sync_clears_a_previous_gather_error(self, server_factory):
        server = server_factory()
        server.set_gather_error("last sync failed")
        server.begin_sync()
        assert server._gather_error is None

    def test_the_packed_destination_cache_survives_the_sync_boundary(
        self, server_factory
    ):
        """The packed layout repeats every sync — that is what makes caching the
        destination views worth ~5ms per 384-spec group."""
        server = server_factory()
        server._pack_dsts[("sentinel",)] = (0, [])
        server.begin_sync()
        assert ("sentinel",) in server._pack_dsts


class TestServedNamesGuard:
    """Under partial (pipeline-parallel) ownership a producer serves only its
    own stage's names. Without this guard a misrouted pull would block forever
    in the cache wait for a name this rank never gathers."""

    def test_a_pull_for_an_unserved_name_fails_loudly(self, server_factory):
        server = server_factory(served_names=list(GROUP_A))
        server.begin_sync()
        with pytest.raises(RuntimeError, match="wrong producer"):
            server.rdt_produce_weights_batched([(GROUP_B[0], ())], consumer_id=0)

    def test_the_error_names_the_unserved_weights(self, server_factory):
        server = server_factory(served_names=list(GROUP_A))
        server.begin_sync()
        with pytest.raises(RuntimeError, match=GROUP_B[0]):
            server.rdt_produce_weights_batched([(GROUP_B[0], ())], consumer_id=0)

    def test_served_names_none_accepts_anything(self, server_factory):
        """Gather-to-all: every producer holds every group, so there is nothing
        to guard. The call proceeds to the cache wait (not exercised here)."""
        server = server_factory(served_names=None)
        assert server._served_names is None


class TestFakeServerAgreesWithTheRealOne:
    """`test_weight_transfer.py`'s `_FakeProducerServer` is a second, independent
    model of this protocol, and 12 engine-side tests assert against it. Those
    tests drive `_run_gather_loop`, whose CUDA-IPC export makes them GPU-only, so
    the fake cannot simply be deleted here. Pin the agreement instead: if the two
    diverge on the observable protocol, this fails on CPU rather than the engine
    tests silently passing against wrong semantics.
    """

    def _drive(self, server, *, free_target=1):
        """One sync's observable outputs: what each publish reports freed, and
        what end_sync returns."""
        server.begin_sync()
        out = []
        out.append(("publish", _publish(server, GROUP_A, free_target=free_target)))
        server.free_gather(list(GROUP_A))
        out.append(("publish", _publish(server, GROUP_B, free_target=free_target)))
        server.free_gather(list(GROUP_B))
        out.append(("end", server.end_sync()))
        return out

    def _fake(self):
        from tests.distributed.test_weight_transfer import _FakeProducerServer

        fake = _FakeProducerServer(auto_free=False)
        # The fake takes no (storages, views); adapt it to the real signature.
        real_publish = fake.publish_group
        fake.publish_group = lambda key, entries, target: real_publish(
            key, entries, target
        )
        return fake

    def test_freed_key_reporting_matches(self, server_factory):
        assert self._drive(self._fake()) == self._drive(server_factory())

    def test_free_before_publish_completion_matches(self, server_factory):
        def _drive_early_free(server):
            server.begin_sync()
            server.free_gather(list(GROUP_A))
            return server.publish_group(GROUP_A, _entries(GROUP_A), 1)

        assert _drive_early_free(self._fake()) == _drive_early_free(server_factory())

    def test_the_fake_does_not_model_backpressure(self, server_factory):
        """Documented divergence, kept explicit: the fake never blocks, so the
        `gather_lookahead` bound is unexercised by the engine tests. It is
        covered against the real server in TestBackpressure above."""
        fake = self._fake()
        fake.begin_sync()
        for i in range(5):
            fake.publish_group((f"g{i}",), _entries((f"g{i}",)), 1)
        assert len(fake.inflight) == 5, "the fake admits unbounded groups"


class TestConcurrentPublishAndFree:
    def test_the_lifecycle_survives_concurrent_publishes_and_frees(
        self, server_factory
    ):
        """Every group published must end up freed and out of the cache, with
        end_sync returning cleanly — the property whose violation is a hang."""
        server = server_factory(gather_lookahead=2)
        server.begin_sync()
        keys = [(f"model.layers.{i}.w",) for i in range(12)]

        def _publisher():
            for key in keys:
                _publish(server, key, free_target=1)

        def _freer():
            for key in keys:
                while True:
                    with server._cache_cond:
                        armed = key in server._free_targets
                    if armed:
                        break
                server.free_gather(list(key))

        pub = threading.Thread(target=_publisher, daemon=True)
        free = threading.Thread(target=_freer, daemon=True)
        pub.start()
        free.start()
        pub.join(timeout=10)
        free.join(timeout=10)
        assert not pub.is_alive() and not free.is_alive()

        server.end_sync()
        assert server._inflight_keys == []
        assert server._cache == {}

    def test_concurrent_frees_of_one_group_release_it_exactly_once(
        self, server_factory
    ):
        """The C>P fan-in case with all consumers freeing at once: the release
        must happen on the last free and only once, or the engine drops IPC refs
        twice (or never)."""
        server = server_factory()
        server.begin_sync()
        _publish(server, GROUP_A, free_target=8)

        barrier = threading.Barrier(8)

        def _free():
            barrier.wait()
            server.free_gather(list(GROUP_A))

        threads = [threading.Thread(target=_free, daemon=True) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
            assert not t.is_alive()

        assert server._inflight_keys == []
        assert server._freed_pending == [GROUP_A]
        assert server._cache == {}
