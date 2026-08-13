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

import contextlib
import gc
import threading
import time

import pytest
import torch

import vllm.distributed.weight_transfer.sharded_rdt_trainer as trainer_mod
from vllm.distributed.weight_transfer.sharded_rdt_trainer import (
    DEFAULT_GATHER_LOOKAHEAD,
    ShardedRDTTrainerInitInfo,
    ShardedRDTTrainerWeightTransferEngine,
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

    def test_publish_records_the_group_as_unfreed(self, server_factory):
        server = server_factory()
        server.begin_sync(1)
        _publish(server, GI_A, GROUP_A)
        assert server._inflight_groups == [GI_A]

    def test_the_default_lookahead_is_one(self):
        """Under gather crediting, 1 is the sweet spot: group N+1 is gathered
        AND published while N is being pulled (the overlap that lookahead=2 had
        to buy under publish parking), with resident memory at its floor of 2
        groups — the bound the larger-model runs size against."""
        assert DEFAULT_GATHER_LOOKAHEAD == 1


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
        _publish(server, GI_A, GROUP_A)
        assert server._inflight_groups == []
        assert server._freed_pending == [GI_A]

    def test_early_signals_from_every_consumer_complete_at_publish(
        self, server_factory
    ):
        server = server_factory()
        server.begin_sync(3)
        for _ in range(3):
            server.free_group(GI_A)
        _publish(server, GI_A, GROUP_A)
        assert server._freed_pending == [GI_A]

    def test_freed_groups_are_reported_back_to_the_engine(self, server_factory):
        """The engine holds the CUDA-IPC export refs and may only drop them once
        the server says the group is gone — and it hears that ONLY through
        ``wait_freed`` / ``end_sync``, never a publish return (a freed notice
        riding an unharvested async publish while the engine blocks in
        ``wait_freed`` would wedge the loop)."""
        server = server_factory()
        server.begin_sync(1)
        _publish(server, GI_A, GROUP_A)
        server.free_group(GI_A)
        assert server.wait_freed() == [GI_A]
        assert server._freed_pending == []

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


class _CountingSource:
    """`iter_groups` yielding one CPU-tensor group per call, built lazily so a
    group's tensors exist only once the engine's credit gate lets it gather."""

    def __init__(self, groups):
        self._groups = groups

    def iter_groups(self):
        for names in self._groups:
            yield list(names), [torch.zeros(16, dtype=torch.float32) for _ in names]


class _ResidencyDict(dict):
    """The engine's `_inflight`, instrumented: `max_len` is the high-water mark
    of groups whose CUDA-IPC export refs were alive at once — the trainer's
    resident-memory bound in groups."""

    def __init__(self):
        super().__init__()
        self.max_len = 0

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.max_len = max(self.max_len, len(self))


def _loop_engine(server, n_groups, *, lookahead):
    """A real engine wired straight to a real (non-Ray) server, with only the
    state `_run_gather_loop` reads. The `_rpc` seam dispatches inline;
    `_publish_async` sees no `.remote` and runs inline too."""
    groups = [(f"model.layers.{gi}.w",) for gi in range(n_groups)]
    e = ShardedRDTTrainerWeightTransferEngine.__new__(
        ShardedRDTTrainerWeightTransferEngine
    )
    e._init_info = ShardedRDTTrainerInitInfo(
        rank=0, num_consumers=1, gather_lookahead=lookahead
    )
    e.source = _CountingSource(groups)
    e._server = server
    e._rpc = lambda method, *a: getattr(server, method)(*a)
    e._groups = [list(g) for g in groups]
    e._owned_idx = list(range(n_groups))
    e._held_names = None
    e._inflight = _ResidencyDict()
    return e


class TestGatherCredit:
    """The memory bound moved from parking publishes to gating gathers: a
    gathered group is published (serveable) immediately, and the ENGINE's loop
    stops gathering while more than `gather_lookahead` groups are unfreed. So
    at most `lookahead + 1` groups are resident — at the default of 1, AT MOST
    TWO — while group N+1 is already pulled the instant N's pulls finish."""

    @pytest.fixture
    def gather_engine(self, monkeypatch):
        """Neutralize the two CUDA touches in the export path so the REAL
        gather loop runs on CPU: `.cuda()` becomes identity, and the storage
        export ships the `(nbytes, ..., device, ...)` tuple the fixture's
        stubbed `rebuild_cuda_tensor` expects."""
        monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
        monkeypatch.setattr(
            trainer_mod,
            "reduce_tensor",
            lambda base: (None, (base.numel(), None, None, None, None, None, -1, None)),
        )
        return _loop_engine

    def test_publish_never_blocks(self, server_factory):
        """Publishing must not park on a credit: the whole point is that a
        gathered group becomes serveable immediately. Three publishes with no
        frees, inline on this thread — a block here hangs the test."""
        server = server_factory(gather_lookahead=1)
        server.begin_sync(1)
        for gi in range(3):
            _publish(server, gi, (f"model.layers.{gi}.w",))
        assert server._inflight_groups == [0, 1, 2]

    def test_lookahead_one_holds_at_most_two_groups_resident(
        self, server_factory, gather_engine
    ):
        """THE memory invariant: at lookahead=1 the trainer never holds more
        than 2 groups of gathered weights, no matter how the consumer paces —
        and the pipeline still overlaps, because the consumer here refuses to
        free group N until N+1 is already published (pullable). A gate an
        off-by-one too tight deadlocks this test; too loose fails max_len."""
        server = server_factory(gather_lookahead=1)
        n_groups = 8
        engine = gather_engine(server, n_groups, lookahead=1)

        failures: list[str] = []

        def _consumer():
            for gi in range(n_groups):
                # N+1 must be pullable before N frees
                target = min(gi + 1, n_groups - 1)
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    with server._cache_cond:
                        if target in server._group_names:
                            break
                    time.sleep(0.002)
                else:
                    failures.append(
                        f"group {target} never published while {gi} was unfreed"
                    )
                    return
                server.free_group(gi)

        consumer = threading.Thread(target=_consumer, daemon=True)
        consumer.start()
        engine._run_gather_loop(update_future=None, live_count=1)
        consumer.join(timeout=10)

        assert not consumer.is_alive() and not failures, failures
        assert engine._inflight.max_len <= 2, (
            f"{engine._inflight.max_len} groups were resident at lookahead=1"
        )
        assert engine._inflight == {} and server._cache == {}, (
            "everything must be freed by end_sync"
        )

    def test_the_residency_bound_scales_with_the_lookahead(
        self, server_factory, gather_engine
    ):
        """lookahead + 1, not a hardcoded 2: at lookahead=2 a free-nothing
        consumer sees exactly 3 groups gathered before the loop parks."""
        server = server_factory(gather_lookahead=2)
        engine = gather_engine(server, 8, lookahead=2)

        def _run():
            # unwound via set_gather_error below
            with contextlib.suppress(Exception):
                engine._run_gather_loop(update_future=None, live_count=1)

        loop = threading.Thread(target=_run, daemon=True)
        loop.start()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and len(server._inflight_groups) < 3:
            time.sleep(0.005)
        time.sleep(0.1)  # would-be 4th gather gets every chance to (wrongly) run
        try:
            assert engine._inflight.max_len == 3
            assert server._inflight_groups == [0, 1, 2]
        finally:
            server.set_gather_error("test over")  # unblock the parked wait_freed
            loop.join(timeout=10)
        assert not loop.is_alive()

    def test_wait_freed_returns_the_freed_backlog(self, server_factory):
        server = server_factory()
        server.begin_sync(1)
        _publish(server, GI_A, GROUP_A)
        _publish(server, GI_B, GROUP_B)
        server.free_group(GI_A)
        assert server.wait_freed() == [GI_A]
        server.free_group(GI_B)
        assert server.wait_freed() == [GI_B]

    def test_a_gather_error_releases_a_blocked_wait_freed(self, server_factory):
        """Otherwise a trainer-side failure on another rank deadlocks this one
        inside its credit gate. It must RAISE, not return empty — an empty
        return would spin the engine straight back into the wait."""
        server = server_factory()
        server.begin_sync(1)
        _publish(server, GI_A, GROUP_A)

        errors: list = []
        done = threading.Event()

        def _wait():
            try:
                server.wait_freed()
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
            done.set()

        threading.Thread(target=_wait, daemon=True).start()
        assert not done.wait(timeout=0.3), (
            "wait_freed must block while nothing is freed"
        )

        server.set_gather_error("rank 3 export failed")
        assert done.wait(timeout=5), "set_gather_error must unblock wait_freed"
        assert errors and "rank 3 export failed" in str(errors[0])


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
    completes, and `wait_freed` / `end_sync` / the serve cache wait block
    forever — which stops this rank iterating its WeightSource, which is a
    collective, which wedges every other trainer rank in NCCL with no exception
    anywhere.

    The watchdog converts that into one real error. These use a tiny timeout;
    the production default is 300s.
    """

    def test_a_blocked_wait_freed_fails_instead_of_hanging(self, server_factory):
        server = server_factory(gather_lookahead=1, stall_timeout_s=0.3)
        server.begin_sync(1)
        # the engine's credit gate now waits; nobody will signal
        _publish(server, GI_A, GROUP_A)

        errors: list = []
        done = threading.Event()

        def _wait():
            try:
                server.wait_freed()
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
            done.set()

        threading.Thread(target=_wait, daemon=True).start()

        assert done.wait(timeout=10), (
            "wait_freed must give up rather than block forever"
        )
        assert isinstance(server._gather_error, RuntimeError)
        assert "RDT stall" in str(server._gather_error)
        assert errors, "the engine must get an exception, not an empty credit list"

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
        timing out separately — here both a consumer parked in the serve cache
        wait and the engine parked in its credit gate."""
        server = server_factory(gather_lookahead=1, stall_timeout_s=0.3)
        server.begin_sync(1)
        _publish(server, GI_A, GROUP_A)

        errors: list = []

        def _pull():
            try:
                server.rdt_produce_weights_batched([(GROUP_B[0], ())], consumer_id=0)
            except BaseException as e:  # noqa: BLE001
                errors.append(("pull", e))

        def _wait():
            try:
                server.wait_freed()
            except BaseException as e:  # noqa: BLE001
                errors.append(("credit", e))

        puller = threading.Thread(target=_pull, daemon=True)
        waiter = threading.Thread(target=_wait, daemon=True)
        puller.start()
        waiter.start()
        puller.join(timeout=10)
        waiter.join(timeout=10)
        assert not puller.is_alive() and not waiter.is_alive()
        assert {who for who, _ in errors} == {"pull", "credit"}
        assert all("gather errored" in str(e) for _, e in errors)

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
            assert publish(GI_A, GROUP_A) is None, "publish returns nothing"
            assert server._inflight_groups == [GI_A]
            server.free_group(GI_A)
            assert server._inflight_groups == []
            assert server.wait_freed() == [GI_A]
            assert server.end_sync() == []
