# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the sharded-RDT trainer-side engine.

The trainer engine's own surface: init info and factory registration, the
send_weights round trip against an in-process fake producer server, the
ownership table it resolves and ships to the workers, and the producer's packed
destination-view cache.

The plan/replay half (including `RdtRouter`, which is consumer-side) lives in
test_sharded_rdt_plan.py; the producer server's protocol in
test_sharded_rdt_producer.py.
"""

import itertools
from dataclasses import asdict

import pytest
import torch

from tests.distributed.test_weight_transfer import RecordingClient
from vllm.distributed.weight_transfer import (
    ModuleSource,
    ParamMeta,
    WeightSource,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.base import layerwise_groups
from vllm.distributed.weight_transfer.sharded_rdt_trainer import (
    ShardedRDTTrainerInitInfo,
    ShardedRDTTrainerWeightTransferEngine,
)


class _ListSource(WeightSource):
    """A WeightSource over an explicit ordered (name, cpu-tensor) list, so the
    sharded-RDT group/order logic can be tested without a real model."""

    def __init__(self, pairs):
        self._pairs = list(pairs)

    def metadata(self):
        return [ParamMeta(n, t.dtype, tuple(t.shape)) for n, t in self._pairs]

    def __iter__(self):
        return iter(self._pairs)


class _FakeProducerServer:
    """In-process stand-in for the _RDTProducerServer Ray actor. Records the
    engine->server call sequence and, by default, frees each group as soon as
    it is published (simulating the consumers' free_group barrier) so the
    gather loop's credit gate never parks. Mirrors the real server's
    per-group barrier: publish/free are keyed by GROUP INDEX, signals count
    to the ``begin_sync`` live total, publish_group returns nothing, and freed
    groups flow back only through wait_freed / end_sync (see
    test_sharded_rdt_producer.TestFakeServerAgreesWithTheRealOne)."""

    def __init__(self, auto_free=True):
        self.order: list[str] = []
        self.published: list[int] = []
        self.live_count = 1
        self._inflight_groups: list[int] = []
        self.auto_free = auto_free
        self.free_counts: dict[int, int] = {}
        self._pending_freed: list[int] = []

    # Alias for tests that inspect in-flight groups.
    @property
    def inflight(self):
        return self._inflight_groups

    def begin_sync(self, live_count, live_consumer_ids=None):
        self.order.append("begin")
        self.live_count = max(1, int(live_count))
        self.live_ids = live_consumer_ids

    def publish_group(self, group_idx, entries):
        self.order.append("publish")
        self.published.append(group_idx)
        self._inflight_groups.append(group_idx)
        if self.auto_free or self.free_counts.get(group_idx, 0) >= self.live_count:
            self._inflight_groups.remove(group_idx)
            self._pending_freed.append(group_idx)

    def free_group(self, group_idx):
        """Consumer back-edge; may arrive before the group's publish."""
        self.free_counts[group_idx] = self.free_counts.get(group_idx, 0) + 1
        if (
            group_idx in self._inflight_groups
            and self.free_counts[group_idx] >= self.live_count
        ):
            self._inflight_groups.remove(group_idx)
            self._pending_freed.append(group_idx)

    def wait_freed(self):
        """The engine's credit gate. The real server blocks here; the fake
        must already have a banked credit when the gate asks (auto_free, or a
        test's own free_group) — anything else is the deadlock the real
        watchdog would kill, so fail loudly. Not appended to ``order``: it is
        pacing, not a lifecycle milestone."""
        assert self._pending_freed, (
            "wait_freed with nothing freed: the gather loop would deadlock"
        )
        freed = self._pending_freed
        self._pending_freed = []
        return freed

    def end_sync(self):
        self.order.append("end")
        freed = self._pending_freed
        self._pending_freed = []
        return freed

    def set_gather_error(self, message):
        self.order.append("error")


def _rdt_engine_with_fake_server(
    source, *, is_sender, client, server, monkeypatch, fleet_owned=None
):
    """Build a ShardedRDTTrainerWeightTransferEngine wired to an in-process fake
    server (no Ray, no CUDA IPC): bypass trainer_init's spawn, set the
    group-major metadata, and route _rpc to the fake."""
    import vllm.distributed.weight_transfer.sharded_rdt_trainer as mod

    # reduce_tensor needs CUDA; the fake server never rebuilds, so stub it.
    monkeypatch.setattr(mod, "reduce_tensor", lambda t: (None, ("fake",)))

    init_info = ShardedRDTTrainerInitInfo(num_consumers=1, rank=0 if is_sender else 1)
    engine = ShardedRDTTrainerWeightTransferEngine(
        client=client, source=source, is_sender=is_sender, init_info=init_info
    )
    engine._meta = list(source.metadata())
    names = [m.name for m in engine._meta]
    engine._groups = layerwise_groups(names)
    engine._server = server
    engine._rpc = lambda method, *args: getattr(server, method)(*args)
    # What trainer_init resolves from the source's held names + the fleet's
    # all-gather. ``fleet_owned`` stands in for that all-gather so a partial-
    # ownership rank can be tested without a real process group; the fleet must
    # cover every name or the transpose rejects it (nothing would serve the rest).
    if fleet_owned is None:
        engine._resolve_ownership(1, 0)
    else:
        # Stands in for the (metadata digest, held-name bitmask) all-gather:
        # each rank's group list becomes the mask its held names would set.
        names = [m.name for m in engine._meta]
        groups = engine._groups

        def _fake_gather(_world, mine):
            out = []
            for owned in fleet_owned:
                held = {n for gi in owned for n in groups[gi]}
                mask = bytearray((len(names) + 7) // 8)
                for i, n in enumerate(names):
                    if n in held:
                        mask[i >> 3] |= 1 << (i & 7)
                out.append((mine[0], bytes(mask)))
            return out

        monkeypatch.setattr(engine, "_all_gather_owned", _fake_gather)
        engine._resolve_ownership(len(fleet_owned), 0)
    return engine


def _rdt_source_two_layers():
    return _ListSource(
        [
            ("embed.weight", torch.zeros(2)),
            ("model.layers.0.w", torch.zeros(2)),
            ("model.layers.1.w", torch.zeros(2)),
            ("norm.weight", torch.zeros(2)),
        ]
    )


class TestShardedRDTTrainerInitInfo:
    def test_declares_backend(self):
        assert ShardedRDTTrainerInitInfo.backend == "sharded_rdt"

    def test_rank_is_keyword_only_and_drives_is_sender(self):
        assert ShardedRDTTrainerInitInfo(num_consumers=4, rank=0).is_sender is True
        assert ShardedRDTTrainerInitInfo(num_consumers=4, rank=1).is_sender is False
        with pytest.raises(TypeError):
            # rank is keyword-only.
            ShardedRDTTrainerInitInfo(4, 0)  # type: ignore[misc]

    def test_registered_in_trainer_factory(self):
        cls = WeightTransferTrainerFactory._registry["sharded_rdt"]()
        assert cls is ShardedRDTTrainerWeightTransferEngine


def test_sharded_rdt_trainer_init_requires_source():
    with pytest.raises(ValueError, match="requires a WeightSource"):
        ShardedRDTTrainerWeightTransferEngine.trainer_init(
            ShardedRDTTrainerInitInfo(num_consumers=1, rank=0),
            client=RecordingClient(),
            source=None,
        )


def test_sharded_rdt_worker_init_info_is_group_major(monkeypatch):
    source = _rdt_source_two_layers()
    engine = _rdt_engine_with_fake_server(
        source,
        is_sender=True,
        client=RecordingClient(),
        server=_FakeProducerServer(),
        monkeypatch=monkeypatch,
    )
    worker_init = engine._build_worker_init_info(["srv_rk0"])
    # 4 params -> pre / layer0 / layer1 / post = 4 groups of length 1.
    assert worker_init.names == [
        "embed.weight",
        "model.layers.0.w",
        "model.layers.1.w",
        "norm.weight",
    ]
    assert worker_init.group_lens == [1, 1, 1, 1]
    assert worker_init.trainer_actor_names == ["srv_rk0"]
    assert worker_init.produce_method_name == "rdt_produce_weights_batched"
    assert sum(worker_init.group_lens) == len(worker_init.names)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the gather loop's CUDA-IPC export needs a device",
)
def test_sharded_rdt_send_weights_drives_client_in_order(monkeypatch):
    server = _FakeProducerServer(auto_free=True)
    client = RecordingClient()
    engine = _rdt_engine_with_fake_server(
        _rdt_source_two_layers(),
        is_sender=True,
        client=client,
        server=server,
        monkeypatch=monkeypatch,
    )
    engine.send_weights()

    assert client.order == ["start", "update", "finish"]
    # begin, one publish per group (4), end.
    assert server.order == ["begin", "publish", "publish", "publish", "publish", "end"]
    assert len(server.published) == 4
    # every group freed -> no engine-held refs remain.
    assert engine._inflight == {}


def test_sharded_rdt_send_weights_group_order_mismatch_raises(monkeypatch):
    # Source whose iteration order disagrees with its metadata order.
    class _BadSource(_ListSource):
        def __iter__(self):
            reordered = list(self._pairs)
            reordered[0], reordered[1] = reordered[1], reordered[0]
            return iter(reordered)

    server = _FakeProducerServer()
    engine = _rdt_engine_with_fake_server(
        _BadSource(_rdt_source_two_layers()._pairs),
        is_sender=True,
        client=RecordingClient(),
        server=server,
        monkeypatch=monkeypatch,
    )
    with pytest.raises(RuntimeError, match="iteration order must match"):
        engine.send_weights()
    assert "error" in server.order  # gather error propagated to the server


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the gather loop's CUDA-IPC export needs a device",
)
def test_sharded_rdt_non_sender_skips_client(monkeypatch):
    class _RaisingClient(RecordingClient):
        def start_weight_update(self):
            raise AssertionError("non-sender must not touch the client")

        def update_weights(self, update_info):
            raise AssertionError("non-sender must not touch the client")

        def finish_weight_update(self, weight_version: str | None = None):
            raise AssertionError("non-sender must not touch the client")

    server = _FakeProducerServer(auto_free=True)
    client = _RaisingClient()
    engine = _rdt_engine_with_fake_server(
        _rdt_source_two_layers(),
        is_sender=False,
        client=client,
        server=server,
        monkeypatch=monkeypatch,
    )
    engine.send_weights()  # gathers only; must not raise
    assert client.order == []
    assert server.order == ["begin", "publish", "publish", "publish", "publish", "end"]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the gather loop's CUDA-IPC export needs a device",
)
def test_sharded_rdt_send_weights_surfaces_update_error(monkeypatch):
    class _FailingUpdateClient(RecordingClient):
        def update_weights(self, update_info):
            self.order.append("update")
            raise RuntimeError("inference side rejected update")

    server = _FakeProducerServer(auto_free=True)
    engine = _rdt_engine_with_fake_server(
        _rdt_source_two_layers(),
        is_sender=True,
        client=_FailingUpdateClient(),
        server=server,
        monkeypatch=monkeypatch,
    )
    with pytest.raises(RuntimeError, match="inference side rejected update"):
        engine.send_weights()


class _OwnedSource(_ListSource):
    """A source holding only some groups' names, like a pipeline-parallel rank.
    Takes group indices for convenience and declares the names inside them."""

    def __init__(self, pairs, owned_group_idx):
        super().__init__(pairs)
        self._owned = list(owned_group_idx)
        groups = layerwise_groups([n for n, _ in pairs])
        self._owned_names = [n for gi in self._owned for n in groups[gi]]

    def held_names(self):
        return list(self._owned_names)

    def __iter__(self):
        by_name = dict(self._pairs)
        return iter([(n, by_name[n]) for n in self._owned_names])


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the gather loop's CUDA-IPC export needs a device",
)
def test_sharded_rdt_publishes_only_the_groups_it_holds(monkeypatch):
    server = _FakeProducerServer(auto_free=True)
    engine = _rdt_engine_with_fake_server(
        _OwnedSource(_rdt_source_two_layers()._pairs, [1, 2]),
        is_sender=False,
        client=RecordingClient(),
        server=server,
        monkeypatch=monkeypatch,
        fleet_owned=[[1, 2], [0, 3]],  # this rank holds the layers, rank 1 the rest
    )
    engine.send_weights()

    assert server.published == [1, 2]
    assert server.order == ["begin", "publish", "publish", "end"]
    assert engine._inflight == {}
    # embed + norm on rank 1, the two layers here: two distinct owner sets,
    # numbered by first appearance in metadata order.
    assert engine._owner_sets == [[1], [0]]
    assert engine._name_owner_class == [0, 1, 1, 0]


def test_sharded_rdt_owned_group_order_mismatch_raises(monkeypatch):
    class _MisorderedOwned(_OwnedSource):
        def __iter__(self):
            return iter(list(super().__iter__())[::-1])

    server = _FakeProducerServer(auto_free=True)
    engine = _rdt_engine_with_fake_server(
        _MisorderedOwned(_rdt_source_two_layers()._pairs, [1, 2]),
        is_sender=False,
        client=RecordingClient(),
        server=server,
        monkeypatch=monkeypatch,
        fleet_owned=[[1, 2], [0, 3]],
    )
    with pytest.raises(RuntimeError, match="iteration order must match"):
        engine.send_weights()
    assert "error" in server.order


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the gather loop's CUDA-IPC export needs a device",
)
def test_sharded_rdt_begin_sync_carries_the_live_count(monkeypatch):
    """The free barrier's target is one integer per sync: the live consumer
    count, defaulting to the whole provisioned fleet."""
    server = _FakeProducerServer(auto_free=True)
    engine = _rdt_engine_with_fake_server(
        _rdt_source_two_layers(),
        is_sender=False,
        client=RecordingClient(),
        server=server,
        monkeypatch=monkeypatch,
    )
    engine.send_weights()
    assert server.live_count == 1  # num_consumers=1 in the harness
    engine.send_weights(live_consumer_ids=[0])
    assert server.live_count == 1


class TestLiveCountPlumbing:
    """``send_weights(live_consumer_ids)`` -> the barrier target. The
    provisioned geometry is frozen; a degraded sync only lowers the target and
    narrows each slot-sharing group's rendezvous."""

    @staticmethod
    def _engine(num_consumers, world=2, rank=0):
        engine = ShardedRDTTrainerWeightTransferEngine.__new__(
            ShardedRDTTrainerWeightTransferEngine
        )
        engine._init_info = ShardedRDTTrainerInitInfo(
            rank=rank, num_consumers=num_consumers
        )
        engine.source = object()  # send_weights asserts a source is present
        received: list = []
        engine._send_weights_inner = lambda count, ids: received.append((count, ids))
        return engine, received

    def test_none_counts_the_whole_provisioned_fleet(self):
        engine, got = self._engine(num_consumers=8)
        engine.send_weights(None)
        assert got == [(8, list(range(8)))]

    def test_a_live_set_counts_its_distinct_members(self):
        engine, got = self._engine(num_consumers=8)
        engine.send_weights([0, 1, 4, 5, 5])
        assert got == [(4, [0, 1, 4, 5])]

    def test_a_full_live_set_matches_the_provisioned_count(self):
        engine, got = self._engine(num_consumers=8)
        engine.send_weights(list(range(8)))
        engine.send_weights(None)
        assert got == [(8, list(range(8)))] * 2

    def test_which_consumers_died_does_not_matter_only_how_many(self):
        """The whole point of the barrier: no routed per-producer targets, so
        the identity of the dead consumer is irrelevant to the producers."""
        counts = []
        for live in ([0, 1, 2, 3], [4, 5, 6, 7], [0, 2, 4, 6]):
            engine, got = self._engine(num_consumers=8)
            engine.send_weights(live)
            counts += [c for c, _ids in got]
        assert counts == [4, 4, 4]

    def test_the_live_ids_travel_with_the_count(self):
        """The count sizes the free barrier, the ids size the slot-sharing
        rendezvous, so a producer that shares slots can tell WHICH consumers it
        is still waiting for. They must describe the same set."""
        engine, got = self._engine(num_consumers=8)
        engine.send_weights([6, 0, 2, 2])
        ((count, ids),) = got
        assert ids == [0, 2, 6] and count == len(ids)


def test_sharded_rdt_worker_init_info_carries_the_ownership_table(monkeypatch):
    import json

    server = _FakeProducerServer(auto_free=True)
    engine = _rdt_engine_with_fake_server(
        _OwnedSource(_rdt_source_two_layers()._pairs, [0, 1]),
        is_sender=True,
        client=RecordingClient(),
        server=server,
        monkeypatch=monkeypatch,
        fleet_owned=[[0, 1], [2, 3]],
    )
    worker_init = engine._build_worker_init_info(["srv_rk0", "srv_rk1"])
    assert worker_init.owner_sets == [[0], [1]]
    assert worker_init.name_owner_class == [0, 0, 1, 1]
    assert len(worker_init.name_owner_class) == len(worker_init.names)
    # The payload crosses the control plane as JSON: nested lists must survive.
    round_tripped = json.loads(json.dumps(asdict(worker_init)))
    assert round_tripped["owner_sets"] == [[0], [1]]
    assert round_tripped["name_owner_class"] == [0, 0, 1, 1]


def test_weight_source_holds_everything_by_default():
    """The contract's default: a source that says nothing holds the whole model."""
    src = _rdt_source_two_layers()
    assert src.held_names() is None
    assert ModuleSource(torch.nn.Linear(2, 2)).held_names() is None


def test_sharded_rdt_rejects_a_held_name_outside_metadata(monkeypatch):
    class _BadHeld(_OwnedSource):
        def held_names(self):
            return ["embed.weight", "not.a.real.name"]

    with pytest.raises(ValueError, match="not"):
        _rdt_engine_with_fake_server(
            _BadHeld(_rdt_source_two_layers()._pairs, [0]),
            is_sender=False,
            client=RecordingClient(),
            server=_FakeProducerServer(),
            monkeypatch=monkeypatch,
        )


def test_sharded_rdt_rejects_a_rank_holding_nothing(monkeypatch):
    class _HoldsNothing(_ListSource):
        def held_names(self):
            return []

    with pytest.raises(ValueError, match="empty"):
        _rdt_engine_with_fake_server(
            _HoldsNothing(_rdt_source_two_layers()._pairs),
            is_sender=False,
            client=RecordingClient(),
            server=_FakeProducerServer(),
            monkeypatch=monkeypatch,
        )


def test_sharded_rdt_rejects_a_name_no_rank_holds(monkeypatch):
    """Every name must be held somewhere or it can never be served — caught
    when the holdings are transposed, naming the orphan."""
    with pytest.raises(ValueError, match="no trainer rank holds"):
        _rdt_engine_with_fake_server(
            _OwnedSource(_rdt_source_two_layers()._pairs, [0, 1]),
            is_sender=False,
            client=RecordingClient(),
            server=_FakeProducerServer(),
            monkeypatch=monkeypatch,
            fleet_owned=[[0, 1], [2]],  # nobody holds group 3
        )


def test_sharded_rdt_rejects_metadata_disagreement_across_ranks(monkeypatch):
    """Only the sender's metadata reaches the consumers, so a rank describing
    just its own share must fail loudly rather than silently drop the rest."""
    engine = _rdt_engine_with_fake_server(
        _OwnedSource(_rdt_source_two_layers()._pairs, [1, 2]),
        is_sender=False,
        client=RecordingClient(),
        server=_FakeProducerServer(auto_free=True),
        monkeypatch=monkeypatch,
        fleet_owned=[[1, 2], [0, 3]],  # a covering fleet, so construction succeeds
    )
    # Now rank 1 reports a DIFFERENT metadata digest for the same model.
    monkeypatch.setattr(
        engine,
        "_all_gather_owned",
        lambda w, mine: [mine, ("deadbeefdeadbeef", mine[1])],
    )
    with pytest.raises(ValueError, match="disagrees across trainer ranks"):
        engine._resolve_ownership(2, 0)


def _serve_ring_server(src_name, src):
    """A producer server with one cached tensor and a pre-seeded serve ring, so a
    pull needs no Ray and no NIXL registration. Returns (server, serve) where
    ``serve(chain)`` packs one spec into the SAME ring slot every time — which is
    what puts the destination-view cache, and only it, under test."""
    from vllm.distributed.weight_transfer.sharded_rdt_trainer import (
        _RDTProducerServer,
    )

    srv = _RDTProducerServer(
        num_rdt_buffers=2,
        buffer_presize_gb=0.0,
        gather_lookahead=2,
    )
    srv._cache[src_name] = src
    srv._serve_rings[0] = [
        torch.empty(1 << 16, dtype=torch.uint8, device="cuda") for _ in range(2)
    ]

    # The slot is ``seq % nring``, so stepping seq by nring keeps every pull in
    # slot 0 while still making each one its own generation -- reusing a seq
    # would rendezvous with the finished pack and return it unchanged.
    seqs = itertools.count(0, srv._nring)

    def serve(chain):
        return srv.rdt_produce_weights_batched(
            [(src_name, chain)], consumer_id=0, seq=next(seqs)
        )[0]

    return srv, serve


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the producer server needs a CUDA device"
)
def test_serve_does_not_reuse_packed_views_of_another_shape():
    """Two requests can share a name yet pack different slices of it.

    The producer caches the destination views it carves into a serve ring slot.
    Keyed by name alone, the second request is packed through the first's views.
    Reachable when one name's copies split across owner-class chunks.
    """
    name = "model.layers.0.w"
    src = torch.arange(64, dtype=torch.bfloat16, device="cuda").reshape(8, 8)
    _srv, serve = _serve_ring_server(name, src)

    serve((("narrow", (0, 0, 2), ()),))  # 2 rows
    wide = src.narrow(0, 0, 6)  # 6 rows, same name, same slot
    blob = serve((("narrow", (0, 0, 6), ()),))

    got = blob[: wide.numel() * wide.element_size()].view(wide.dtype).reshape(6, 8)
    assert torch.equal(got, wide)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the producer server needs a CUDA device"
)
def test_serve_does_not_reuse_packed_views_of_another_dtype():
    """The SILENT case of the same cache: two requests whose slices have the same
    name and the same shape but different dtypes pack at identical offsets, so
    reusing the stale views raises nothing — ``copy_`` just casts, and the blob
    carries the wrong bytes with no check downstream.

    ``view(dtype)`` rather than ``to(dtype)`` because the chain is replayed under
    ``ALLOWED_OPS``, which rejects ``to``: same-itemsize dtypes keep the byte
    length identical, which is exactly what makes the stale view fit.
    """
    name = "model.layers.0.w"
    src = torch.arange(64, dtype=torch.bfloat16, device="cuda").reshape(8, 8)
    _srv, serve = _serve_ring_server(name, src)

    serve((("view", (torch.float16,), ()),))  # same shape and bytes, fp16
    blob = serve(())  # same name and shape, bf16

    got = blob[: src.numel() * src.element_size()].view(src.dtype).reshape(8, 8)
    assert torch.equal(got, src), "packed through a cached view of the wrong dtype"
