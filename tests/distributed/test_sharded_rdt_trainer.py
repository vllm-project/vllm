# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the sharded-RDT trainer-side engine.

The trainer engine's own surface: init info and factory registration, the
send_weights round trip against an in-process fake producer server, group
ownership and the metadata it ships to the workers, `RdtRouter`'s M:N routing,
and the producer's packed destination-view cache.

The plan/replay half lives in test_sharded_rdt_plan.py and the producer
server's protocol in test_sharded_rdt_producer.py.
"""

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
from vllm.distributed.weight_transfer.sharded_rdt_common import (
    RdtRouter,
    assign_producer_indices,
)
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

    # kept under the old name for tests that inspect in-flight groups
    @property
    def inflight(self):
        return self._inflight_groups

    def begin_sync(self, live_count):
        self.order.append("begin")
        self.live_count = max(1, int(live_count))

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
    # What trainer_init resolves from the source's ownership + the fleet's
    # all-gather. ``fleet_owned`` stands in for that all-gather so a partial-
    # ownership rank can be tested without a real process group; the fleet must
    # cover every group or the router rejects it (nothing would serve the rest).
    if fleet_owned is None:
        engine._build_router(1, 0)
    else:
        # Stands in for the (metadata digest, owned groups, ep coord, stamp
        # digest) all-gather; every rank reports THIS rank's digests, i.e.
        # agreeing metadata and stamps.
        monkeypatch.setattr(
            engine,
            "_all_gather_owned",
            lambda w, mine: [(mine[0], o, mine[2], mine[3]) for o in fleet_owned],
        )
        engine._build_router(len(fleet_owned), 0)
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

        def finish_weight_update(self):
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


class TestRdtRouter:
    """Who serves each (gather group, ep_rank) pull unit.

    A wrong answer here is not a wrong number but a hang or a loud misroute: a
    consumer pulling from a producer that never gathered the name trips its
    served-names guard. Freeing does NOT route through the router — every
    consumer signals every owner of a group, and the producers count to the
    live total — so these tests pin PULL routing only."""

    def test_identity_when_fleets_match(self):
        r = RdtRouter(8, 8, None, num_groups=6)
        assert all(r.producer_for(3, g) == 3 for g in range(6))
        assert all(r.producer_for(c, 0) == c for c in range(8))

    def test_gather_to_all_keeps_the_historical_binding(self):
        """16 producers / 8 consumers: the block rule spreads each consumer's
        pulls over its block, alternating by group, and every producer NIC
        still carries traffic."""
        r = RdtRouter(16, 8, None, num_groups=95)
        for c in range(8):
            block = assign_producer_indices(16, 8, c)
            assert {r.producer_for(c, g) for g in range(95)} == set(block)
        assert [r.producer_for(0, g) for g in range(4)] == [0, 1, 0, 1]
        # No producer is left out of the pull traffic entirely.
        assert {r.producer_for(c, g) for c in range(8) for g in range(95)} == set(
            range(16)
        )

    def test_fan_in_shares_one_producer(self):
        r = RdtRouter(2, 8, None, num_groups=5)
        assert [{r.producer_for(c, g) for g in range(5)} for c in range(8)] == [
            {c // 4} for c in range(8)
        ]

    def test_pipeline_stages_route_to_the_owning_stage(self):
        """2 stages x 8 ranks: groups 0-2 on stage 0, 3-5 on stage 1. Each
        consumer must reach both stages, pulling each group from an owner."""
        owners = [list(range(8))] * 3 + [list(range(8, 16))] * 3
        r = RdtRouter(16, 8, owners)
        for c in range(8):
            assert [r.producer_for(c, g) for g in range(6)] == [c] * 3 + [c + 8] * 3
        assert r.owned_groups(0) == [0, 1, 2]
        assert r.owned_groups(8) == [3, 4, 5]
        r.validate()

    def test_expert_units_route_to_the_matching_coordinate(self):
        """The two stamp lists must match: a pull for a name stamped k goes to a
        group owner whose producer_ep_ranks entry is k; -1 keeps the full owner
        set (the historical routing)."""
        # 2 stages x (tp2 x ep2): coords repeat per stage.
        owners = [[0, 1, 2, 3]] * 2 + [[4, 5, 6, 7]] * 2
        coords = [0, 0, 1, 1, 0, 0, 1, 1]
        r = RdtRouter(8, 4, owners, producer_ep_ranks=coords)
        r.validate()
        assert r.owners(0, 1) == [2, 3]
        assert r.owners(2, 1) == [6, 7]
        assert r.owners(2) == [4, 5, 6, 7]  # -1: unchanged
        for c in range(4):
            assert r.producer_for(c, 0, 1) in (2, 3)
            assert r.producer_for(c, 2, 0) in (4, 5)

    def test_consumers_spread_over_a_coordinates_owner_set(self):
        """Several ranks share a coordinate (its TP peers); the block rule must
        spread consumers across them, not funnel through one."""
        r = RdtRouter(4, 4, None, num_groups=6, producer_ep_ranks=[0, 0, 1, 1])
        served = {r.producer_for(c, g, 0) for c in range(4) for g in range(6)}
        assert served == {0, 1}

    def test_an_empty_pull_unit_raises(self):
        """A stamped name whose coordinate has no rank in the group's owner set
        is a routing impossibility and must raise, not hang."""
        r = RdtRouter(2, 2, None, num_groups=2, producer_ep_ranks=[0, 0])
        with pytest.raises(ValueError, match="has no owner"):
            r.producer_for(0, 0, 1)

    def test_stamped_routing_without_coords_raises(self):
        """name stamps and producer stamps must ship together."""
        r = RdtRouter(2, 2, None, num_groups=2)
        with pytest.raises(ValueError, match="producer_ep_ranks"):
            r.owners(0, 1)

    def test_validate_rejects_an_unowned_group(self):
        with pytest.raises(ValueError, match="no owner"):
            RdtRouter(4, 2, [[0, 1], [], [2, 3]]).validate()

    def test_validate_rejects_an_out_of_range_owner(self):
        with pytest.raises(ValueError, match="out of range"):
            RdtRouter(2, 2, [[0, 5]]).validate()

    def test_validate_rejects_a_short_coordinate_list(self):
        with pytest.raises(ValueError, match="producer_ep_ranks"):
            RdtRouter(4, 2, None, num_groups=2, producer_ep_ranks=[0, 1]).validate()


class _OwnedSource(_ListSource):
    """A source that gathers only some groups, like a pipeline-parallel rank."""

    def __init__(self, pairs, owned_group_idx):
        super().__init__(pairs)
        self._owned = list(owned_group_idx)
        groups = layerwise_groups([n for n, _ in pairs])
        self._owned_names = [n for gi in self._owned for n in groups[gi]]

    def owned_groups(self):
        return list(self._owned)

    def __iter__(self):
        by_name = dict(self._pairs)
        return iter([(n, by_name[n]) for n in self._owned_names])


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the gather loop's CUDA-IPC export needs a device",
)
def test_sharded_rdt_publishes_only_owned_groups(monkeypatch):
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
    assert engine._group_owners == [[1], [0], [0], [1]]


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
    provisioned geometry is frozen; a degraded sync only lowers the one integer
    every owned group's barrier counts to."""

    @staticmethod
    def _engine(num_consumers, world=2, rank=0):
        engine = ShardedRDTTrainerWeightTransferEngine.__new__(
            ShardedRDTTrainerWeightTransferEngine
        )
        engine._init_info = ShardedRDTTrainerInitInfo(
            rank=rank, num_consumers=num_consumers
        )
        engine._router = RdtRouter(world, num_consumers, None, num_groups=4)
        engine.source = object()  # send_weights asserts a source is present
        received: list = []
        engine._send_weights_inner = received.append
        return engine, received

    def test_none_counts_the_whole_provisioned_fleet(self):
        engine, got = self._engine(num_consumers=8)
        engine.send_weights(None)
        assert got == [8]

    def test_a_live_set_counts_its_distinct_members(self):
        engine, got = self._engine(num_consumers=8)
        engine.send_weights([0, 1, 4, 5, 5])
        assert got == [4]

    def test_a_full_live_set_matches_the_provisioned_count(self):
        engine, got = self._engine(num_consumers=8)
        engine.send_weights(list(range(8)))
        engine.send_weights(None)
        assert got == [8, 8]

    def test_which_consumers_died_does_not_matter_only_how_many(self):
        """The whole point of the barrier: no routed per-producer targets, so
        the identity of the dead consumer is irrelevant to the producers."""
        counts = []
        for live in ([0, 1, 2, 3], [4, 5, 6, 7], [0, 2, 4, 6]):
            engine, got = self._engine(num_consumers=8)
            engine.send_weights(live)
            counts += got
        assert counts == [4, 4, 4]


def test_sharded_rdt_worker_init_info_carries_group_owners(monkeypatch):
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
    assert worker_init.group_owners == [[0], [0], [1], [1]]
    assert len(worker_init.group_owners) == len(worker_init.group_lens)
    # The payload crosses the control plane as JSON.
    assert json.loads(json.dumps(asdict(worker_init)))["group_owners"] == [
        [0],
        [0],
        [1],
        [1],
    ]


def test_weight_source_owns_every_group_by_default():
    """The contract's default: a source that says nothing owns the whole model."""
    src = _rdt_source_two_layers()
    assert src.owned_groups() is None
    assert ModuleSource(torch.nn.Linear(2, 2)).owned_groups() is None


def test_sharded_rdt_rejects_out_of_range_owned_group(monkeypatch):
    class _BadOwned(_OwnedSource):
        def owned_groups(self):
            return [0, 99]

    with pytest.raises(ValueError, match="out of range"):
        _rdt_engine_with_fake_server(
            _BadOwned(_rdt_source_two_layers()._pairs, [0]),
            is_sender=False,
            client=RecordingClient(),
            server=_FakeProducerServer(),
            monkeypatch=monkeypatch,
        )


def test_sharded_rdt_rejects_empty_owned_groups(monkeypatch):
    class _OwnsNothing(_ListSource):
        def owned_groups(self):
            return []

    with pytest.raises(ValueError, match="empty"):
        _rdt_engine_with_fake_server(
            _OwnsNothing(_rdt_source_two_layers()._pairs),
            is_sender=False,
            client=RecordingClient(),
            server=_FakeProducerServer(),
            monkeypatch=monkeypatch,
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
        lambda w, mine: [mine, ("deadbeefdeadbeef", [0, 3])],
    )
    with pytest.raises(ValueError, match="disagrees across trainer ranks"):
        engine._build_router(2, 0)


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
        arena_presize_gb=0.0,
        gather_lookahead=2,
    )
    srv._cache[src_name] = src
    srv._serve_rings[0] = [
        torch.empty(1 << 16, dtype=torch.uint8, device="cuda") for _ in range(2)
    ]

    def serve(chain):
        srv._serve_idx[0] = 0
        return srv.rdt_produce_weights_batched([(src_name, chain)], consumer_id=0)[0]

    return srv, serve


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the producer server needs a CUDA device"
)
def test_serve_does_not_reuse_packed_views_of_another_shape():
    """Two requests can share a name yet pack different slices of it.

    The producer caches the destination views it carves into a serve ring slot.
    Keyed by name alone, the second request is packed through the first's views.
    Reachable when one name's copies split across (group, ep_rank) chunks.
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
    carries the wrong bytes with no check downstream."""
    name = "model.layers.0.w"
    src = torch.arange(64, dtype=torch.bfloat16, device="cuda").reshape(8, 8)
    _srv, serve = _serve_ring_server(name, src)

    serve((("to", (torch.float16,), ()),))  # same shape, fp16
    blob = serve(())  # same name and shape, bf16

    got = blob[: src.numel() * src.element_size()].view(src.dtype).reshape(8, 8)
    assert torch.equal(got, src), "packed through a cached view of the wrong dtype"
