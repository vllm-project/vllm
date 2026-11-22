# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sharded-RDT trainer engine on CPU, against a real producer server
(CUDA IPC replaced by an in-process registry)."""

import json
import threading
import time
from functools import cached_property
from types import SimpleNamespace

import pytest
import torch

import vllm.distributed.weight_transfer.sharded_rdt_trainer as trainer_mod
from tests.distributed.sharded_rdt_utils import make_server, patch_for_cpu
from tests.distributed.test_weight_transfer import RecordingClient
from vllm.distributed.weight_transfer import ParamMeta, WeightSource
from vllm.distributed.weight_transfer.sharded_rdt_trainer import (
    ShardedRDTTrainerInitInfo,
    ShardedRDTTrainerWeightTransferEngine,
)


class _Source(WeightSource):
    """Ordered (name, tensor) pairs; yields what `rank` owns."""

    def __init__(self, pairs, owners=None, rank=0):
        self._pairs, self._owners, self._rank = list(pairs), owners or {}, rank
        self.yielded: list[str] = []

    @cached_property
    def metadata(self):
        return [
            ParamMeta(n, t.dtype, tuple(t.shape), owners=self._owners.get(n))
            for n, t in self._pairs
        ]

    def iter_full(self):
        for n, t in self._pairs:
            if n not in self._owners or self._rank in self._owners[n]:
                self.yielded.append(n)
                yield n, t


def _pairs(n=6, numel=8):
    return [(f"model.layers.{i}.w", torch.full((numel,), float(i))) for i in range(n)]


# embed and layer 0 on stage 0, layer 1 on stage 1, norm everywhere
_PP = [
    ("embed.weight", torch.zeros(2)),
    ("model.layers.0.w", torch.zeros(2)),
    ("model.layers.1.w", torch.zeros(2)),
    ("norm.weight", torch.zeros(2)),
]
_PP_OWNERS = {"embed.weight": (0,), "model.layers.0.w": (0,), "model.layers.1.w": (1,)}


@pytest.fixture
def cpu_ipc(monkeypatch):
    """CUDA IPC in process: an export is a key into a registry of storages."""
    registry: dict[int, torch.Tensor] = {}

    def reduce(base):
        registry[base.untyped_storage().data_ptr()] = base
        ptr = base.untyped_storage().data_ptr()
        # rebuild_cuda_tensor's 15 arguments: [7:10] locate the allocation
        return None, (ptr, *(0,) * 6, ptr, base.nbytes, 0, *(None,) * 4, True)

    monkeypatch.setattr(trainer_mod, "reduce_tensor", reduce)
    monkeypatch.setattr(trainer_mod, "rebuild_cuda_tensor", lambda *a: registry[a[0]])
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    # server calls run inline: a call's "ref" is its result
    monkeypatch.setattr(trainer_mod.ray, "get", lambda result: result)
    patch_for_cpu(monkeypatch)


def _engine(source, *, rank=0, world=1, client=None, **init):
    init.setdefault("num_consumers", 1)
    engine = ShardedRDTTrainerWeightTransferEngine(
        client=client or RecordingClient(),
        source=source,
        is_sender=rank == 0,
        init_info=ShardedRDTTrainerInitInfo(rank=rank, **init),
    )
    engine._plan(world, rank)
    return engine


def _attach_server(engine):
    """A real producer for `engine`, as `_spawn_server` builds it."""
    served = {
        m.name: (str(m.dtype).removeprefix("torch."), m.shape)
        for m in engine._meta
        if m.name in engine._owned_names
    }
    units = {m.name: u for m, u in zip(engine._meta, engine._units)}
    server = make_server(
        served=served,
        units=units,
        arena_bytes=1 << 20,
        bucket_bytes=1 << 20,
        stall_timeout_s=20,
    )
    engine._server = server
    engine._remote = lambda method, *args: getattr(server, method)(*args)
    return server


@pytest.mark.parametrize("field", ["buffer_bytes", "num_buffers", "num_consumers"])
def test_init_info_validates(field):
    with pytest.raises(ValueError, match=field):
        ShardedRDTTrainerInitInfo(rank=0, **{"num_consumers": 1, field: 0})


def test_trainer_init_needs_a_full_source():
    info = ShardedRDTTrainerInitInfo(rank=0, num_consumers=1)
    with pytest.raises(ValueError, match="requires a WeightSource"):
        ShardedRDTTrainerWeightTransferEngine.trainer_init(
            info, client=RecordingClient()
        )

    class _LocalOnly(WeightSource):
        @cached_property
        def metadata(self):
            return []

    with pytest.raises(TypeError, match="iter_full"):
        ShardedRDTTrainerWeightTransferEngine.trainer_init(
            info, client=RecordingClient(), source=_LocalOnly()
        )


@pytest.mark.parametrize(
    "rank,owned,units,gathered",
    [
        (0, {"embed.weight", "model.layers.0.w", "norm.weight"}, [0, 1], None),
        (
            1,
            {"model.layers.1.w", "norm.weight"},
            [1],
            ["model.layers.1.w", "norm.weight"],
        ),
    ],
)
def test_a_rank_plans_and_gathers_what_it_owns(cpu_ipc, rank, owned, units, gathered):
    """Ownership comes from metadata, alike on every rank; units are global
    (8 B names, 16 B units) and a rank gathers those it owns part of."""
    source = _Source(_PP, _PP_OWNERS, rank)
    engine = _engine(source, rank=rank, world=2, buffer_bytes=16)
    assert (engine._owner_sets, engine._name_owner_class) == (
        [[0], [1], [0, 1]],
        [0, 0, 1, 2],
    )
    assert engine._owned_names == owned and engine._units == [0, 0, 1, 1]
    assert list(engine._owned_units) == units
    if gathered:
        _attach_server(engine).register_consumer(0, [])
        engine.send_weights()
        assert source.yielded == gathered


def test_owners_outside_the_trainer_are_refused():
    with pytest.raises(ValueError, match="only 2 trainer ranks"):
        _engine(_Source(_PP, {"embed.weight": (0, 2)}), world=2)


@pytest.mark.parametrize("replicas", [1, 3])
def test_arena_covers_the_consumers_that_may_pull_from_the_rank(replicas):
    """Layer 1 only on rank 1: both workers of a deployment may pull from it,
    one from 0. Other deployments share their blocks: the arena is the same."""
    sizes = [
        _engine(
            _Source(_PP, {"model.layers.1.w": (1,)}, rank),
            rank=rank,
            world=2,
            num_consumers=2 * replicas,
            workers_per_replica=2 if replicas > 1 else 0,
            buffer_bytes=100,
        )._arena_bytes()
        for rank in (0, 1)
    ]
    assert sizes == [(2 * 1 + 1) * 116, (2 * 2 + 1) * 116]


def test_worker_payload_survives_json():
    engine = _engine(_Source(_PP, _PP_OWNERS, 0), world=2, buffer_bytes=64)
    engine._server_names = ["srv0", "srv1"]
    payload = json.loads(json.dumps(engine.get_worker_init_payload()))
    assert payload["trainer_actor_names"] == ["srv0", "srv1"]
    assert payload["name_owner_class"] == [0, 0, 1, 2]
    assert payload["dtype_names"] == ["float32"] * 4
    assert payload["buffer_bytes"] == 64 and payload["workers_per_replica"] == 0


def test_metadata_disagreement_across_ranks_fails_at_init(monkeypatch):
    engine = _engine(_Source(_PP, _PP_OWNERS, 0), world=2)
    engine._server_name = "srv0"

    def fake_all_gather(out, mine):
        out[:] = [mine, ("srv1", "deadbeefdeadbeef")]

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather)
    with pytest.raises(ValueError, match="disagree across trainer ranks"):
        engine._all_gather_server_names(2, 0)


# ------------------------------------------------------------ gather loop


def test_the_sender_drives_the_workers_around_its_gathers(cpu_ipc, monkeypatch):
    monkeypatch.setattr(trainer_mod, "_EXPORT_RING_MAX_BYTES", 0)  # no export slot
    client = RecordingClient()
    engine = _engine(_Source(_pairs(2)), client=client, num_consumers=2)
    server = _attach_server(engine)
    for c in (0, 1):
        server.register_consumer(c, [])  # pulls nothing: units release at publish
    keeps, remote = [], engine._remote

    def recording_remote(method, *args):
        if method == "publish_unit":
            keeps.append(bool(args[1][3]))
        return remote(method, *args)

    engine._remote = recording_remote
    engine.send_weights()
    engine.send_weights(live_consumer_ids=[1])  # degraded: consumer 0 died
    # storages that outlived the last sync stay imported on the producer
    assert keeps == [False, True]
    assert client.order == ["start", "update", "finish"] * 2
    assert client.last_update_info == {"sync_idx": 1} and server._live == {1}
    assert not engine._inflight


@pytest.mark.parametrize("failing", ["start_weight_update", "update_weights"])
def test_a_failed_worker_call_fails_the_sync_on_every_producer_at_once(
    cpu_ipc, monkeypatch, failing
):
    """The sync fails at once with the worker's error, not by the watchdog."""

    def fail():
        raise RuntimeError("Failed to register cuda memory with NIXL")

    class _Failing(RecordingClient):
        def start_weight_update(self):
            return fail() if failing == "start_weight_update" else None

        def update_weights(self, update_info):
            return fail() if failing == "update_weights" else None

    engine = _engine(_Source(_pairs(2)), client=_Failing(), buffer_bytes=32)
    server = _attach_server(engine)
    # consumer 0 needs both units but never pulls: only the error ends the sync
    server.register_consumer(
        0, [[("model.layers.0.w", ((0, 8),))], [("model.layers.1.w", ((0, 8),))]]
    )
    engine._server_names = ["rk0"]
    actor = SimpleNamespace(
        set_gather_error=SimpleNamespace(remote=server.set_gather_error)
    )
    monkeypatch.setattr(trainer_mod.ray, "get_actor", lambda *a, **k: actor)
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="register cuda memory"):
        engine.send_weights()
    assert time.monotonic() - t0 < 10  # the stall watchdog is 20 s
    server.begin_sync([0], 0)  # a rank that begins the failed sync late
    with pytest.raises(RuntimeError, match="register cuda memory"):
        server.wait_freed()


@pytest.mark.parametrize("lookahead", [0, 2])
def test_at_most_lookahead_plus_one_units_are_resident(cpu_ipc, lookahead):
    """The rank gathers `gather_lookahead` units ahead of a worker pulling one
    at a time, no more, and serves the published bytes."""
    pairs = _pairs(6)
    engine = _engine(_Source(pairs), buffer_bytes=32, gather_lookahead=lookahead)
    server = _attach_server(engine)
    server.register_consumer(0, [[(n, ((0, 8),))] for n, _ in pairs])
    peak, served, remote = [0], [], engine._remote

    def counting_remote(method, *args):
        if method == "publish_unit":
            peak[0] = max(peak[0], len(engine._inflight))
        return remote(method, *args)

    engine._remote = counting_remote

    def consumer():
        for seq in range(len(pairs)):
            (t,) = server.bucket(0, seq, 0)
            served.append(t.view(torch.float32)[0].item())
            server.free()

    worker = threading.Thread(target=consumer)
    worker.start()
    engine.send_weights()
    worker.join(timeout=20)
    assert peak[0] <= lookahead + 1
    assert served == [float(i) for i in range(6)]


def test_a_source_out_of_order_fails_the_sync(cpu_ipc):
    pairs = _pairs(2)
    source = _Source(pairs)
    engine = _engine(source, buffer_bytes=32)
    source._pairs = pairs[::-1]  # iter_full now disagrees with the metadata
    server = _attach_server(engine)
    server.register_consumer(0, [])
    with pytest.raises(RuntimeError, match="metadata order"):
        engine.send_weights()
    assert server._error is not None
