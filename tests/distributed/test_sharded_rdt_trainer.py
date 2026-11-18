# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sharded-RDT trainer engine, on CPU: init info, the ownership and unit
plan it derives from metadata, the worker init payload, and the gather loop
against a real producer server (CUDA IPC replaced by an in-process registry).
"""

import json
import threading
from functools import cached_property

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
    """Ordered (name, tensor) pairs with optional owners; yields what `rank`
    owns."""

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
        # rebuild_cuda_tensor's arguments: [7:10] locate the allocation
        return None, (ptr, 0, 0, 0, 0, 0, 0, ptr, base.nbytes, 0)

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


def _attach_server(engine, consumer_ids=(0,)):
    """A real producer server for `engine`, as `_spawn_server` would build it."""
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


# ------------------------------------------------------------ init info


@pytest.mark.parametrize(
    "kw,match",
    [
        ({"buffer_bytes": 0}, "buffer_bytes"),
        ({"num_buffers": 0}, "num_buffers"),
        ({"num_consumers": 0}, "num_consumers"),
    ],
)
def test_init_info_validates(kw, match):
    kw = {"num_consumers": 1} | kw
    with pytest.raises(ValueError, match=match):
        ShardedRDTTrainerInitInfo(rank=0, **kw)


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


# ------------------------------------------------------------ plan


def test_ownership_comes_from_metadata_alike_on_every_rank():
    owned = [
        {"embed.weight", "model.layers.0.w", "norm.weight"},
        {"model.layers.1.w", "norm.weight"},
    ]
    tables = []
    for rank in (0, 1):
        engine = _engine(_Source(_PP, _PP_OWNERS, rank), rank=rank, world=2)
        tables.append((engine._owner_sets, engine._name_owner_class))
        assert engine._owned_names == owned[rank]
    assert tables[0] == tables[1] == ([[0], [1], [0, 1]], [0, 0, 1, 2])


def test_units_are_global_and_a_rank_gathers_those_it_owns_part_of():
    """8 B per name, 16 B units: [embed, l0], [l1, norm] on every rank."""
    for rank, units in ((0, [0, 1]), (1, [1])):
        engine = _engine(
            _Source(_PP, _PP_OWNERS, rank), rank=rank, world=2, buffer_bytes=16
        )
        assert engine._units == [0, 0, 1, 1]
        assert list(engine._owned_units) == units


def test_owners_outside_the_trainer_are_refused():
    with pytest.raises(ValueError, match="only 2 trainer ranks"):
        _engine(_Source(_PP, {"embed.weight": (0, 2)}), world=2)


def test_arena_covers_the_consumers_that_may_pull_from_the_rank():
    """With experts on rank 1 only, both consumers may pull from it; rank 0
    serves the shared names to consumer 0 alone."""
    owners = {"model.layers.1.w": (1,)}
    sizes = []
    for rank in (0, 1):
        engine = _engine(
            _Source(_PP, owners, rank),
            rank=rank,
            world=2,
            num_consumers=2,
            buffer_bytes=100,
            num_buffers=2,
        )
        sizes.append(engine._arena_bytes())
    assert sizes == [(2 * 1 + 1) * 116, (2 * 2 + 1) * 116]


def test_worker_payload_survives_json():
    engine = _engine(_Source(_PP, _PP_OWNERS, 0), world=2, buffer_bytes=64)
    engine._server_names = ["srv0", "srv1"]
    payload = json.loads(json.dumps(engine.get_worker_init_payload()))
    assert payload["trainer_actor_names"] == ["srv0", "srv1"]
    assert payload["owner_sets"] == [[0], [1], [0, 1]]
    assert payload["name_owner_class"] == [0, 0, 1, 2]
    assert payload["dtype_names"] == ["float32"] * 4
    assert payload["buffer_bytes"] == 64


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


def test_the_sender_drives_the_workers_around_its_gathers(cpu_ipc):
    client = RecordingClient()
    engine = _engine(_Source(_pairs(2)), client=client, buffer_bytes=32)
    server = _attach_server(engine)
    server.register_consumer(0, [])  # pulls nothing: units release at publish
    engine.send_weights()
    assert client.last_update_info == {"sync_idx": 0}
    engine.send_weights()
    assert client.order == ["start", "update", "finish"] * 2
    assert client.last_update_info == {"sync_idx": 1}
    assert not engine._inflight


def test_a_failed_update_fails_the_sync_on_every_producer_at_once(cpu_ipc, monkeypatch):
    """A worker's pull fails (say NIXL cannot register a block): the sync
    fails with that error right away, not after the stall watchdog."""
    import time

    class _Failing(RecordingClient):
        def update_weights(self, update_info):
            raise RuntimeError("Failed to register cuda memory with NIXL")

    engine = _engine(_Source(_pairs(2)), client=_Failing(), buffer_bytes=32)
    server = _attach_server(engine)
    # consumer 0 needs both units but never pulls: only the error ends the sync
    server.register_consumer(
        0, [[("model.layers.0.w", [[0, 8]])], [("model.layers.1.w", [[0, 8]])]]
    )
    engine._server_names = ["rk0"]

    class _Actor:
        class set_gather_error:  # noqa: N801
            remote = staticmethod(server.set_gather_error)

    monkeypatch.setattr(trainer_mod.ray, "get_actor", lambda *a, **k: _Actor)
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="register cuda memory"):
        engine.send_weights()
    assert time.monotonic() - t0 < 10  # the stall watchdog is 20 s


def test_a_rank_gathers_only_units_it_owns(cpu_ipc):
    source = _Source(_PP, _PP_OWNERS, 1)
    engine = _engine(source, rank=1, world=2, buffer_bytes=16)
    server = _attach_server(engine)
    server.register_consumer(0, [])
    engine.send_weights()
    assert source.yielded == ["model.layers.1.w", "norm.weight"]


@pytest.mark.parametrize("lookahead", [0, 1, 2])
def test_at_most_lookahead_plus_one_units_are_resident(cpu_ipc, lookahead):
    """A worker pulls every unit, one at a time; the rank gathers ahead of it
    by `gather_lookahead` units and no more, and every byte served is the
    published one."""
    pairs = _pairs(6)
    engine = _engine(_Source(pairs), buffer_bytes=32, gather_lookahead=lookahead)
    server = _attach_server(engine)
    server.register_consumer(0, [[(n, [[0, 8]])] for n, _ in pairs])
    peak = [0]
    served = []
    original = engine._remote

    def remote(method, *args):
        if method == "publish_unit":
            peak[0] = max(peak[0], len(engine._inflight))
        return original(method, *args)

    engine._remote = remote

    def consumer():
        for seq in range(len(pairs)):
            (t,) = server.bucket(0, seq, 0)
            served.append(t.view(torch.float32).clone())
            server.free()

    worker = threading.Thread(target=consumer)
    worker.start()
    engine.send_weights()
    worker.join(timeout=20)
    assert peak[0] <= lookahead + 1
    assert [t[0].item() for t in served] == [float(i) for i in range(6)]


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


def test_a_degraded_sync_serves_the_live_set(cpu_ipc):
    engine = _engine(_Source(_pairs(1)), num_consumers=3, buffer_bytes=32)
    server = _attach_server(engine)
    server.register_consumer(2, [])
    engine.send_weights(live_consumer_ids=[2])
    assert server._live == {2}
