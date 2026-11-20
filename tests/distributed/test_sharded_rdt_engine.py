# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sharded-RDT consumer on CPU: the real engine pulls from real producer
servers through a loopback for Ray, and must equal a cold start after two
syncs, byte for byte and address for address."""

import dataclasses
import weakref
from types import SimpleNamespace

import pytest
import torch

import vllm.distributed.weight_transfer.sharded_rdt_engine as engine_mod
from tests.distributed.sharded_rdt_utils import make_server, patch_for_cpu
from tests.model_executor.model_loader.test_load_plan import (
    _EXPERT_SHAPES,
    _LINEAR_SHAPES,
    _MODEL_CONFIG,
    _TOY_SHAPES,
    _checkpoint,
    _cold_start,
    _experts_model,
    _linear_model,
    _state,
    _toy_model,
)
from vllm.distributed.weight_transfer.sharded_rdt_common import (
    RdtRouter,
    gather_units,
)
from vllm.distributed.weight_transfer.sharded_rdt_engine import (
    ShardedRDTWeightTransferEngine,
    ShardedRDTWeightTransferInitInfo,
    ShardedRDTWeightTransferUpdateInfo,
    _Bucket,
    _PullHooks,
)
from vllm.model_executor.model_loader.reload import lifecycle
from vllm.model_executor.model_loader.reload.direct import track_loaded_params
from vllm.model_executor.model_loader.utils import process_weights_after_loading

BF16 = torch.bfloat16


@pytest.fixture(autouse=True)
def _direct_mode(monkeypatch):
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    monkeypatch.setattr(lifecycle, "_weight_update_active", False)
    monkeypatch.setattr(engine_mod, "check_ray_rdt_version", lambda: None)
    patch_for_cpu(monkeypatch)
    _PINNED.clear()
    monkeypatch.setattr(engine_mod, "register_nixl_memory", _PINNED.append)
    monkeypatch.setattr(engine_mod, "deregister_nixl_memory", _PINNED.remove)


# receive arenas registered with NIXL
_PINNED: list[torch.Tensor] = []


# ------------------------------------------------------------ loopback


class _Handle:
    def __init__(self, server, sent, target, key):
        # Ray keeps only a weak reference to a target
        self.server, self.sent, self.key = server, sent, key
        self.target = weakref.ref(target)

    def __del__(self):
        # dropping the last ref frees the object, and with it the block
        self.server._freed_bucket(*self.key)


class _Loopback:
    """Ray's part of a pull, in process."""

    def __init__(self, servers, consumer_id):
        self.servers, self.consumer_id = servers, consumer_id
        self.pulls: list[_Bucket] = []

    def hooks(self) -> _PullHooks:
        return _PullHooks(
            issue=self.issue,
            wait=lambda hs: list(hs),
            get=self.get,
            register=self.register,
        )

    def register(self, per_producer):
        for server, buckets in zip(self.servers, per_producer):
            server.register_consumer(self.consumer_id, buckets)

    def issue(self, bucket, sync_idx, target):
        self.pulls.append(bucket)
        server = self.servers[bucket.producer]
        (sent,) = server.bucket(self.consumer_id, bucket.seq, sync_idx)
        assert any(
            target.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
            for t in _PINNED
        ), "the target is not in the pinned arena"
        assert sent.dtype == target.dtype and sent.shape == target.shape
        return _Handle(server, sent, target, server.outstanding.pop())

    def get(self, handles, targets):
        """Ray's get: poison the target, then land the pull in it."""
        for h, target in zip(handles, targets, strict=True):
            assert h.target() is target, "the target died in flight"
            target.fill_(0xA5)
            target.copy_(h.sent)


# ------------------------------------------------------------ fleet


def _metas_info(shapes):
    names = list(shapes)
    return dict(
        names=names,
        dtype_names=[str(dt).removeprefix("torch.") for _, dt in shapes.values()],
        shapes=[list(shape) for shape, _ in shapes.values()],
    )


class _Fleet:
    """Real producers holding `shapes` per `owners`, and one engine on `model`."""

    def __init__(
        self,
        monkeypatch,
        model,
        shapes,
        *,
        owners=None,
        num_producers=2,
        buffer_bytes=64,
        num_buffers=2,
        consumer_id=0,
        num_consumers=1,
        servers=None,
    ):
        owners = owners or {}
        owner_sets: list[list[int]] = []
        classes = []
        for name in shapes:
            own = sorted(owners.get(name, range(num_producers)))
            if own not in owner_sets:
                owner_sets.append(own)
            classes.append(owner_sets.index(own))
        nbytes = [torch.Size(s).numel() * dt.itemsize for s, dt in shapes.values()]
        units = dict(zip(shapes, gather_units(nbytes, buffer_bytes)))
        self.shapes, self.units = shapes, units
        self.servers = servers or [
            make_server(
                served={
                    n: (str(dt).removeprefix("torch."), tuple(s))
                    for i, (n, (s, dt)) in enumerate(shapes.items())
                    if p in owner_sets[classes[i]]
                },
                units=units,
                arena_bytes=1 << 22,
                bucket_bytes=buffer_bytes,
                stall_timeout_s=5,
            )
            for p in range(num_producers)
        ]
        self.owner_sets, self.classes = owner_sets, classes
        monkeypatch.setattr(engine_mod, "_resolve_producers", lambda ii: self.servers)
        self.loopback = _Loopback(self.servers, consumer_id)
        parallel = SimpleNamespace(
            distributed_executor_backend="ray",
            data_parallel_index=0,
            world_size=num_consumers,
            rank=consumer_id,
            world_size_across_dp=num_consumers,
        )
        self.engine = ShardedRDTWeightTransferEngine(
            SimpleNamespace(backend="sharded_rdt"),
            SimpleNamespace(parallel_config=parallel, model_config=_MODEL_CONFIG),
            torch.device("cpu"),
            model,
        )
        self.engine.init_transfer_engine(
            ShardedRDTWeightTransferInitInfo(
                trainer_actor_names=[f"p{p}" for p in range(num_producers)],
                owner_sets=owner_sets,
                name_owner_class=classes,
                num_consumers=num_consumers,
                buffer_bytes=buffer_bytes,
                num_buffers=num_buffers,
                **_metas_info(shapes),
            )
        )
        self.engine._hooks = self.loopback.hooks()

    def sync(self, weights, sync_idx):
        self.engine.start_weight_update()
        for p, server in enumerate(self.servers):
            server.begin_sync([self.loopback.consumer_id], sync_idx)
            by_unit: dict[int, dict] = {}
            for i, name in enumerate(self.shapes):
                if p in self.owner_sets[self.classes[i]]:
                    by_unit.setdefault(self.units[name], {})[name] = weights[name]
            for unit, tensors in sorted(by_unit.items()):
                server._publish(unit, tensors)
        self.engine.update_weights({"sync_idx": sync_idx})
        for server in self.servers:
            server.end_sync()
            assert not server._resident, "a unit outlived the sync"
            assert not server._blocks
        self.engine.finish_weight_update()


def _check_equals_cold_start(fleet, make, shapes):
    ptrs = {n: s[4] for n, s in _state(fleet.engine.model).items()}
    for sync_idx in (1, 2):
        weights = _checkpoint(shapes, seed=sync_idx)
        fleet.sync(weights, sync_idx)
        actual, expected = (
            _state(fleet.engine.model),
            _state(_cold_start(make, weights)),
        )
        assert actual.keys() == expected.keys()
        for name, (dtype, shape, stride, data, ptr) in actual.items():
            assert (dtype, shape, stride) == expected[name][:3], name
            assert torch.equal(
                data.view(torch.uint8), expected[name][3].view(torch.uint8)
            ), name
            assert ptr == ptrs[name], f"{name} moved"


def _toy_fleet(monkeypatch, **kw):
    model = _toy_model()
    track_loaded_params(model)  # so finish checks each update loaded everything
    model.load_weights(_checkpoint(_TOY_SHAPES, seed=0).items())
    process_weights_after_loading(model, _MODEL_CONFIG, torch.device("cpu"))
    return _Fleet(monkeypatch, model, _TOY_SHAPES, **kw)


def _staged(bucket):
    return {staging.name for _, staging, _ in bucket.staged}


def _tp(monkeypatch, rank, size):
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear

    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: rank)
        monkeypatch.setattr(
            module, "get_tensor_model_parallel_world_size", lambda: size
        )


# ------------------------------------------------------------ byte equality


@pytest.mark.parametrize("buffer_bytes", [24, 64, 1 << 20])
def test_toy_loaders_equal_cold_start(monkeypatch, buffer_bytes):
    """Transpose (strided), tied alias (not pulled), cast and step slice
    (fallbacks), embedding; `router.bias` only on producer 1. At 24 B pieces
    go in row segments, a fallback's joined in its staging."""
    fleet = _toy_fleet(
        monkeypatch, owners={"router.bias": [1]}, buffer_bytes=buffer_bytes
    )
    _check_equals_cold_start(fleet, _toy_model, _TOY_SHAPES)
    buckets = fleet.engine._buckets
    assert "lm_head.weight" not in {
        n for b in fleet.loopback.pulls for n, _ in b.pieces
    }
    assert {b.producer for b in buckets if "router.bias" in _staged(b)} == {1}
    (b, i) = next(
        (b, i)
        for b in buckets
        for i, (n, _) in enumerate(b.pieces)
        if n == "proj.weight"
    )
    assert [(j, d.stride()) for j, d in b.typed] == [(i, (1, 4))]
    pieces = [n for b in buckets for n, _ in b.pieces]
    split = {n for n in pieces if pieces.count(n) > 1}
    if buffer_bytes == 24:
        staged = set().union(*map(_staged, buckets))
        assert split - staged and split & staged  # a direct and a fallback split
        assert all(st.tensor is None for b in buckets for _, st, _ in b.staged)
    # segments are cut to the largest regular bucket: the slot does not grow
    regular = [b.nbytes for b in buckets if not split & {n for n, _ in b.pieces}]
    assert fleet.engine._slot_bytes == max(regular) <= buffer_bytes


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("buffer_bytes", [32, 1 << 20])
def test_qkv_and_merged_column_under_tp(
    default_vllm_config, monkeypatch, tp_rank, buffer_bytes
):
    _tp(monkeypatch, tp_rank, 2)
    make = _linear_model()
    model = _cold_start(make, _checkpoint(_LINEAR_SHAPES, seed=0))
    fleet = _Fleet(monkeypatch, model, _LINEAR_SHAPES, buffer_bytes=buffer_bytes)
    _check_equals_cold_start(fleet, make, _LINEAR_SHAPES)
    assert all(not b.typed and not b.staged for b in fleet.engine._buckets)
    if buffer_bytes > 32:  # q, k, v (gate, up) land in a fused weight with one copy
        copies = [d for b in fleet.engine._buckets for d in b.byte_dsts]
        assert len({d.untyped_storage().data_ptr() for d in copies}) == len(copies)

    # a replica whose weights sit in reverse address order cuts the same buckets
    other = _cold_start(make, _checkpoint(_LINEAR_SHAPES, seed=0))
    params = list(other.parameters())
    arena = bytearray(1024 * len(params))  # a storage per weight, placed by hand
    for i, p in enumerate(params):
        at = 1024 * (len(params) - 1 - i)
        raw = torch.frombuffer(memoryview(arena)[at : at + 1024], dtype=torch.uint8)
        p.data = raw[: p.nbytes].view(p.dtype).view(p.shape).copy_(p.data)
    other_fleet = _Fleet(monkeypatch, other, _LINEAR_SHAPES, buffer_bytes=buffer_bytes)
    other_fleet.engine.start_weight_update()
    assert [(b.producer, b.wire()) for b in other_fleet.engine._buckets] == [
        (b.producer, b.wire()) for b in fleet.engine._buckets
    ]


@pytest.mark.parametrize("ep_rank", [0, 1])
def test_experts_under_ep_owned_by_one_producer(monkeypatch, ep_rank):
    make = _experts_model(ep_rank, 2, 0, 1)
    model = _cold_start(make, _checkpoint(_EXPERT_SHAPES, seed=0))
    owners = {n: [1] for n in _EXPERT_SHAPES}
    fleet = _Fleet(monkeypatch, model, _EXPERT_SHAPES, owners=owners)
    _check_equals_cold_start(fleet, make, _EXPERT_SHAPES)
    assert {b.producer for b in fleet.engine._buckets} == {1}
    rows = {box[0] for b in fleet.engine._buckets for _, box in b.pieces}
    assert all(2 * ep_rank <= a < z <= 2 * ep_rank + 2 for a, z in rows)


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("buffer_bytes", [16, 1 << 20])
def test_expert_tp_two_boxes_per_name(monkeypatch, tp_rank, buffer_bytes):
    """Gate and up rows are two strided boxes of one name; at 16 B every row
    is its own pull."""
    make = _experts_model(0, 1, tp_rank, 2)
    model = _cold_start(make, _checkpoint(_EXPERT_SHAPES, seed=0))
    fleet = _Fleet(monkeypatch, model, _EXPERT_SHAPES, buffer_bytes=buffer_bytes)
    _check_equals_cold_start(fleet, make, _EXPERT_SHAPES)
    buckets = fleet.engine._buckets
    gate_up = [n for b in buckets for n, _ in b.pieces if n == "experts.gate_up_proj"]
    if buffer_bytes == 16:
        assert len(gate_up) > 2 and all(len(b.pieces) == 1 for b in buckets)
    else:
        assert len(gate_up) == 2


def test_plan_failure_leaves_a_note_and_retries(monkeypatch):
    fleet = _toy_fleet(monkeypatch)
    hooks = fleet.engine._hooks

    def boom(_):
        raise RuntimeError("producers unreachable")

    fleet.engine._hooks = dataclasses.replace(hooks, register=boom)
    with pytest.raises(RuntimeError, match="unreachable") as info:
        fleet.engine.start_weight_update()
    assert "mid-reload" in "".join(info.value.__notes__)
    fleet.engine._hooks = hooks
    _check_equals_cold_start(fleet, _toy_model, _TOY_SHAPES)


# ------------------------------------------------------------ plan and transport


def test_a_pull_that_misses_its_slot_fails_the_sync(monkeypatch):
    """Ray silently receives into its own buffer when it cannot use a target."""
    import ray
    import ray.experimental

    fleet = _toy_fleet(monkeypatch)
    target = torch.zeros(16, dtype=torch.uint8)
    monkeypatch.setattr(ray.experimental, "set_target_for_ref", lambda ref, ts: None)
    monkeypatch.setattr(ray, "get", lambda refs: [[target.clone()] for _ in refs])
    with pytest.raises(RuntimeError, match="did not land"):
        fleet.engine._ray_hooks().get(["ref"], [target])


def test_a_row_over_buffer_bytes_is_refused_at_plan_time(monkeypatch):
    with pytest.raises(RuntimeError, match="raise buffer_bytes"):
        _toy_fleet(monkeypatch, buffer_bytes=4).engine.start_weight_update()


def test_rows_round_down_to_whole_rows_within_the_cap():
    t = torch.zeros(10, 3, dtype=BF16)  # rows of 6 B: 6 fit in 40 B, 5 in 32
    piece = ("w", ((0, 10), (0, 3)), BF16, [t])
    segs = engine_mod._rows(piece, 40)
    assert [seg[1][0] for seg in segs] == [(0, 5), (5, 10)]
    assert [seg[3][0].data_ptr() for seg in segs] == [t.data_ptr(), t[5].data_ptr()]
    assert engine_mod._rows(piece, 64) == [piece]


def test_receive_arena_is_pinned_once_per_plan(monkeypatch):
    fleet = _toy_fleet(monkeypatch, buffer_bytes=48, num_buffers=3)
    for sync_idx in (1, 2):
        fleet.sync(_checkpoint(_TOY_SHAPES, seed=sync_idx), sync_idx)
        (arena,) = _PINNED
        assert arena.numel() == 3 * max(b.nbytes for b in fleet.engine._buckets)
    fleet.engine.shutdown()
    assert not _PINNED


def test_buckets_never_cross_a_unit_and_respect_the_budget(monkeypatch):
    fleet = _toy_fleet(monkeypatch, buffer_bytes=48)
    fleet.engine.start_weight_update()
    seqs: dict[int, list[int]] = {}
    for b in fleet.engine._buckets:
        assert {fleet.units[n] for n, _ in b.pieces} == {b.unit} and b.nbytes <= 48
        seqs.setdefault(b.producer, []).append(b.seq)
    assert all(s == list(range(len(s))) for s in seqs.values())


def test_pipeline_stages_pull_side_by_side(monkeypatch):
    """Stages interleave, each in unit order, instead of draining stage 0."""
    _tp(monkeypatch, 0, 1)
    make = _linear_model()
    model = _cold_start(make, _checkpoint(_LINEAR_SHAPES, seed=0))
    owners = {n: [0] if i < 3 else [1] for i, n in enumerate(_LINEAR_SHAPES)}
    fleet = _Fleet(monkeypatch, model, _LINEAR_SHAPES, owners=owners, buffer_bytes=128)
    fleet.engine.start_weight_update()
    buckets = fleet.engine._buckets
    assert [b.producer for b in buckets][:2] == [0, 1]
    for p in (0, 1):
        units = [b.unit for b in buckets if b.producer == p]
        assert units == sorted(units)


def test_init_and_receive_refusals(monkeypatch):
    fleet = _toy_fleet(monkeypatch)
    with pytest.raises(RuntimeError, match="start_weight_update"):
        fleet.engine.receive_weights(ShardedRDTWeightTransferUpdateInfo(sync_idx=0))
    bad = ShardedRDTWeightTransferInitInfo(
        trainer_actor_names=["p0"], names=["a", "b"], dtype_names=["bf16"], shapes=[[1]]
    )
    with pytest.raises(ValueError, match="2 names, 1 dtypes"):
        fleet.engine.init_transfer_engine(bad)
    monkeypatch.setenv("VLLM_RELOAD_MODE", "layerwise")
    with pytest.raises(RuntimeError, match="sharded_rdt.*VLLM_RELOAD_MODE=direct"):
        _toy_fleet(monkeypatch)


@pytest.mark.parametrize(
    "replica_rank,rank,expected",
    [(0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3), (2, 0, None), (0, 2, None)],
)
def test_two_deployments_get_distinct_consumer_ids(
    monkeypatch, replica_rank, rank, expected
):
    """Four workers in two deployments of two: the trainer ships
    `workers_per_replica`, each deployment sets its `replica_rank`."""
    fleet = _toy_fleet(monkeypatch, num_consumers=4)
    fleet.engine.parallel_config.world_size = 2
    fleet.engine.parallel_config.rank = rank
    info = ShardedRDTWeightTransferInitInfo(
        trainer_actor_names=["p0", "p1"],
        num_consumers=4,
        workers_per_replica=2,
        replica_rank=replica_rank,
        **_metas_info(_TOY_SHAPES),
    )
    if expected is None:
        with pytest.raises(ValueError, match="2 per deployment"):
            fleet.engine.init_transfer_engine(info)
    else:
        fleet.engine.init_transfer_engine(info)
        assert fleet.engine._consumer_id == expected


def test_a_failed_sync_leaves_no_partial_fallback(monkeypatch):
    """A sync that fails between a fallback's segments; the next one must
    load that fallback whole again."""
    fleet = _toy_fleet(monkeypatch, buffer_bytes=24)
    get = fleet.loopback.get

    def get_then_fail(handles, targets):
        stagings = [st for b in fleet.engine._buckets for _, st, _ in b.staged]
        if any(st.landed for st in stagings):
            raise RuntimeError("pull failed")
        get(handles, targets)

    fleet.loopback.get = get_then_fail
    fleet.engine._hooks = fleet.loopback.hooks()
    with pytest.raises(RuntimeError, match="pull failed"):
        fleet.sync(_checkpoint(_TOY_SHAPES, seed=5), 5)
    with pytest.raises(RuntimeError, match="did not load"):  # a partial update
        fleet.engine.finish_weight_update()
    fleet.loopback.get = get
    fleet.engine._hooks = fleet.loopback.hooks()
    _check_equals_cold_start(fleet, _toy_model, _TOY_SHAPES)


# ------------------------------------------------------------ common


@pytest.mark.parametrize("installed,ok", [("2.56.0", True), ("2.55.1", False)])
def test_ray_version_floor(monkeypatch, installed, ok):
    import importlib.metadata

    from vllm.distributed.weight_transfer import sharded_rdt_common

    monkeypatch.setattr(importlib.metadata, "version", lambda pkg: installed)
    if ok:
        sharded_rdt_common.check_ray_rdt_version()
    else:
        with pytest.raises(ValueError, match="requires Ray"):
            sharded_rdt_common.check_ray_rdt_version()


def test_gather_units():
    assert gather_units([4, 4, 4, 4], 8) == [0, 0, 1, 1]
    assert gather_units([4, 20, 4], 8) == [0, 1, 2]  # an oversized name alone


@pytest.mark.parametrize(
    "router,expected",
    [
        (RdtRouter(4, 4), [0, 1, 2, 3]),  # identity
        (RdtRouter(2, 4), [0, 0, 1, 1]),  # fan-in
        (RdtRouter(2, 4, workers_per_replica=2), [0, 1, 0, 1]),  # two deployments
    ],
)
def test_router_spreads_consumers(router, expected):
    assert [router.producer_for(c, "w", 0) for c in range(4)] == expected


def test_router_owner_sets_fan_in_and_components():
    assert {RdtRouter(4, 2).producer_for(0, "w", u) for u in range(4)} == {0, 1}
    r = RdtRouter(4, 2, [[0, 1], [2, 3], [3]], [0, 1, 2], ["s0", "s1", "e"])
    assert r.producer_for(1, "s0", 0) in (0, 1) and r.producer_for(0, "s1", 0) in (2, 3)
    assert r.producer_for(0, "e", 0) == r.producer_for(1, "e", 5) == 3
    assert r.consumers_of(0) == [0] and r.consumers_of(3) == [0, 1]
    assert r.components() == [0, 0, 2, 2]
    assert len(set(RdtRouter(3, 1).components())) == 1


@pytest.mark.parametrize(
    "owner_sets,classes,match",
    [([[]], [0], "empty"), ([[0, 5]], [0], "out of range"), ([[0]], [3], "class 3")],
)
def test_router_validate(owner_sets, classes, match):
    with pytest.raises(ValueError, match=match):
        RdtRouter(2, 1, owner_sets, classes, ["w"]).validate()
    with pytest.raises(ValueError, match="uniform"):
        RdtRouter(2, 3, workers_per_replica=2)
