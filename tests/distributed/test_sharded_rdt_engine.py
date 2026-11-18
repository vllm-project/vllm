# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sharded-RDT consumer on the load plan, on CPU.

The engine runs for real against real producer servers through a loopback
transport that does what Ray's receive into a target does: a pull's one flat
byte tensor is copied into the slice of the receive arena the engine named,
after the slice is poisoned. Getting the pull frees it at the producer.
After two syncs every tensor of the model must equal a cold start on the
same weights, byte for byte, at the same address.
"""

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
    arena_bytes,
    flat_layout,
    flat_view,
    gather_units,
    piece_nbytes,
)
from vllm.distributed.weight_transfer.sharded_rdt_engine import (
    ShardedRDTWeightTransferEngine,
    ShardedRDTWeightTransferInitInfo,
    ShardedRDTWeightTransferUpdateInfo,
    _Bucket,
    _PullHooks,
)
from vllm.model_executor.model_loader.reload import lifecycle

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
        """Ray's get: into the target while it lives, after poisoning it."""
        for h, target in zip(handles, targets, strict=True):
            assert h.target() is target, "the target died in flight"
            target.fill_(0xA5)
            target.copy_(h.sent)
            h.server._freed_bucket(*h.key)


# ------------------------------------------------------------ fleet


def _metas_info(shapes):
    names = list(shapes)
    return dict(
        names=names,
        dtype_names=[str(dt).removeprefix("torch.") for _, dt in shapes.values()],
        shapes=[list(shape) for shape, _ in shapes.values()],
    )


class _Fleet:
    """`num_producers` real servers holding `shapes` per `owners`, and one
    consumer engine on `model`."""

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
        """One sync: start (plans the first time), the trainers gather every
        unit, the worker pulls, every unit is released, finish."""
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


# ------------------------------------------------------------ byte equality


@pytest.mark.parametrize("buffer_bytes", [64, 1 << 20])
def test_toy_loaders_equal_cold_start(monkeypatch, buffer_bytes):
    """Transpose (a strided copy), a tied alias (nothing to pull), a cast and
    a step slice (fallbacks), and an embedding; `router.bias` lives only on
    producer 1."""
    model = _cold_start(_toy_model, _checkpoint(_TOY_SHAPES, seed=0))
    fleet = _Fleet(
        monkeypatch,
        model,
        _TOY_SHAPES,
        owners={"router.bias": [1]},
        buffer_bytes=buffer_bytes,
    )
    _check_equals_cold_start(fleet, _toy_model, _TOY_SHAPES)
    pulled = {name for b in fleet.loopback.pulls for name, _ in b.pieces}
    assert "lm_head.weight" not in pulled
    (router,) = [
        b for b in fleet.engine._buckets if ("router.bias", ((0, 5),)) in b.pieces
    ]
    assert router.producer == 1 and "router.bias" in _staged(router)
    pieces = {
        name: (b, i)
        for b in fleet.engine._buckets
        for i, (name, _) in enumerate(b.pieces)
    }
    b, i = pieces["proj.weight"]
    assert "proj.weight" not in _staged(b)
    ((j, dest),) = b.typed
    assert j == i and dest.stride() == (1, 4)  # strided
    start, end = b.layout[i][0], b.layout[i][0] + b.layout[i][1]
    assert all(o + n <= start or o >= end for o, n in b.byte_srcs)


def _staged(bucket):
    """Names of the fallbacks whose pieces `bucket` carries."""
    return {staging.name for _, staging, _ in bucket.staged}


def test_a_pull_that_misses_its_slot_fails_the_sync(monkeypatch):
    """Ray falls back to its own buffer, silently, when it cannot use a
    target; the engine must not copy the stale slot out."""
    import ray
    import ray.experimental

    model = _cold_start(_toy_model, _checkpoint(_TOY_SHAPES, seed=0))
    fleet = _Fleet(monkeypatch, model, _TOY_SHAPES)
    target = torch.zeros(16, dtype=torch.uint8)
    monkeypatch.setattr(ray.experimental, "set_target_for_ref", lambda ref, ts: None)
    monkeypatch.setattr(ray, "get", lambda refs: [[target.clone()] for _ in refs])
    hooks = fleet.engine._ray_hooks()
    with pytest.raises(RuntimeError, match="did not land"):
        hooks.get(["ref"], [target])


def test_pieces_over_buffer_bytes_go_in_row_segments(monkeypatch):
    """A piece over `buffer_bytes` is cut into segments of whole rows: a
    direct destination's land in its rows, a fallback's are joined and
    loaded whole; every pull fits a receive slot."""
    model = _cold_start(_toy_model, _checkpoint(_TOY_SHAPES, seed=0))
    fleet = _Fleet(monkeypatch, model, _TOY_SHAPES, buffer_bytes=24, num_buffers=2)
    _check_equals_cold_start(fleet, _toy_model, _TOY_SHAPES)
    pieces = [(n, box) for b in fleet.engine._buckets for n, box in b.pieces]
    split = {n for n, _ in pieces if sum(m == n for m, _ in pieces) > 1}
    loads = set().union(*map(_staged, fleet.engine._buckets))
    assert split - loads and split & loads  # a direct and a fallback split
    assert all(b.nbytes <= 24 for b in fleet.engine._buckets)
    # every staging was loaded and freed
    assert all(
        st.tensor is None for b in fleet.engine._buckets for _, st, _ in b.staged
    )
    # segments are no larger than the buckets the other pieces make, so the
    # receive slot does not grow
    regular = [
        b.nbytes
        for b in fleet.engine._buckets
        if not any(n in split for n, _ in b.pieces)
    ]
    assert fleet.engine._slot_bytes == max(regular)


def test_a_row_over_buffer_bytes_is_refused_at_plan_time(monkeypatch):
    model = _cold_start(_toy_model, _checkpoint(_TOY_SHAPES, seed=0))
    fleet = _Fleet(monkeypatch, model, _TOY_SHAPES, buffer_bytes=4)
    with pytest.raises(RuntimeError, match="raise buffer_bytes"):
        fleet.sync(_checkpoint(_TOY_SHAPES, seed=1), 1)


def test_rows_round_down_to_whole_rows_within_the_budget():
    t = torch.zeros(10, 3, dtype=torch.bfloat16)  # rows of 6 B
    piece = ("w", ((0, 10), (0, 3)), torch.bfloat16, [t])
    segs = engine_mod._rows(piece, 32)  # 5 rows = 30 B, with alignment 32
    assert [seg[1][0] for seg in segs] == [(0, 5), (5, 10)]
    assert [seg[3][0].data_ptr() for seg in segs] == [t.data_ptr(), t[5].data_ptr()]
    assert engine_mod._rows(piece, 64) == [piece]


def test_receive_arena_is_pinned_once_per_plan(monkeypatch):
    """One slot of the largest bucket of at most `buffer_bytes` per pull in
    flight, pinned at the first plan, reused by every sync, and unpinned when
    the plan is dropped."""
    model = _cold_start(_toy_model, _checkpoint(_TOY_SHAPES, seed=0))
    fleet = _Fleet(monkeypatch, model, _TOY_SHAPES, buffer_bytes=48, num_buffers=3)
    for sync_idx in (1, 2):
        fleet.sync(_checkpoint(_TOY_SHAPES, seed=sync_idx), sync_idx)
        (arena,) = _PINNED
        slot = max(b.nbytes for b in fleet.engine._buckets)
        assert arena.numel() == 3 * slot and arena is fleet.engine._arena
    fleet.engine.shutdown()
    assert not _PINNED


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("buffer_bytes", [32, 1 << 20])
def test_qkv_and_merged_column_under_tp(
    default_vllm_config, monkeypatch, tp_rank, buffer_bytes
):
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear

    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: tp_rank)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 2)
    make = _linear_model()
    model = _cold_start(make, _checkpoint(_LINEAR_SHAPES, seed=0))
    fleet = _Fleet(monkeypatch, model, _LINEAR_SHAPES, buffer_bytes=buffer_bytes)
    _check_equals_cold_start(fleet, make, _LINEAR_SHAPES)
    # every destination is contiguous: one fused byte copy per pull
    assert all(not b.typed and not b.staged for b in fleet.engine._buckets)
    if buffer_bytes > 32:
        # q, k, v (and gate, up) fill adjacent rows of one fused weight, so
        # a pull lands each fused weight's pieces from it with one copy
        copies = [d for b in fleet.engine._buckets for d in b.byte_dsts]
        assert len({d.untyped_storage().data_ptr() for d in copies}) == len(copies)


@pytest.mark.parametrize("ep_rank", [0, 1])
def test_experts_under_ep_owned_by_one_producer(monkeypatch, ep_rank):
    """Experts held only by producer 1 (an EP trainer); this worker needs
    only its own two experts of each."""
    make = _experts_model(ep_rank, 2, 0, 1)
    model = _cold_start(make, _checkpoint(_EXPERT_SHAPES, seed=0))
    fleet = _Fleet(
        monkeypatch,
        model,
        _EXPERT_SHAPES,
        owners={n: [1] for n in _EXPERT_SHAPES},
        buffer_bytes=64,
    )
    _check_equals_cold_start(fleet, make, _EXPERT_SHAPES)
    assert {b.producer for b in fleet.engine._buckets} == {1}
    for b in fleet.engine._buckets:
        for _, box in b.pieces:  # its own experts' rows (maybe in segments)
            assert 2 * ep_rank <= box[0][0] < box[0][1] <= 2 * ep_rank + 2


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("buffer_bytes", [16, 1 << 20])
def test_expert_tp_two_boxes_per_name(monkeypatch, tp_rank, buffer_bytes):
    """Gate and up rows are two boxes of one name, landing strided; with
    16 B budgets each row of each piece is its own pull."""
    make = _experts_model(0, 1, tp_rank, 2)
    model = _cold_start(make, _checkpoint(_EXPERT_SHAPES, seed=0))
    fleet = _Fleet(monkeypatch, model, _EXPERT_SHAPES, buffer_bytes=buffer_bytes)
    _check_equals_cold_start(fleet, make, _EXPERT_SHAPES)
    gate_up = [
        box
        for b in fleet.engine._buckets
        for name, box in b.pieces
        if name == "experts.gate_up_proj"
    ]
    if buffer_bytes == 16:
        assert len(gate_up) > 2  # segments of whole rows
        assert all(len(b.pieces) == 1 for b in fleet.engine._buckets)
    else:
        assert len(gate_up) == 2


def test_two_consumers_release_a_unit_only_when_both_are_done(monkeypatch):
    """A unit stays gathered until every live worker has freed its pulls."""
    make = _experts_model(0, 1, 0, 1)
    fleets: list[_Fleet] = []
    for consumer in (0, 1):
        model = _cold_start(make, _checkpoint(_EXPERT_SHAPES, seed=0))
        fleet = _Fleet(
            monkeypatch,
            model,
            _EXPERT_SHAPES,
            num_producers=1,
            consumer_id=consumer,
            num_consumers=2,
            buffer_bytes=1 << 20,
            servers=fleets[0].servers if fleets else None,
        )
        fleets.append(fleet)
    for fleet in fleets:
        fleet.engine.start_weight_update()
    (server,) = fleets[0].servers
    server.begin_sync([0, 1], 0)
    weights = _checkpoint(_EXPERT_SHAPES, seed=1)
    server._publish(0, dict(weights))
    fleets[0].engine.update_weights({"sync_idx": 0})
    assert server._resident, "released before consumer 1 pulled"
    fleets[1].engine.update_weights({"sync_idx": 0})
    assert server.end_sync() == [0]
    for fleet in fleets:
        fleet.engine.finish_weight_update()
        expected = _state(_cold_start(make, weights))
        for name, s in _state(fleet.engine.model).items():
            assert torch.equal(s[3], expected[name][3]), name


# ------------------------------------------------------------ buckets


def test_buckets_never_cross_a_unit_and_respect_the_budget(monkeypatch):
    model = _cold_start(_toy_model, _checkpoint(_TOY_SHAPES, seed=0))
    fleet = _Fleet(monkeypatch, model, _TOY_SHAPES, buffer_bytes=48)
    fleet.engine.start_weight_update()
    for b in fleet.engine._buckets:
        assert {fleet.units[n] for n, _ in b.pieces} == {b.unit}
        if len(b.pieces) > 1:
            assert b.nbytes <= 48
    seqs: dict[int, list[int]] = {}
    for b in fleet.engine._buckets:
        seqs.setdefault(b.producer, []).append(b.seq)
    assert all(s == list(range(len(s))) for s in seqs.values())


def test_pipeline_stages_pull_side_by_side(monkeypatch):
    """Two stages are independent components: pulls interleave them, each in
    unit order, instead of draining stage 0 first."""
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear

    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    make = _linear_model()
    model = _cold_start(make, _checkpoint(_LINEAR_SHAPES, seed=0))
    names = list(_LINEAR_SHAPES)
    owners = {n: [0] if i < 3 else [1] for i, n in enumerate(names)}
    fleet = _Fleet(monkeypatch, model, _LINEAR_SHAPES, owners=owners, buffer_bytes=128)
    fleet.engine.start_weight_update()
    producers = [b.producer for b in fleet.engine._buckets]
    assert producers[:2] == [0, 1], producers
    for p in (0, 1):
        units = [b.unit for b in fleet.engine._buckets if b.producer == p]
        assert units == sorted(units)


def test_plan_failure_leaves_a_note_and_retries(monkeypatch):
    model = _cold_start(_toy_model, _checkpoint(_TOY_SHAPES, seed=0))
    fleet = _Fleet(monkeypatch, model, _TOY_SHAPES)

    def boom(_):
        raise RuntimeError("producers unreachable")

    hooks = fleet.engine._hooks
    assert hooks is not None
    fleet.engine._hooks = dataclasses.replace(hooks, register=boom)
    with pytest.raises(RuntimeError, match="unreachable") as info:
        fleet.engine.start_weight_update()
    assert "mid-reload" in "".join(info.value.__notes__)
    assert not fleet.engine._planned
    fleet.engine._hooks = hooks
    _check_equals_cold_start(fleet, _toy_model, _TOY_SHAPES)


def test_receive_without_a_plan_raises(monkeypatch):
    model = _cold_start(_toy_model, _checkpoint(_TOY_SHAPES, seed=0))
    fleet = _Fleet(monkeypatch, model, _TOY_SHAPES)
    with pytest.raises(RuntimeError, match="start_weight_update"):
        fleet.engine.receive_weights(ShardedRDTWeightTransferUpdateInfo(sync_idx=0))


def test_init_refuses_layerwise(monkeypatch):
    model = _cold_start(_toy_model, _checkpoint(_TOY_SHAPES, seed=0))
    monkeypatch.setenv("VLLM_RELOAD_MODE", "layerwise")
    with pytest.raises(RuntimeError, match="sharded_rdt.*VLLM_RELOAD_MODE=direct"):
        _Fleet(monkeypatch, model, _TOY_SHAPES)


def test_init_refuses_mismatched_metadata(monkeypatch):
    model = _cold_start(_toy_model, _checkpoint(_TOY_SHAPES, seed=0))
    fleet = _Fleet(monkeypatch, model, _TOY_SHAPES)
    info = ShardedRDTWeightTransferInitInfo(
        trainer_actor_names=["p0"],
        names=["a", "b"],
        dtype_names=["bfloat16"],
        shapes=[[1]],
    )
    with pytest.raises(ValueError, match="2 names, 1 dtypes"):
        fleet.engine.init_transfer_engine(info)


def test_flat_layout_puts_each_piece_on_16_bytes():
    boxes = [((0, 3),), ((0, 2), (0, 5)), ((1, 2), (0, 1), (0, 4))]
    dtypes = [BF16, torch.float32, torch.uint8]
    layout = flat_layout(boxes, dtypes)
    assert [(off, n) for off, n, _, _ in layout] == [(0, 6), (16, 40), (64, 4)]
    pieces = [
        torch.randn(3).to(BF16),
        torch.randn(2, 5),
        torch.arange(4, dtype=torch.uint8).view(1, 1, 4),
    ]
    nbytes = sum(piece_nbytes(b, d) for b, d in zip(boxes, dtypes))
    flat = torch.zeros(nbytes, dtype=torch.uint8)
    torch._foreach_copy_([flat_view(flat, p) for p in layout], pieces)
    for p, want in zip(layout, pieces):
        got = flat_view(flat, p)
        assert got.dtype == want.dtype and torch.equal(got, want)


# ------------------------------------------------------------ common


@pytest.mark.parametrize(
    "installed,ok", [("2.56.0", True), ("2.60.0", True), ("2.55.1", False)]
)
def test_ray_version_floor(monkeypatch, installed, ok):
    import importlib.metadata

    from vllm.distributed.weight_transfer import sharded_rdt_common

    monkeypatch.setattr(importlib.metadata, "version", lambda pkg: installed)
    if ok:
        sharded_rdt_common.check_ray_rdt_version()
    else:
        with pytest.raises(ValueError, match="requires Ray"):
            sharded_rdt_common.check_ray_rdt_version()


def test_gather_units_cut_by_bytes():
    assert gather_units([4, 4, 4, 4], 8) == [0, 0, 1, 1]
    assert gather_units([4, 20, 4], 8) == [0, 1, 2]  # an oversized name alone
    assert gather_units([], 8) == []


def test_arena_holds_every_consumers_window_plus_slack():
    """Slots of `buffer_bytes`, however large the largest tensor."""
    assert (
        arena_bytes(buffer_bytes=100, num_buffers=2, num_consumers=3)
        == (2 * 3 + 1) * 116
    )
    assert arena_bytes(buffer_bytes=100, num_buffers=1, num_consumers=1) == 2 * 116


class TestRouter:
    def test_identity_when_fleets_match(self):
        r = RdtRouter(4, 4)
        assert [r.producer_for(c, "w", 0) for c in range(4)] == [0, 1, 2, 3]

    def test_fan_in_shares_a_producer(self):
        r = RdtRouter(2, 4)
        assert [r.producer_for(c, "w", 0) for c in range(4)] == [0, 0, 1, 1]

    def test_a_consumer_spreads_its_units_over_its_block(self):
        r = RdtRouter(4, 2)
        assert {r.producer_for(0, "w", u) for u in range(4)} == {0, 1}

    def test_owner_sets_route_pipeline_stages_and_experts(self):
        r = RdtRouter(4, 2, [[0, 1], [2, 3], [3]], [0, 1, 2], ["s0", "s1", "e"])
        assert r.producer_for(1, "s0", 0) in (0, 1)
        assert r.producer_for(0, "s1", 0) in (2, 3)
        assert r.producer_for(0, "e", 0) == r.producer_for(1, "e", 5) == 3

    def test_every_deployment_routes_like_the_first(self):
        r = RdtRouter(2, 4, workers_per_replica=2)
        assert [r.producer_for(c, "w", 0) for c in range(4)] == [0, 1, 0, 1]

    def test_uneven_deployments_are_refused(self):
        with pytest.raises(ValueError, match="uniform"):
            RdtRouter(2, 3, workers_per_replica=2)

    def test_consumers_of_bounds_fan_in(self):
        r = RdtRouter(2, 2, [[0, 1], [1]], [0, 1], ["w", "e"])
        assert r.consumers_of(0) == [0]
        assert r.consumers_of(1) == [0, 1]

    def test_components_couple_co_owners_only(self):
        assert RdtRouter(
            4, 1, [[0, 1], [2], [3]], [0, 1, 2], ["a", "b", "c"]
        ).components() == [0, 0, 2, 3]
        assert len(set(RdtRouter(3, 1).components())) == 1

    @pytest.mark.parametrize(
        "owner_sets,classes,match",
        [
            ([[]], [0], "empty"),
            ([[0, 5]], [0], "out of range"),
            ([[0]], [3], "class 3"),
        ],
    )
    def test_validate(self, owner_sets, classes, match):
        with pytest.raises(ValueError, match=match):
            RdtRouter(2, 1, owner_sets, classes, ["w"]).validate()


def test_consumer_id_offsets_independent_deployments(monkeypatch):
    model = _cold_start(_toy_model, _checkpoint(_TOY_SHAPES, seed=0))
    fleet = _Fleet(monkeypatch, model, _TOY_SHAPES, num_consumers=4, consumer_id=1)
    info = ShardedRDTWeightTransferInitInfo(
        trainer_actor_names=["p0", "p1"],
        num_consumers=4,
        replica_rank=1,
        num_replicas=2,
        **_metas_info(_TOY_SHAPES),
    )
    fleet.engine.parallel_config.world_size = 2
    fleet.engine.init_transfer_engine(info)
    assert fleet.engine._consumer_id == 2 + 1
