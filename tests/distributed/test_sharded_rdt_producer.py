# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sharded-RDT producer server, on CPU; frees come from the test."""

import contextlib
import gc
import os
import sys
import weakref
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.distributed.weight_transfer.sharded_rdt_trainer as trainer_mod
from tests.distributed.sharded_rdt_utils import make_server, patch_for_cpu
from vllm.distributed.nixl_utils import alias_nixl_for_ray
from vllm.distributed.weight_transfer.sharded_rdt_common import (
    flat_view,
    register_nixl_memory,
)
from vllm.distributed.weight_transfer.sharded_rdt_trainer import (
    ShardedRDTTrainerInitInfo,
    ShardedRDTTrainerWeightTransferEngine,
    _RDTProducerServer,
)


@pytest.fixture
def ray_nixl_transport(monkeypatch):
    import ray
    from ray.experimental.rdt import util
    from ray.experimental.rdt.nixl_tensor_transport import NixlTensorTransport

    from vllm.distributed import nixl_utils
    from vllm.platforms import current_platform

    transport = NixlTensorTransport()
    package = ModuleType("nixl_rocm")
    api = ModuleType("nixl_rocm._api")
    api.__dict__.update(nixl_agent=Mock(), nixl_agent_config=Mock())
    package.__dict__["_api"] = api
    modules = {package.__name__: package, api.__name__: api}
    imported_rcache_limits = []

    def import_nixl(name):
        imported_rcache_limits.append(os.environ.get("UCX_RCACHE_MAX_UNRELEASED"))
        monkeypatch.setitem(sys.modules, name, modules[name])
        return modules[name]

    nixl_names = ("nixl", "nixl._api", *modules)
    for name in nixl_names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delenv("UCX_RCACHE_MAX_UNRELEASED", raising=False)
    importer = Mock(side_effect=import_nixl)
    monkeypatch.setattr(
        nixl_utils, "importlib", SimpleNamespace(import_module=importer)
    )
    monkeypatch.setattr(nixl_utils, "is_nixl_available", lambda: True)
    monkeypatch.setattr(util, "transport_managers", {"NIXL": transport})
    monkeypatch.setattr(current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(
        ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_actor_id=lambda: "rdt-actor"),
    )
    try:
        yield transport, api, importer, imported_rcache_limits
    finally:
        # The helper inserts aliases directly; remove them before monkeypatch
        # restores any modules and environment values present before this test.
        for name in nixl_names:
            sys.modules.pop(name, None)
        os.environ.pop("UCX_RCACHE_MAX_UNRELEASED", None)


def test_ray_registers_memory_through_rocm_alias(ray_nixl_transport):
    """Ray's registration must reach the ROCm agent through the alias."""
    _, api, importer, _ = ray_nixl_transport
    register_nixl_memory(torch.empty(4, dtype=torch.uint8))
    api.nixl_agent.return_value.register_memory.assert_called()
    assert sys.modules["nixl"] is sys.modules["nixl_rocm"]
    assert sys.modules["nixl._api"] is api

    importer.reset_mock()
    alias_nixl_for_ray()
    importer.assert_not_called()


@pytest.mark.parametrize("rcache_limit", [None, "4096"])
def test_ray_nixl_sets_ucx_limit_before_import(
    monkeypatch, ray_nixl_transport, rcache_limit
):
    """UCX must see the default or the user's override during the first import."""
    _, _, _, imported_rcache_limits = ray_nixl_transport
    if rcache_limit is not None:
        monkeypatch.setenv("UCX_RCACHE_MAX_UNRELEASED", rcache_limit)
    alias_nixl_for_ray()
    assert imported_rcache_limits == [rcache_limit or "1024"] * 2


@pytest.mark.parametrize("existing_api", [False, True])
def test_ray_nixl_preserves_non_rocm_or_existing_imports(
    monkeypatch, ray_nixl_transport, existing_api
):
    from vllm.platforms import current_platform

    _, _, importer, _ = ray_nixl_transport
    if existing_api:
        api = ModuleType("nixl._api")
        monkeypatch.setitem(sys.modules, "nixl._api", api)
    else:
        monkeypatch.setattr(current_platform, "is_rocm", lambda: False)
    alias_nixl_for_ray()
    importer.assert_not_called()
    assert "UCX_RCACHE_MAX_UNRELEASED" not in os.environ
    if existing_api:
        assert sys.modules["nixl._api"] is api
    else:
        assert "nixl._api" not in sys.modules


def test_ray_nixl_alias_reports_missing_rocm_package(monkeypatch, ray_nixl_transport):
    from vllm.distributed import nixl_utils

    transport, api, importer, _ = ray_nixl_transport
    monkeypatch.setattr(nixl_utils, "is_nixl_available", lambda: False)
    with pytest.raises(ImportError, match="nixl_rocm"):
        register_nixl_memory(torch.empty(1))
    assert transport._nixl_agent is None
    api.nixl_agent.assert_not_called()
    importer.assert_not_called()
    assert "nixl" not in sys.modules
    assert "nixl._api" not in sys.modules


# ------------------------------------------------------------ the server

N, M = "model.layers.0.w", "model.layers.0.b"
E = "model.layers.1.experts.w"
_SERVED = {N: ("bfloat16", (8, 4)), M: ("float32", (4,)), E: ("bfloat16", (2, 6))}
_UNITS = {N: 0, M: 0, E: 1}
WHOLE_N = [(N, ((0, 8), (0, 4)))]  # 64 B


@pytest.fixture
def server(monkeypatch):
    patch_for_cpu(monkeypatch)
    monkeypatch.setattr(trainer_mod, "_STALL_POLL_S", 0.05)

    def make(cls=None, arena_bytes=1 << 20, **kw):
        kw = {"bucket_bytes": arena_bytes, "stall_timeout_s": 30} | kw
        return make_server(
            cls, served=_SERVED, units=_UNITS, arena_bytes=arena_bytes, **kw
        )

    return make


def _weights(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn(shape, generator=g).to(getattr(torch, dt))
        for name, (dt, shape) in _SERVED.items()
    }


def _publish(server, weights, *units):
    for unit in units:
        server._publish(unit, {n: weights[n] for n, u in _UNITS.items() if u == unit})


def _call(fn, *args, timeout=0.3):
    """`fn` on a thread: its future, still pending after `timeout` if it blocks."""
    pool = ThreadPoolExecutor(1)
    future = pool.submit(fn, *args)
    with contextlib.suppress(FuturesTimeout):
        future.result(timeout=timeout)
    pool.shutdown(wait=False)
    return future


def _boxes(server, flat, consumer=0, seq=0):
    return [flat_view(flat, p) for p in server._buckets[consumer][seq].layout]


def _bytes(t):
    return t.contiguous().view(-1).view(torch.uint8)


@pytest.mark.parametrize("moved", [False, True])
def test_cached_sources_follow_the_published_memory(server, monkeypatch, moved):
    """Each sync exports anew (new ref counter, event): the same allocation
    keeps a bucket's cached sources, a moved one rebuilds them."""
    bases: dict[str, torch.Tensor] = {}
    monkeypatch.setattr(
        trainer_mod,
        "rebuild_cuda_tensor",
        lambda *a: bases.setdefault(a[7], torch.zeros(80, dtype=torch.uint8)),
    )
    views = {N: (1, "bfloat16", [8, 4], [4, 1], 0), M: (1, "float32", [4], [1], 16)}
    s = server()
    s.register_consumer(0, [[(N, ((2, 6), (1, 3))), (M, ((0, 4),))]])
    cached = []
    for sync_idx in (0, 1):
        s.begin_sync([0], sync_idx)
        alloc = f"a{sync_idx}" if moved else "a"
        export = (None,) * 6 + (0, alloc, 80, 0, False, sync_idx, 0, sync_idx)
        s.publish_unit(0, ({1: export}, views, None))
        base = bases[alloc]
        base.copy_(torch.randn(20).view(torch.uint8))
        (flat,) = s.bucket(0, 0, sync_idx)
        assert flat.untyped_storage().data_ptr() == s._arena.data_ptr()
        a, b = _boxes(s, flat)
        n = base[:64].view(torch.bfloat16).view(8, 4)
        assert torch.equal(_bytes(a), _bytes(n[2:6, 1:3]))
        assert torch.equal(_bytes(b), base[64:])
        cached.append(s._sources[0][0])
        s.free()
    assert (cached[0] is cached[1]) != moved


def test_a_pull_waits_for_its_unit(server):
    s = server()
    s.register_consumer(0, [[(E, ((0, 2), (0, 6)))]])
    s.begin_sync([0], 0)
    w = _weights()
    future = _call(s.bucket, 0, 0, 0)
    _publish(s, w, 0)
    assert not future.done()
    _publish(s, w, 1)
    (got,) = _boxes(s, future.result(timeout=5)[0])
    assert torch.equal(got, w[E])


def test_a_pull_for_the_next_sync_waits_for_it(server):
    """A worker may pull for sync 1 before this rank released sync 0."""
    s = server()
    s.register_consumer(0, [[(E, ((0, 2), (0, 6)))], WHOLE_N])
    s.begin_sync([0], 0)
    _publish(s, _weights(0), 0, 1)
    s.bucket(0, 0, 0)
    future = _call(s.bucket, 0, 0, 1)
    s.bucket(0, 1, 0)
    s.free()
    assert s.end_sync() == [1, 0]
    assert not future.done(), "served sync 1 out of sync 0's gather"
    s.begin_sync([0], 1)
    new = _weights(1)
    _publish(s, new, 1)
    (got,) = _boxes(s, future.result(timeout=5)[0])
    assert torch.equal(got, new[E])


def test_a_pull_left_behind_by_a_new_sync_fails(server):
    s = server()
    s.register_consumer(0, [WHOLE_N])
    s.begin_sync([0], 0)
    future = _call(s.bucket, 0, 0, 0)
    s.begin_sync([0], 1)
    with pytest.raises(ValueError, match="sync 0, now 1"):
        future.result(timeout=5)


def test_a_failed_pack_gives_its_block_back(server, monkeypatch):
    s = server()
    s.register_consumer(0, [WHOLE_N])
    s.begin_sync([0], 0)
    _publish(s, _weights(), 0, 1)

    def boom(*a):
        raise RuntimeError("pack failed")

    monkeypatch.setattr(trainer_mod, "_sources", boom)
    with pytest.raises(RuntimeError, match="pack failed"):
        s.bucket(0, 0, 0)
    assert s._blocks == {}


@pytest.mark.parametrize(
    "consumer,seq,sync_idx,match",
    [(0, 0, 2, "sync 2"), (0, 1, 3, "never registered"), (1, 0, 3, "never registered")],
)
def test_bad_pulls_are_refused(server, consumer, seq, sync_idx, match):
    s = server()
    s.register_consumer(0, [WHOLE_N])
    s.begin_sync([0], 3)
    with pytest.raises(ValueError, match=match):
        s.bucket(consumer, seq, sync_idx)


def test_the_arena_admits_a_bucket_only_when_a_block_fits(server):
    """Served, unfreed buckets hold their blocks; a full arena makes a pull
    wait, and a freed block is reused."""
    s = server(arena_bytes=128)
    s.register_consumer(0, [WHOLE_N] * 3)
    s.begin_sync([0], 0)
    _publish(s, _weights(), 0, 1)
    s.bucket(0, 0, 0)
    s.bucket(0, 1, 0)
    future = _call(s.bucket, 0, 2, 0)
    assert not future.done()
    s.free(1)
    (flat,) = future.result(timeout=5)
    assert flat.data_ptr() == s._arena.data_ptr()
    s.free()
    assert s._blocks == {}


@pytest.mark.parametrize(
    "bucket,match",
    [
        ([("other", ((0, 1),))], "not served"),
        ([(N, ((0, 9), (0, 4)))], "outside"),
        ([(N, ((0, 8),))], "outside"),
        ([(N, ((0, 1), (0, 4))), (E, ((0, 1), (0, 6)))], "spans units"),
        (WHOLE_N, "larger buffer_bytes"),
    ],
)
def test_bad_buckets_are_refused(server, bucket, match):
    with pytest.raises(ValueError, match=match):
        server(bucket_bytes=48).register_consumer(0, [bucket])


def test_publish_checks_the_metadata(server):
    with pytest.raises(ValueError, match="metadata"):
        server()._publish(0, {N: torch.zeros(8, 4)})


# ------------------------------------------------------------ release


def test_a_unit_waits_for_every_live_consumer(server):
    s = server()
    for c in (0, 1):
        s.register_consumer(c, [WHOLE_N])
    s.begin_sync([0, 1], 0)
    _publish(s, _weights(), 0)
    s.bucket(0, 0, 0)
    s.bucket(1, 0, 0)
    future = _call(s.end_sync)
    s.free(1)
    assert 0 in s._resident and not future.done()
    s.free()
    assert future.result(timeout=5) == [0]


def test_an_unneeded_unit_is_released_at_publish(server):
    s = server()
    s.register_consumer(0, [WHOLE_N])
    s.begin_sync([0], 0)
    _publish(s, _weights(), 1)
    assert s.wait_freed() == [1]


def test_an_unregistered_live_consumer_holds_every_unit(server):
    s = server()
    s.register_consumer(0, [])
    s.begin_sync([0, 1], 0)
    _publish(s, _weights(), 1)
    assert 1 in s._resident
    s.register_consumer(1, [])
    assert s.wait_freed() == [1]


def test_a_degraded_sync_ignores_the_dead(server):
    s = server()
    for c in (0, 1):
        s.register_consumer(c, [WHOLE_N])
    s.begin_sync([1], 0)
    _publish(s, _weights(), 0)
    s.bucket(1, 0, 0)
    s.free()
    assert s.end_sync() == [0]


def test_a_free_from_the_previous_sync_does_not_count(server):
    s = server()
    s.register_consumer(0, [WHOLE_N])
    s.begin_sync([0], 0)
    _publish(s, _weights(), 0)
    s.bucket(0, 0, 0)
    s.begin_sync([0], 1)
    _publish(s, _weights(1), 0)
    s.free()
    assert 0 in s._resident and s._blocks == {}


def test_a_gather_error_releases_every_wait(server):
    s = server()
    s.register_consumer(0, [WHOLE_N])
    s.begin_sync([0], 0)
    waits = [_call(s.wait_freed), _call(s.bucket, 0, 0, 0)]
    s.set_gather_error("gather died")
    for w in waits:
        with pytest.raises(RuntimeError, match="gather died"):
            w.result(timeout=5)
    s.set_gather_error("sender failed", sync_idx=1)  # before this rank begins 1
    s.begin_sync([0], 1)
    with pytest.raises(RuntimeError, match="sender failed"):
        s.wait_freed()
    s.begin_sync([0], 2)  # a new sync clears it
    _publish(s, _weights(), 1)
    assert s.end_sync() == [1]


def test_a_dead_consumer_trips_the_watchdog(server):
    s = server(stall_timeout_s=0.2)
    s.register_consumer(0, [WHOLE_N])
    s.begin_sync([0], 0)
    _publish(s, _weights(), 0)
    with pytest.raises(RuntimeError, match="stall"):
        s.end_sync()


# ------------------------------------------------------------ frees (Ray's)


def test_a_bucket_is_freed_when_ray_drops_it(server):
    """The free is Ray dropping its last reference to the served block."""
    s = server(cls=_RDTProducerServer)
    s.register_consumer(0, [WHOLE_N])
    s.begin_sync([0], 0)
    _publish(s, _weights(), 0)
    ray_store = s.bucket(0, 0, 0)
    block = weakref.ref(ray_store[0])
    gc.collect()
    assert s._blocks and s._freed_pending == []
    del ray_store
    assert block() is None, "the producer kept a reference"
    assert s.wait_freed() == [0] and s._blocks == {}


def test_buckets_freed_as_soon_as_served_all_count(server):
    """Two consumers, two pulls each, a full arena, every bucket freed the
    moment it is served: no free is missed (polling Ray's store did)."""
    s = server(cls=_RDTProducerServer, arena_bytes=3 * 16)
    buckets = [[(N, ((r, r + 2), (0, 4)))] for r in range(0, 8, 2)] + [
        [(M, ((0, 4),))],
        [(E, ((0, 1), (0, 6)))],
        [(E, ((1, 2), (0, 6)))],
    ]
    for c in (0, 1):
        s.register_consumer(c, buckets)

    def pull(c, seqs, sync_idx):
        for seq in seqs:
            s.bucket(c, seq, sync_idx)  # Ray stores and frees at once

    for sync_idx in range(3):
        s.begin_sync([0, 1], sync_idx)
        _publish(s, _weights(sync_idx), 0, 1)
        pulls = [
            _call(pull, c, range(half, len(buckets), 2), sync_idx, timeout=5)
            for c in (0, 1)
            for half in (0, 1)
        ]
        for f in pulls:
            f.result(timeout=10)
        assert sorted(_call(s.end_sync, timeout=10).result(timeout=10)) == [0, 1]
        assert s._blocks == {}


# ------------------------------------------------------------ packing


def test_strided_boxes_of_consecutive_experts_pack_as_one_copy():
    w = torch.arange(4 * 3 * 8, dtype=torch.float32).view(4, 3, 8)
    boxes = [((e, e + 1), (0, 3), (2, 6)) for e in range(4)]
    layout = trainer_mod.flat_layout(boxes, [torch.float32] * 4)
    src = trainer_mod._sources(0, [w[e, :, 2:6] for e in range(4)], layout, 4 * 48)
    assert src.typed_idx == ((0, 4),) and torch.equal(src.typed_srcs[0], w[:, :, 2:6])


def test_aligned_contiguous_boxes_copy_as_words():
    w = torch.zeros(8, 8, dtype=torch.bfloat16)
    views = [w[0:4], w[4:8].view(-1)[:3].view(1, 3)]  # 64 B; 6 B
    boxes = [((0, 4), (0, 8)), ((4, 5), (0, 3))]
    layout = trainer_mod.flat_layout(boxes, [torch.bfloat16] * 2)
    src = trainer_mod._sources(0, views, layout, 80)
    # odd chunks: words 0:8 (piece 0), bytes 64:70 (piece 1)
    assert src.word_splits == (0, 8, 2) and src.word_srcs[0].dtype == torch.int64
    assert src.byte_splits == (64, 6, 10) and src.byte_srcs[0].dtype == torch.uint8


def test_small_tensors_pack_into_an_export_slot_as_publish_rebuilds_them(monkeypatch):
    monkeypatch.setattr(trainer_mod, "reduce_tensor", lambda base: (None, ()))
    eng = ShardedRDTTrainerWeightTransferEngine(
        client=Mock(),
        source=None,
        init_info=ShardedRDTTrainerInitInfo(rank=0, num_consumers=1),
    )
    eng._export_ring, eng._export_ring_args = [None], [None]
    held = [
        ("bf16.2d", torch.randn(12, 7).bfloat16()),
        ("fp32.1d", torch.randn(9)),
        ("bf16.transposed", torch.randn(6, 11).bfloat16().t()),
        ("bf16.sliced", torch.randn(8, 32).bfloat16()[:, 3:19]),
    ]
    storages, views, _ = eng._pack_for_export(held, 0)
    (slot,) = eng._export_ring
    assert len(storages) == 1
    for name, t in held:
        _, dt, shape, stride, off = views[name]
        got = torch.as_strided(slot.view(getattr(torch, dt)), shape, stride, off)
        assert got.dtype is t.dtype and torch.equal(got, t), name
