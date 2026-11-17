# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sharded-RDT producer server (`_RDTProducerServer`), on CPU.

The server serves registered buckets and releases a gathered unit once every
live consumer has freed every bucket reading from it. Releasing early lets
the trainer overwrite memory a pull still reads; releasing never wedges the
gather loop. Frees come from the test here; on GPU, from Ray's object store.
"""

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


class _Server(_RDTProducerServer):
    """Frees come from the test (`free`), not from Ray's object store."""

    def _track_free(self, flat, *key):
        self.outstanding.append(key)


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(
        torch.accelerator, "current_device_index", lambda *a, **kw: 0, raising=False
    )
    monkeypatch.setattr(gc, "freeze", lambda: None)
    monkeypatch.setattr(trainer_mod, "_STALL_POLL_S", 0.05)
    built = []

    def make(cls=_Server, **kw):
        kw.setdefault("served", _SERVED)
        kw.setdefault("units", _UNITS)
        kw.setdefault("arena_bytes", 1 << 20)
        kw.setdefault("stall_timeout_s", 30)
        s = cls(**kw)
        s.outstanding = []
        built.append(s)
        return s

    yield make
    for s in built:
        s.shutdown()


def _weights(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn(shape, generator=g).to(getattr(torch, dt))
        for name, (dt, shape) in _SERVED.items()
    }


def _publish(server, weights, *units):
    for unit in units:
        server._publish(unit, {n: weights[n] for n, u in _UNITS.items() if u == unit})


def free(server, n=None):
    """Ray freed the oldest `n` served buckets (all by default)."""
    for key in server.outstanding[: n if n is not None else len(server.outstanding)]:
        server._freed_bucket(*key)
    del server.outstanding[: n if n is not None else len(server.outstanding)]


def _call(fn, *args, timeout=0.3):
    """Run `fn` on a thread; its future, still pending after `timeout` if it
    blocks."""
    pool = ThreadPoolExecutor(1)
    future = pool.submit(fn, *args)
    with contextlib.suppress(FuturesTimeout):
        future.result(timeout=timeout)
    pool.shutdown(wait=False)
    return future


def _boxes(server, flat, consumer=0, seq=0):
    return [flat_view(flat, p) for p in server._buckets[consumer][seq].layout]


class TestServe:
    def test_a_bucket_packs_its_boxes_into_the_arena(self, server):
        s = server()
        s.register_consumer(0, [[(N, [[2, 6], [1, 3]]), (M, [[0, 4]])]])
        s.begin_sync([0], 0)
        w = _weights()
        _publish(s, w, 0)
        (flat,) = s.bucket(0, 0, 0)
        assert flat.dtype == torch.uint8 and flat.numel() == 32
        assert flat.untyped_storage().data_ptr() == s._arena.data_ptr()
        a, b = _boxes(s, flat)
        assert torch.equal(a, w[N][2:6, 1:3]) and torch.equal(b, w[M])

    def test_new_unit_tensors_rebuild_the_cached_sources(self, server):
        """A bucket's source boxes are cached per publish of its unit: new
        tensors (a changed storage) are served, not the cached boxes."""
        s = server()
        s.register_consumer(0, [[(N, [[2, 6], [1, 3]]), (M, [[0, 4]])]])
        for sync_idx, seed in enumerate((0, 1)):
            s.begin_sync([0], sync_idx)
            w = _weights(seed)
            _publish(s, w, 0)
            (flat,) = s.bucket(0, 0, sync_idx)
            a, b = _boxes(s, flat)
            assert torch.equal(a, w[N][2:6, 1:3]) and torch.equal(b, w[M])
            free(s)

    def test_the_same_unit_tensors_reuse_the_cached_sources(self, server):
        """Republishing the same tensors (an identical CUDA-IPC publish)
        keeps the cached boxes, which read the tensors' current bytes."""
        s = server()
        s.register_consumer(0, [[(N, [[2, 6], [1, 3]]), (M, [[0, 4]])]])
        w = _weights()
        unit0 = {n: w[n] for n, u in _UNITS.items() if u == 0}
        for sync_idx in (0, 1):
            s.begin_sync([0], sync_idx)
            for t in unit0.values():
                t.add_(1)
            s._publish(0, unit0)
            (flat,) = s.bucket(0, 0, sync_idx)
            if sync_idx == 0:
                cached = s._sources[0, 0]
            a, b = _boxes(s, flat)
            assert torch.equal(a, w[N][2:6, 1:3]) and torch.equal(b, w[M])
            free(s)
        assert s._sources[0, 0] is cached

    def test_a_pull_waits_for_its_unit(self, server):
        s = server()
        s.register_consumer(0, [[(E, [[0, 2], [0, 6]])]])
        s.begin_sync([0], 0)
        w = _weights()
        future = _call(s.bucket, 0, 0, 0)
        assert not future.done()
        _publish(s, w, 0)
        assert not future.done()
        _publish(s, w, 1)
        (got,) = _boxes(s, future.result(timeout=5)[0])
        assert torch.equal(got, w[E])

    def test_a_pull_for_the_next_sync_waits_for_it(self, server):
        """A worker can finish a sync and pull for the next before this rank
        has released (or even gathered) it; it must not get stale bytes."""
        s = server()
        s.register_consumer(0, [[(E, [[0, 2], [0, 6]])], [(N, [[0, 8], [0, 4]])]])
        s.begin_sync([0], 0)
        old = _weights(0)
        _publish(s, old, 0, 1)
        s.bucket(0, 0, 0)
        future = _call(s.bucket, 0, 0, 1)
        assert not future.done(), "served sync 1 out of sync 0's gather"
        s.bucket(0, 1, 0)
        free(s)
        assert s.end_sync() == [1, 0]
        s.begin_sync([0], 1)
        new = _weights(1)
        _publish(s, new, 1)
        (got,) = _boxes(s, future.result(timeout=5)[0])
        assert torch.equal(got, new[E])

    def test_a_pull_for_a_past_sync_is_refused(self, server):
        s = server()
        s.register_consumer(0, [[(E, [[0, 2], [0, 6]])]])
        s.begin_sync([0], 3)
        with pytest.raises(ValueError, match="sync 2"):
            s.bucket(0, 0, 2)

    def test_an_unregistered_bucket_is_refused(self, server):
        s = server()
        s.register_consumer(0, [])
        with pytest.raises(ValueError, match="never registered"):
            s.bucket(0, 0, 0)
        with pytest.raises(ValueError, match="never registered"):
            s.bucket(1, 0, 0)

    def test_the_arena_admits_a_bucket_only_when_a_block_fits(self, server):
        """Served, unfreed buckets hold their blocks; a full arena slows
        pulls down instead of failing in Ray. Freed blocks are reused, first
        fit."""
        s = server(arena_bytes=128)
        whole = [(N, [[0, 8], [0, 4]])]  # 64 B
        s.register_consumer(0, [whole, whole, whole])
        s.begin_sync([0], 0)
        _publish(s, _weights(), 0, 1)
        s.bucket(0, 0, 0)
        s.bucket(0, 1, 0)
        assert sorted(s._blocks) == [0, 64]
        future = _call(s.bucket, 0, 2, 0)
        assert not future.done()
        free(s, 1)
        (flat,) = future.result(timeout=5)
        assert flat.data_ptr() == s._arena.data_ptr()
        free(s)
        assert s._blocks == {}


class TestRegister:
    @pytest.mark.parametrize(
        "bucket,match",
        [
            ([("other", [[0, 1]])], "not served"),
            ([(N, [[0, 9], [0, 4]])], "outside"),
            ([(N, [[0, 8]])], "outside"),
            ([(N, [[0, 1], [0, 4]]), (E, [[0, 1], [0, 6]])], "spans units"),
        ],
    )
    def test_bad_buckets_are_refused(self, server, bucket, match):
        with pytest.raises(ValueError, match=match):
            server().register_consumer(0, [bucket])

    def test_a_bucket_of_several_pieces_over_bucket_bytes_is_refused(self, server):
        with pytest.raises(ValueError, match="larger buffer_bytes"):
            server(bucket_bytes=16).register_consumer(
                0, [[(N, [[0, 4], [0, 4]]), (M, [[0, 4]])]]
            )

    def test_an_oversize_piece_is_served_outside_the_arena(self, server):
        """One piece over `bucket_bytes`: a temporary copy, not an arena
        block (nor the CUDA-IPC source, which NIXL cannot register), and its
        unit stays held until Ray frees it."""
        s = server(cls=_RDTProducerServer, arena_bytes=64, bucket_bytes=16)
        s.register_consumer(0, [[(N, [[0, 8], [0, 4]])]])
        s.begin_sync([0], 0)
        w = _weights()
        _publish(s, w, 0)
        ray_store = s.bucket(0, 0, 0)
        (flat,) = ray_store
        assert flat.data_ptr() != w[N].data_ptr() and flat.numel() == 64
        assert torch.equal(flat.view(torch.bfloat16).view(8, 4), w[N])
        assert s._blocks == {}  # no arena block
        del flat
        gc.collect()
        assert s._freed_pending == []  # the unit is held: Ray still has it
        del ray_store
        assert s.wait_freed() == [0]

    def test_an_oversize_strided_piece_is_served_through_a_temporary(self, server):
        s = server(cls=_RDTProducerServer, arena_bytes=64, bucket_bytes=8)
        s.register_consumer(0, [[(N, [[0, 8], [1, 3]])]])  # columns: strided
        s.begin_sync([0], 0)
        w = _weights()
        _publish(s, w, 0)
        ray_store = s.bucket(0, 0, 0)
        (flat,) = ray_store
        assert flat.data_ptr() != w[N].data_ptr() and s._blocks == {}
        assert torch.equal(flat.view(torch.bfloat16).view(8, 2), w[N][:, 1:3])
        del flat, ray_store
        assert s.wait_freed() == [0]

    def test_publish_checks_the_metadata(self, server):
        s = server()
        with pytest.raises(ValueError, match="metadata"):
            s._publish(0, {N: torch.zeros(8, 4)})


class TestRelease:
    def test_a_unit_waits_for_every_live_consumer(self, server):
        s = server()
        for c in (0, 1):
            s.register_consumer(c, [[(N, [[0, 8], [0, 4]])]])
        s.begin_sync([0, 1], 0)
        _publish(s, _weights(), 0)
        s.bucket(0, 0, 0)
        s.bucket(1, 0, 0)
        free(s, 1)
        assert 0 in s._unit_names
        free(s)
        assert 0 not in s._unit_names
        assert s.wait_freed() == [0]

    def test_serving_is_not_freeing(self, server):
        """Until Ray frees a served bucket the consumer may still read its
        arena block, so a served unit stays gathered."""
        s = server()
        s.register_consumer(0, [[(N, [[0, 8], [0, 4]])]])
        s.begin_sync([0], 0)
        _publish(s, _weights(), 0)
        s.bucket(0, 0, 0)
        assert 0 in s._unit_names

    def test_an_unneeded_unit_is_released_at_publish(self, server):
        s = server()
        s.register_consumer(0, [[(N, [[0, 8], [0, 4]])]])
        s.begin_sync([0], 0)
        _publish(s, _weights(), 1)
        assert s.wait_freed() == [1]

    def test_an_unregistered_live_consumer_holds_every_unit(self, server):
        s = server()
        s.register_consumer(0, [])
        s.begin_sync([0, 1], 0)
        _publish(s, _weights(), 1)
        assert 1 in s._unit_names
        s.register_consumer(1, [])
        assert s.wait_freed() == [1]

    def test_a_degraded_sync_ignores_the_dead(self, server):
        s = server()
        for c in (0, 1):
            s.register_consumer(c, [[(N, [[0, 8], [0, 4]])]])
        s.begin_sync([1], 0)
        _publish(s, _weights(), 0)
        s.bucket(1, 0, 0)
        free(s)
        assert s.end_sync() == [0]

    def test_a_free_from_the_previous_sync_does_not_count(self, server):
        s = server()
        s.register_consumer(0, [[(N, [[0, 8], [0, 4]])]])
        s.begin_sync([0], 0)
        _publish(s, _weights(), 0)
        s.bucket(0, 0, 0)
        stale = s.outstanding.pop()
        s.begin_sync([0], 1)
        _publish(s, _weights(1), 0)
        s._freed_bucket(*stale)
        assert 0 in s._unit_names
        assert s._blocks == {}

    def test_end_sync_waits_for_every_unit(self, server):
        s = server()
        s.register_consumer(0, [[(N, [[0, 8], [0, 4]])]])
        s.begin_sync([0], 0)
        _publish(s, _weights(), 0)
        s.bucket(0, 0, 0)
        future = _call(s.end_sync)
        assert not future.done()
        free(s)
        assert future.result(timeout=5) == [0]


class TestFailure:
    def test_a_gather_error_releases_every_wait(self, server):
        s = server()
        s.register_consumer(0, [[(N, [[0, 8], [0, 4]])]])
        s.begin_sync([0], 0)
        waits = [_call(s.wait_freed), _call(s.bucket, 0, 0, 0)]
        s.set_gather_error("gather died")
        for w in waits:
            with pytest.raises(RuntimeError, match="gather died"):
                w.result(timeout=5)

    def test_a_dead_consumer_trips_the_watchdog(self, server):
        s = server(stall_timeout_s=0.2)
        s.register_consumer(0, [[(N, [[0, 8], [0, 4]])]])
        s.begin_sync([0], 0)
        _publish(s, _weights(), 0)
        with pytest.raises(RuntimeError, match="stall"):
            s.end_sync()

    def test_begin_sync_clears_an_error(self, server):
        s = server()
        s.set_gather_error("x")
        s.begin_sync([0], 1)
        s.register_consumer(0, [])
        _publish(s, _weights(), 0)
        assert s.end_sync() == [0]


def test_a_bucket_is_freed_when_ray_drops_it(server):
    """The free is Ray dropping its last reference to the served block: while
    Ray holds it the block and its unit stay taken, however long that is."""
    s = server(cls=_RDTProducerServer)
    s.register_consumer(0, [[(N, [[0, 8], [0, 4]])]])
    s.begin_sync([0], 0)
    _publish(s, _weights(), 0)
    ray_store = s.bucket(0, 0, 0)  # what Ray keeps until the consumer drops it
    block = weakref.ref(ray_store[0])
    gc.collect()
    assert s._blocks and s._freed_pending == []
    del ray_store
    # the producer kept no reference: Ray's was the last one
    assert block() is None
    assert s.wait_freed() == [0]
    assert s._blocks == {}


def test_buckets_freed_as_soon_as_served_all_count(server):
    """Two consumers, two pulls in flight each, every bucket freed the
    moment it is served: no free is missed, so every unit is released.
    (A store-then-free that polling Ray's store saw neither half of left a
    unit held forever.)"""
    s = server(cls=_RDTProducerServer, arena_bytes=3 * 16)  # pulls wait for room
    buckets = [[(N, [[r, r + 2], [0, 4]])] for r in range(0, 8, 2)] + [
        [(M, [[0, 4]])],
        [(E, [[0, 1], [0, 6]])],
        [(E, [[1, 2], [0, 6]])],
    ]
    for c in (0, 1):
        s.register_consumer(c, buckets)
    for sync_idx in range(3):
        s.begin_sync([0, 1], sync_idx)
        _publish(s, _weights(sync_idx), 0, 1)

        def pull(c, seqs, sync_idx):
            for seq in seqs:
                s.bucket(c, seq, sync_idx)  # Ray stores and frees at once

        pulls = [
            _call(pull, c, range(half, len(buckets), 2), sync_idx, timeout=5)
            for c in (0, 1)
            for half in (0, 1)
        ]
        for f in pulls:
            f.result(timeout=10)
        assert sorted(_call(s.end_sync, timeout=10).result(timeout=10)) == [0, 1]
        assert s._blocks == {}


# ------------------------------------------------------------ export ring


class TestExportRing:
    """The trainer packs small tensors into one export slot exactly the way
    `publish_unit` rebuilds them."""

    @staticmethod
    def _engine():
        eng = ShardedRDTTrainerWeightTransferEngine(
            client=Mock(),
            source=None,
            is_sender=True,
            init_info=ShardedRDTTrainerInitInfo(rank=0, num_consumers=1),
        )
        eng._export_ring = [None, None]
        eng._export_ring_args = [None, None]
        return eng

    @pytest.fixture(autouse=True)
    def _cpu_ipc(self, monkeypatch):
        monkeypatch.setattr(
            trainer_mod,
            "reduce_tensor",
            lambda base: (None, (base.numel(), None, None, None, None, None, -1)),
        )

    def test_packed_tensors_rebuild_bit_identically(self):
        held = [
            ("bf16.2d", torch.randn(12, 7).bfloat16()),
            ("fp32.1d", torch.randn(9)),
            ("bf16.transposed", torch.randn(6, 11).bfloat16().t()),
            ("bf16.sliced", torch.randn(8, 32).bfloat16()[:, 3:19]),
        ]
        eng = self._engine()
        storages, views, _ = eng._pack_for_export(held, 0)
        assert len(storages) == 1
        slot = eng._export_ring[0]
        for name, t in held:
            _, dt, shape, stride, off = views[name]
            got = torch.as_strided(slot.view(getattr(torch, dt)), shape, stride, off)
            assert got.dtype is t.dtype and torch.equal(got, t), name

    def test_slots_are_distinct(self):
        eng = self._engine()
        eng._pack_for_export([("a", torch.randn(8, 8).bfloat16())], 0)
        eng._pack_for_export([("b", torch.randn(8, 8).bfloat16())], 1)
        assert eng._export_ring[0].data_ptr() != eng._export_ring[1].data_ptr()
