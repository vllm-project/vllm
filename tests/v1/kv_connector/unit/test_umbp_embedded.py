# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import threading
import time
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.v1.kv_connector.umbp_test_utils import (
    MemoryRuntime,
    _kv_cache_config,
    _vllm_config,
    install_memory_runtime,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.connector import (
    UMBPStoreConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.data import (
    BlockIdentityCodec,
    BlockLoadBatch,
    BlockTransferPlan,
    KVLayoutDescriptor,
    KVLayoutPlanner,
    KVRange,
    KVRegion,
    RankTopology,
    TransferJobState,
    TransferJobStatus,
    UMBPConnectorMetadata,
    UMBPNamespace,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime import (
    EmbeddedRuntime,
    UMBPRuntimeConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime.embedded import (
    _configure_dram,
    _lookup_socket_path,
    _MoriLookupServer,
    _MoriSchedulerHandle,
    _MoriWorkerHandle,
    _rank_namespace_from_key_prefix,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.worker import (
    UMBPStoreConnectorWorker,
)


def _cpu_caches() -> dict[str, torch.Tensor]:
    return {
        name: torch.empty_strided(
            (8, 2, 16, 8),
            (512, 256, 8, 1),
            dtype=torch.float16,
        )
        for name in ("layer1", "layer2")
    }


def _request(request_id: str, block_ids: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        req_id=request_id,
        num_tokens=32,
        block_hashes=[f"{request_id}-a".encode(), f"{request_id}-b".encode()],
        block_ids=(block_ids,),
        num_computed_tokens=0,
    )


def _scheduler_output(request: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[request],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={request.req_id: 32},
    )


@pytest.mark.parametrize("for_store", [False, True])
def test_mori_range_arguments_preserve_sparse_offsets_and_empty_objects(for_store):
    plans = [
        BlockTransferPlan("empty", 0),
        BlockTransferPlan(
            "scatter",
            1,
            ranges=(
                KVRange("layer0", 0, 1, 1000, 64, 16, 48),
                KVRange("layer1", 0, 1, 2000, 64, 8, 0),
            ),
        ),
    ]
    assert _MoriWorkerHandle._range_args(plans, for_store=for_store) == (
        ["empty", "scatter"],
        [0, 64] if for_store else [],
        [[], [1000, 2000]],
        [[], [16, 8]],
        [[], [48, 0]],
    )


@pytest.fixture
def bulk_worker(monkeypatch, tmp_path):
    calls: list[
        tuple[list[str], list[int], list[list[int]], list[list[int]], list[list[int]]]
    ] = []

    def get(keys, pointers, sizes, offsets):
        calls.append((keys, [], pointers, sizes, offsets))
        return [key != "missing" for key in keys]

    def put(keys, object_sizes, pointers, sizes, offsets):
        calls.append((keys, object_sizes, pointers, sizes, offsets))
        return [True] * len(keys)

    monkeypatch.setitem(
        sys.modules,
        "mori.cpp",
        SimpleNamespace(MemoryLocationType=SimpleNamespace(CPU=0, GPU=1)),
    )
    client = SimpleNamespace(
        register_memory=lambda *a: True,
        deregister_memory=lambda *a: True,
        batch_get_ranges_into_ptr=get,
        batch_put_ranges_from_ptr=put,
        flush=lambda: True,
        close=lambda: None,
    )
    planner = KVLayoutPlanner(
        [
            KVRegion("a", 0, 64, 32, 0, 16),
            KVRegion("b", 1, 32, 16, 32, 16),
            KVRegion("c", 0, 128, 64, 48, 16),
        ]
    )
    handle = _MoriWorkerHandle(
        client,
        "bulk",
        RankTopology(),
        str(tmp_path),
        1,
        5,
        planner.describe(RankTopology()),
    )
    worker = UMBPStoreConnectorWorker(handle, planner)
    buffers = {
        r.layer_name: torch.empty((8, r.block_stride), dtype=torch.uint8)
        for r in planner.regions
    }
    worker.register_kv_caches(buffers)
    yield worker, buffers, calls
    handle.close()


@pytest.mark.parametrize("batched", [False, True])
def test_bulk_load_preserves_group_order_failures_and_new_buffers(
    bulk_worker, batched, monkeypatch
):
    worker, buffers, calls = bulk_worker
    plans = [
        BlockTransferPlan("group1", 3, group_id=1, request_id="r"),
        BlockTransferPlan("missing", 4, group_id=0, request_id="r"),
        BlockTransferPlan("group0", 6, group_id=0, request_id="r"),
    ]
    for caches in (buffers, {name: torch.empty_like(t) for name, t in buffers.items()}):
        worker.register_kv_caches(caches)
        expected = _MoriWorkerHandle._range_args(
            [worker.layout.materialize(p) for p in plans], for_store=False
        )
        supplied = plans
        if batched:
            supplied = BlockLoadBatch(
                [p.key for p in plans],
                [p.block_id for p in plans],
                [p.group_id for p in plans],
                "r",
            )
            monkeypatch.setattr(
                BlockLoadBatch,
                "__getitem__",
                lambda *a: pytest.fail(
                    "bulk loading must not materialize per-block plans"
                ),
            )
        job = worker.runtime.load_blocks(supplied)
        assert job is not None
        result = worker.runtime.wait(job)
        assert calls[-1] == expected
        assert result.plans is supplied if batched else result.plans == tuple(plans)
        assert result.failed_block_ids == {4}
        assert result.completed_keys == {"group0", "group1"}
        assert result.completed_bytes == 112
    assert len(calls) == 2
    assert calls[0][2] != calls[1][2]


def test_bulk_store_matches_materialized_ranges(bulk_worker, monkeypatch):
    worker, _, calls = bulk_worker
    plans = [
        BlockTransferPlan("group1", 3, group_id=1, request_id="r"),
        BlockTransferPlan("group0", 6, group_id=0, request_id="r"),
    ]
    expected = _MoriWorkerHandle._range_args(
        [worker.layout.materialize(p) for p in plans]
    )
    monkeypatch.setattr(
        worker.layout,
        "materialize",
        lambda *a: pytest.fail("bulk stores must not materialize per-block plans"),
    )

    worker.enqueue_stores(UMBPConnectorMetadata(store_requests={"r": plans}))
    (job,) = worker._store_jobs.values()
    result = worker.runtime.wait(job)

    assert calls == [expected]
    assert result.plans == tuple(plans)
    assert result.completed_keys == {"group0", "group1"}
    assert result.completed_bytes == 112


@pytest.mark.parametrize("special", ["subblocks", "materialized", "batch-subblocks"])
def test_bulk_load_falls_back_for_entire_mixed_batch(bulk_worker, special):
    worker, buffers, calls = bulk_worker
    plans = [
        BlockTransferPlan("a", 1, group_id=1),
        BlockTransferPlan("c", 2, group_id=0),
    ]
    if special in ("subblocks", "batch-subblocks"):
        buffers["c"] = torch.empty((16, 32), dtype=torch.uint8)
        worker.register_kv_caches(buffers)
        if special == "batch-subblocks":
            plans = BlockLoadBatch(
                [p.key for p in plans],
                [p.block_id for p in plans],
                [p.group_id for p in plans],
                "r",
            )
    else:
        plans[1] = worker.layout.materialize(plans[1])
    assert worker.runtime.load_blocks(plans) is None
    assert not calls
    expected = _MoriWorkerHandle._range_args(
        [p if p.ranges else worker.layout.materialize(p) for p in plans],
        for_store=False,
    )
    worker.start_load_kv(None, UMBPConnectorMetadata(load_requests={"r": plans}))
    worker.wait_for_layer_load("")
    assert calls == [expected]


def test_embedded_runtime_maps_dram_options_to_mori_config():
    dram = SimpleNamespace()
    config = SimpleNamespace(dram=dram)

    _configure_dram(
        config,
        {
            "capacity_bytes": 1024,
            "dram_use_shared_memory": True,
            "dram_shm_name": "umbp-test",
            "dram_high_watermark": 0.9,
            "dram_low_watermark": 0.7,
            "dram_use_hugepages": True,
            "dram_hugepage_size": 2 * 1024**2,
            "dram_numa_node": 1,
            "dram_prefault": True,
        },
    )

    assert vars(dram) == {
        "capacity_bytes": 1024,
        "use_shared_memory": True,
        "shm_name": "umbp-test",
        "high_watermark": 0.9,
        "low_watermark": 0.7,
        "use_hugepages": True,
        "hugepage_size": 2 * 1024**2,
        "numa_node": 1,
        "prefault": True,
    }


def test_embedded_rejects_test_only_backend():
    """Retired test configuration must not silently select the MORI backend."""
    with pytest.raises(ValueError, match="test-only"):
        EmbeddedRuntime.from_config(
            UMBPRuntimeConfig("embedded", {"backend": "memory"})
        )


def test_mori_client_is_created_only_for_worker(monkeypatch, tmp_path):
    """Scheduler lookup must not allocate a second, unused DRAM store."""
    monkeypatch.setitem(sys.modules, "mori.cpp", None)
    runtime = EmbeddedRuntime.from_config(
        UMBPRuntimeConfig(
            "embedded", {"capacity_bytes": 1024, "lookup_dir": str(tmp_path)}
        )
    )
    topology = RankTopology()
    layout = KVLayoutPlanner.from_kv_cache_config(_kv_cache_config()).describe(topology)
    scheduler = runtime.create_scheduler_handle("worker-only", topology, layout)
    scheduler.close()
    with pytest.raises(RuntimeError, match="requires MORI"):
        runtime.create_worker_handle("worker-only", topology, layout)

    configs = []

    def create_client(config):
        configs.append(config)
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setitem(
        sys.modules,
        "mori.cpp",
        SimpleNamespace(
            UMBPClient=create_client,
            UMBPConfig=lambda: SimpleNamespace(dram=SimpleNamespace()),
        ),
    )
    worker = runtime.create_worker_handle("worker-only", topology, layout)
    try:
        assert len(configs) == 1
        assert configs[0].dram.capacity_bytes == 1024
    finally:
        worker.close()


def test_runtime_config_resolves_total_embedded_capacity_per_rank():
    config = UMBPRuntimeConfig.from_vllm(
        _vllm_config({"mode": "embedded", "total_capacity_bytes": 4096})
    ).resolve_for_rank_count(4)

    runtime = EmbeddedRuntime.from_config(config)
    assert runtime.options["capacity_bytes"] == 1024
    assert runtime.options["_configured_total_capacity_bytes"] == 4096
    assert config.options["total_capacity_bytes"] == 4096
    assert "capacity_bytes" not in config.options


def test_runtime_config_rejects_ambiguous_embedded_capacity():
    with pytest.raises(ValueError, match="mutually exclusive"):
        EmbeddedRuntime.from_config(
            UMBPRuntimeConfig(
                "embedded", {"capacity_bytes": 1024, "total_capacity_bytes": 4096}
            )
        )


def test_embedded_connector_rejects_pipeline_parallelism():
    with pytest.raises(NotImplementedError, match="pipeline parallelism"):
        UMBPStoreConnector(
            _vllm_config(
                {"mode": "embedded"},
                pipeline_parallel_size=2,
            ),
            KVConnectorRole.SCHEDULER,
            _kv_cache_config(),
        )


def test_data_parallel_ranks_get_separate_lookup_sockets(tmp_path):
    topology = RankTopology()
    handles = [
        EmbeddedRuntime.from_config(
            replace(
                UMBPRuntimeConfig.from_vllm(
                    _vllm_config({"mode": "embedded", "lookup_dir": str(tmp_path)})
                ),
                dp_index=dp_index,
            )
        ).create_scheduler_handle("dp-shared", topology, None)
        for dp_index in (0, 1)
    ]

    paths = [handle._paths[topology.local_namespace] for handle in handles]
    assert paths[0] != paths[1]
    for handle in handles:
        handle.close()


def test_lookup_instance_separates_engines_on_one_host(tmp_path):
    def path(options):
        runtime = EmbeddedRuntime.from_config(
            UMBPRuntimeConfig("embedded", {"lookup_dir": str(tmp_path), **options})
        )
        handle = runtime.create_scheduler_handle("same-model", RankTopology(), None)
        return handle._paths[RankTopology().local_namespace]

    assert path({"lookup_instance": "a"}) != path({"lookup_instance": "b"})
    assert path({}) == path({})
    with pytest.raises(ValueError, match="lookup_instance"):
        path({"lookup_instance": 1})


def test_lookup_server_refuses_a_socket_another_engine_serves(tmp_path):
    client = SimpleNamespace(batch_exists=lambda keys: [False] * len(keys))
    path = _lookup_socket_path("taken", (0, 0, 0, 0), str(tmp_path), "dp0")
    first = _MoriLookupServer(path, client)
    first.start()
    try:
        with pytest.raises(RuntimeError, match="another live engine"):
            _MoriLookupServer(path, client).start()
        scheduler = _MoriSchedulerHandle("taken", RankTopology(), str(tmp_path), "dp0")
        # The first engine's pool still answers its own scheduler.
        assert scheduler.lookup(["k"]) == [False]
    finally:
        first.close()


def test_lookup_server_replaces_a_stale_socket_file(tmp_path):
    import socket

    path = _lookup_socket_path("stale", (0, 0, 0, 0), str(tmp_path), "dp0")
    dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    dead.bind(path)
    dead.close()
    server = _MoriLookupServer(
        path, SimpleNamespace(batch_exists=lambda keys: [True] * len(keys))
    )
    server.start()
    try:
        scheduler = _MoriSchedulerHandle("stale", RankTopology(), str(tmp_path), "dp0")
        assert scheduler.lookup(["k"]) == [True]
    finally:
        server.close()


@pytest.mark.parametrize("close_rank", [None, 0, 1])
def test_mori_scheduler_routes_lookup_and_clear_to_all_ranks(tmp_path, close_rank):
    _rank_namespace_from_key_prefix.cache_clear()

    class _Client:
        def __init__(self, keys):
            self.keys = set(keys)

        def batch_exists(self, keys):
            return [key in self.keys for key in keys]

        def clear(self):
            self.keys.clear()
            return True

    namespace = "routed-lookup"
    topology = RankTopology(tp_size=2)
    codec0 = BlockIdentityCodec(UMBPNamespace(namespace), tp_rank=0)
    codec1 = BlockIdentityCodec(UMBPNamespace(namespace), tp_rank=1)
    keys = [codec0.key(b"a"), codec1.key(b"a")]
    servers = []
    clients = [_Client([key, "legacy-key"]) for key in keys]
    for rank, client in zip(topology.all_namespaces(), clients, strict=True):
        server = _MoriLookupServer(
            _lookup_socket_path(namespace, rank, str(tmp_path)),
            client,
        )
        server.start()
        servers.append(server)
    scheduler = _MoriSchedulerHandle(namespace, topology, str(tmp_path))
    try:
        assert scheduler.lookup(keys) == [True, True]
        assert scheduler.last_lookup_diagnostics == {
            "unavailable_ranks": (),
            "missing_keys": (),
        }
        new_keys = [codec0.key(b"new"), codec1.key(b"new")]
        assert scheduler.lookup(new_keys) == [False, False]
        for client, key in zip(clients, new_keys, strict=True):
            client.keys.add(key)
        assert scheduler.lookup(new_keys) == [True, True]
        assert _rank_namespace_from_key_prefix.cache_info().misses == 2
        assert scheduler.lookup([keys[1], "legacy-key", keys[0]]) == [True] * 3
        if close_rank is not None:
            servers[close_rank].close()
            assert scheduler.lookup(keys) == [rank != close_rank for rank in range(2)]
            assert scheduler.last_lookup_diagnostics["unavailable_ranks"] == (
                topology.all_namespaces()[close_rank],
            )
            assert scheduler.last_lookup_diagnostics["missing_keys"] == (
                keys[close_rank],
            )
        assert scheduler.clear() is (close_rank is None)
        for rank, client in enumerate(clients):
            if rank != close_rank:
                assert not client.keys
        assert scheduler.lookup(keys + ["legacy-key"]) == [False] * 3
    finally:
        for server in servers:
            server.close()


def test_single_rank_lookup_avoids_routing_without_caching_hits(monkeypatch, tmp_path):
    """The only rank also handles legacy keys; all results must stay authoritative."""
    namespace = "single-route"
    codec = BlockIdentityCodec(UMBPNamespace(namespace))
    topology = RankTopology()
    keys = [codec.key(b"known"), "legacy", codec.key(b"new")]
    resident = set(keys[:2])
    calls = []

    def query(request_keys):
        calls.append(tuple(request_keys))
        return [key in resident for key in request_keys]

    server = _MoriLookupServer(
        _lookup_socket_path(namespace, topology.local_namespace, str(tmp_path)),
        SimpleNamespace(batch_exists=query),
    )
    scheduler = _MoriSchedulerHandle(namespace, topology, str(tmp_path))
    server.start()
    try:

        def no_routing(*args):
            pytest.fail("single-rank successful lookup must not parse key routing")

        with monkeypatch.context() as patch:
            patch.setattr(
                "vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime."
                "embedded._rank_namespace_from_key_prefix",
                no_routing,
            )
            assert scheduler.lookup(keys) == [True, True, False]
            resident.clear()
            resident.add(keys[-1])
            assert scheduler.lookup(keys) == [False, False, True]
        assert calls == [tuple(keys), tuple(keys)]
        server.close()
        failed_calls = []

        def unavailable(*args):
            failed_calls.append(args)
            raise TimeoutError("worker unavailable")

        monkeypatch.setattr(scheduler, "_request", unavailable)
        assert scheduler.lookup(keys) == [False] * len(keys)
        assert len(failed_calls) == 1
        assert scheduler.last_lookup_diagnostics["unavailable_ranks"] == (
            topology.local_namespace,
        )
    finally:
        scheduler.close()
        server.close()


def test_embedded_round_trip_restores_all_layer_ranges(monkeypatch):
    install_memory_runtime(monkeypatch)
    kv_config = _kv_cache_config()
    vllm_config = _vllm_config(
        {
            "mode": "embedded",
            "load_async": False,
            "key_namespace": "embedded-round-trip",
        }
    )
    source = _cpu_caches()
    for index, cache in enumerate(source.values()):
        cache.copy_(
            torch.arange(cache.numel(), dtype=torch.float16).reshape(cache.shape)
            + index
        )

    scheduler = UMBPStoreConnector(vllm_config, KVConnectorRole.SCHEDULER, kv_config)
    worker = UMBPStoreConnector(vllm_config, KVConnectorRole.WORKER, kv_config)
    worker.register_kv_caches(source)

    producer = _request("producer", [1, 2])
    scheduler.update_state_after_alloc(
        producer,
        SimpleNamespace(get_block_ids=lambda group_ids: producer.block_ids),
        0,
    )
    store_meta = scheduler.build_connector_meta(_scheduler_output(producer))
    worker.bind_connector_metadata(store_meta)
    worker.wait_for_save()

    consumer = _request("consumer", [5, 6])
    consumer.num_tokens = 33
    consumer.block_hashes = producer.block_hashes
    assert scheduler.get_num_new_matched_tokens(consumer, 0) == (32, False)
    scheduler.update_state_after_alloc(
        consumer,
        SimpleNamespace(get_block_ids=lambda group_ids: ([5, 6],)),
        32,
    )
    load_meta = scheduler.build_connector_meta(_scheduler_output(consumer))

    destination = {
        name: torch.empty_strided(
            (8, 2, 16, 8),
            (512, 256, 8, 1),
            dtype=torch.float16,
        )
        for name in source
    }
    for cache in destination.values():
        cache.zero_()
    worker.register_kv_caches(destination)
    worker.bind_connector_metadata(load_meta)
    worker.start_load_kv(None)
    worker.wait_for_layer_load("layer0")

    for name in source:
        if not torch.equal(source[name][1], destination[name][5]):
            raise AssertionError(f"{name} first block did not round-trip")
        if not torch.equal(source[name][2], destination[name][6]):
            raise AssertionError(f"{name} second block did not round-trip")
    assert worker.get_block_ids_with_load_errors() == set()


def test_embedded_missing_object_reports_target_block_for_recompute(monkeypatch):
    install_memory_runtime(monkeypatch)
    kv_config = _kv_cache_config()
    vllm_config = _vllm_config(
        {
            "mode": "embedded",
            "key_namespace": "embedded-missing",
        }
    )
    worker = UMBPStoreConnector(vllm_config, KVConnectorRole.WORKER, kv_config)
    worker.register_kv_caches(_cpu_caches())
    worker_impl = worker.connector_worker
    assert worker_impl is not None
    assert worker_impl.layout is not None
    plan = worker_impl.layout.plan_registered_block("missing-key", 7)

    metadata = UMBPConnectorMetadata(
        load_requests={"request": [plan]},
    )
    worker.bind_connector_metadata(metadata)
    worker.start_load_kv(None)
    worker.wait_for_layer_load("layer0")

    assert worker.get_block_ids_with_load_errors() == {7}


def test_embedded_publish_makes_object_visible_atomically():
    runtime = MemoryRuntime()
    topology = RankTopology()
    layout = KVLayoutDescriptor(
        regions=(KVRegion("layer0", 0, 16, 16, 0),),
        topology=topology,
    )
    scheduler = runtime.create_scheduler_handle("publish-test", topology, layout)
    worker = runtime.create_worker_handle("publish-test", topology, layout)
    cache = torch.zeros(16, dtype=torch.uint8)
    cache.copy_(torch.arange(16, dtype=torch.uint8))
    worker.register_buffers({"layer0": cache})
    plan = BlockTransferPlan(
        key="publish-key",
        block_id=0,
        ranges=(
            KVRange(
                "layer0",
                0,
                0,
                cache.data_ptr(),
                16,
                16,
                0,
            ),
        ),
    )

    job = worker.store([plan])
    assert scheduler.lookup(["publish-key"]) == [False]
    completed = worker.wait(job)
    worker.publish(completed)

    assert scheduler.lookup(["publish-key"]) == [True]
    worker.close()
    scheduler.close()


def test_embedded_scheduler_clear_removes_published_objects():
    runtime = MemoryRuntime()
    topology = RankTopology()
    layout = KVLayoutDescriptor(
        regions=(KVRegion("layer0", 0, 16, 16, 0),),
        topology=topology,
    )
    scheduler = runtime.create_scheduler_handle("clear", topology, layout)
    worker = runtime.create_worker_handle("clear", topology, layout)
    source = torch.arange(16, dtype=torch.uint8)
    worker.register_buffers({"layer0": source})
    plan = BlockTransferPlan(
        "clear-key",
        0,
        ranges=(KVRange("layer0", 0, 0, source.data_ptr(), 16, 16, 0),),
    )
    job = worker.wait(worker.store([plan]))
    worker.publish(job)

    assert scheduler.lookup(["clear-key"]) == [True]
    assert scheduler.clear()
    assert scheduler.lookup(["clear-key"]) == [False]
    worker.close()
    scheduler.close()


def test_mori_worker_reports_published_key_eviction(tmp_path):
    class _Client:
        def flush(self):
            return True

        def batch_exists(self, keys):
            return [False] * len(keys)

        def close(self):
            pass

    handle = _MoriWorkerHandle(
        _Client(),
        "eviction",
        RankTopology(),
        str(tmp_path),
        1,
        1,
    )
    plan = BlockTransferPlan("evicted-key", 0)
    job = TransferJobState((plan,))
    job.start()
    job.complete()
    handle.publish(job)

    assert handle.batch_exists(["evicted-key"]) == [False]
    assert handle.take_evicted_keys() == ("evicted-key",)
    assert handle.take_evicted_keys() == ()
    handle.close()


@pytest.mark.parametrize("operation", ["wait", "cancel"])
@pytest.mark.parametrize("timeout_first", [False, True])
@pytest.mark.parametrize("blocked_stage", ["copy", "flush", "load"])
def test_mori_transfer_keeps_buffers_owned_until_completion(
    tmp_path, operation, timeout_first, blocked_stage
):
    class _Client:
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()

        def register_memory(self, *args):
            return True

        def deregister_memory(self, *args):
            return True

        def batch_exists(self, keys):
            return [False] * len(keys)

        def batch_put_ranges_from_ptr(self, keys, *args):
            if blocked_stage == "copy":
                self.started.set()
                assert self.release.wait(5)
            return [True] * len(keys)

        def batch_get_ranges_into_ptr(self, keys, *args):
            if blocked_stage == "load":
                self.started.set()
                assert self.release.wait(5)
                return [True] * len(keys)
            return [False] * len(keys)

        def flush(self):
            if blocked_stage == "flush":
                self.started.set()
                assert self.release.wait(5)
            return True

        def clear(self):
            return True

        def close(self):
            pass

    client = _Client()
    source = torch.zeros((1, 1024), dtype=torch.uint8)
    handle = _MoriWorkerHandle(
        client,
        "cancel-reuse",
        RankTopology(),
        str(tmp_path),
        1,
        5,
        KVLayoutDescriptor((KVRegion("layer0", 0, 1024, 1024, 0),), RankTopology()),
    )
    handle.register_buffers({"layer0": source})
    job = (
        handle.load_blocks([BlockTransferPlan("cancel-reuse-key", 0, group_id=0)])
        if blocked_stage == "load"
        else handle.store(
            [
                BlockTransferPlan(
                    "cancel-reuse-key",
                    0,
                    ranges=(
                        KVRange(
                            "layer0",
                            0,
                            0,
                            source.data_ptr(),
                            source.numel(),
                            source.numel(),
                            0,
                        ),
                    ),
                )
            ]
        )
    )
    assert job is not None
    assert client.started.wait(5)
    assert handle.poll(job) is None
    assert handle.batch_exists(["cancel-reuse-key"]) == [False]
    result = []
    waiter = threading.Thread(
        target=lambda: result.append(getattr(handle, operation)(job))
    )
    try:
        if timeout_first:
            handle._timeout_s = 0.001
            with pytest.raises(TimeoutError, match="buffers still in use"):
                getattr(handle, operation)(job)
            assert job.status is TransferJobStatus.RUNNING
            assert handle.poll(job) is None
        handle._timeout_s = 5
        waiter.start()
        waiter.join(0.1)
        assert waiter.is_alive()
    finally:
        client.release.set()
        if waiter.ident is not None:
            waiter.join(5)
        handle.close()
    assert not waiter.is_alive()
    assert result[0].status is TransferJobStatus.COMPLETED
    source.fill_(1)


@pytest.mark.parametrize("flush_succeeds", [False, True])
def test_mori_publication_does_not_flush_on_the_worker_thread(tmp_path, flush_succeeds):
    flush_threads = []

    def flush():
        flush_threads.append(threading.get_ident())
        return flush_succeeds

    client = SimpleNamespace(
        batch_put_ranges_from_ptr=lambda keys, *args: [True] * len(keys),
        batch_exists=lambda keys: [True] * len(keys),
        flush=flush,
        close=lambda: None,
    )
    handle = _MoriWorkerHandle(client, "flush", RankTopology(), str(tmp_path), 1, 5)
    try:
        job = handle.wait(handle.store([BlockTransferPlan("key", 0)]))
        assert len(flush_threads) == 1
        assert flush_threads[0] != threading.get_ident()
        if flush_succeeds:
            handle.publish(job)
        else:
            assert job.status is TransferJobStatus.FAILED
            with pytest.raises(RuntimeError, match="incomplete"):
                handle.publish(job)
        assert len(flush_threads) == 1
    finally:
        handle.close()


def test_store_timeout_does_not_count_time_waiting_for_a_worker(tmp_path):
    release = threading.Event()

    def put(keys, *args):
        release.wait(5)
        return [True] * len(keys)

    client = SimpleNamespace(
        batch_put_ranges_from_ptr=put,
        batch_exists=lambda keys: [True] * len(keys),
        flush=lambda: True,
        close=lambda: None,
    )
    handle = _MoriWorkerHandle(client, "queued", RankTopology(), str(tmp_path), 1, 5)
    try:
        running = handle.store([BlockTransferPlan("running", 0)])
        handle._timeout_s = 0.05
        queued = handle.store([BlockTransferPlan("queued", 0)])
        time.sleep(0.2)
        # Still behind the running store on the only transfer thread.
        assert handle.poll(queued) is None
        handle._timeout_s = 5
        release.set()
        assert handle.wait(running).status is TransferJobStatus.COMPLETED
        assert handle.wait(queued).status is TransferJobStatus.COMPLETED
    finally:
        release.set()
        handle.close()


def test_async_store_timeout_does_not_release_buffers():
    handle = _MoriWorkerHandle(
        SimpleNamespace(close=lambda: None), "timeout", RankTopology(), "/tmp", 1, 1
    )
    job = TransferJobState((BlockTransferPlan("pending", 0),))
    job.start()
    future: Future[TransferJobState] = Future()
    handle._futures[id(job)] = future
    handle._store_deadlines[id(job)] = 0
    with pytest.raises(TimeoutError, match="buffers still in use"):
        handle.poll(job)
    assert job.status is TransferJobStatus.RUNNING
    assert handle._futures[id(job)] is future
    job.complete()
    future.set_result(job)
    assert handle.poll(job) is job
    handle.close()


def test_mori_completed_transfer_timeout_is_a_safe_failure():
    handle = _MoriWorkerHandle(
        SimpleNamespace(close=lambda: None), "timeout", RankTopology(), "/tmp", 1, 1
    )
    job = TransferJobState((BlockTransferPlan("failed", 0),))
    job.start()
    future: Future[TransferJobState] = Future()
    future.set_exception(TimeoutError("backend already stopped"))
    handle._futures[id(job)] = future
    assert handle.wait(job).status is TransferJobStatus.FAILED
    assert handle.poll(job) is job
    handle.close()
