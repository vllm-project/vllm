# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.v1.kv_connector.umbp_test_utils import (
    _kv_cache_config,
    _vllm_config,
    handshake,
    install_memory_runtime,
    make_kv_caches,
)
from tests.v1.kv_connector.unit.utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.connector import (
    UMBPStoreConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.data import (
    BlockIdentityCodec,
    BlockLoadBatch,
    BlockTransferPlan,
    KVLayoutPlanner,
    KVRange,
    KVRegion,
    RankTopology,
    TransferJobStatus,
    UMBPConnectorMetadata,
    UMBPNamespace,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime import (
    EmbeddedRuntime,
    UMBPRuntimeConfig,
    UMBPRuntimeFactory,
    embedded,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime.embedded import (
    _lookup_socket_path,
    _MoriLookupServer,
    _MoriSchedulerHandle,
    _MoriWorkerHandle,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.worker import (
    UMBPStoreConnectorWorker,
)
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    SlidingWindowSpec,
)
from vllm.v1.request import RequestStatus

mori_available = importlib.util.find_spec("mori") is not None

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and mori_available),
    reason="UMBP is only available on ROCm with the mori package installed",
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
def bulk_worker(monkeypatch, socket_dir):
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
        socket_dir,
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


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"dram_low_watermark": 0.9, "dram_high_watermark": 0.7}, "must not"),
        ({"dram_use_hugepages": 1}, "boolean"),
        ({"dram_numa_node": -2}, ">= -1"),
        ({"dram_hugepage_size": 0}, "positive integer"),
        ({"capacity_bytes": 1024, "total_capacity_bytes": 4096}, "mutually exclusive"),
    ],
)
def test_runtime_config_rejects_invalid_embedded_dram_options(options, error):
    with pytest.raises(ValueError, match=error):
        UMBPRuntimeFactory.build(
            UMBPRuntimeConfig.from_vllm(_vllm_config({"mode": "embedded", **options}))
        )


def test_mori_client_is_created_only_for_worker(monkeypatch, socket_dir):
    """Scheduler lookup must not allocate a second, unused DRAM store."""
    monkeypatch.setitem(sys.modules, "mori.cpp", None)
    runtime = EmbeddedRuntime.from_config(
        UMBPRuntimeConfig(
            "embedded", {"capacity_bytes": 1024, "lookup_dir": socket_dir}
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
    assert config.options["total_capacity_bytes"] == 4096
    assert "capacity_bytes" not in config.options


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


def test_lookup_sockets_separate_engines_and_data_parallel_ranks(socket_dir):
    def path(options, dp_index=0):
        config = UMBPRuntimeConfig("embedded", {"lookup_dir": socket_dir, **options})
        handle = EmbeddedRuntime.from_config(
            replace(config, dp_index=dp_index)
        ).create_scheduler_handle("same-model", RankTopology(), None)
        return handle._paths[RankTopology().local_namespace]

    assert path({}) == path({})
    assert path({}, dp_index=0) != path({}, dp_index=1)
    assert path({"lookup_instance": "a"}) != path({"lookup_instance": "b"})
    with pytest.raises(ValueError, match="lookup_instance"):
        path({"lookup_instance": 1})


def test_lookup_server_refuses_a_socket_another_engine_serves(socket_dir):
    client = SimpleNamespace(batch_exists=lambda keys: [True] * len(keys))
    path = _lookup_socket_path("taken", (0, 0, 0, 0), socket_dir, "dp0")
    first = _MoriLookupServer(path, client)
    second = _MoriLookupServer(path, client)
    first.start()
    try:
        with pytest.raises(RuntimeError, match="another live engine"):
            second.start()
        second.close()
        scheduler = _MoriSchedulerHandle("taken", RankTopology(), socket_dir, "dp0")
        assert scheduler.lookup(["k"]) == [True]
    finally:
        second.close()
        first.close()


def test_lookup_server_replaces_a_stale_socket_file(socket_dir):
    import socket

    path = _lookup_socket_path("stale", (0, 0, 0, 0), socket_dir, "dp0")
    dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    dead.bind(path)
    dead.close()
    server = _MoriLookupServer(
        path, SimpleNamespace(batch_exists=lambda keys: [True] * len(keys))
    )
    server.start()
    try:
        scheduler = _MoriSchedulerHandle("stale", RankTopology(), socket_dir, "dp0")
        assert scheduler.lookup(["k"]) == [True]
    finally:
        server.close()


@pytest.mark.parametrize("close_rank", [None, 0, 1])
def test_mori_scheduler_routes_lookup_and_clear_to_all_ranks(socket_dir, close_rank):
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
            _lookup_socket_path(namespace, rank, socket_dir),
            client,
        )
        server.start()
        servers.append(server)
    scheduler = _MoriSchedulerHandle(namespace, topology, socket_dir)
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


def test_single_rank_lookup_returns_fresh_results(socket_dir):
    """The only rank also answers legacy keys, and hits are never cached."""
    namespace = "single-route"
    codec = BlockIdentityCodec(UMBPNamespace(namespace))
    topology = RankTopology()
    keys = [codec.key(b"known"), "legacy", codec.key(b"new")]
    resident = set(keys[:2])
    server = _MoriLookupServer(
        _lookup_socket_path(namespace, topology.local_namespace, socket_dir),
        SimpleNamespace(batch_exists=lambda keys: [key in resident for key in keys]),
    )
    scheduler = _MoriSchedulerHandle(namespace, topology, socket_dir)
    server.start()
    try:
        assert scheduler.lookup(keys) == [True, True, False]
        resident.clear()
        resident.add(keys[-1])
        assert scheduler.lookup(keys) == [False, False, True]
        server.close()
        assert scheduler.lookup(keys) == [False] * len(keys)
        assert scheduler.last_lookup_diagnostics["unavailable_ranks"] == (
            topology.local_namespace,
        )
    finally:
        scheduler.close()
        server.close()


@pytest.mark.parametrize("flush_succeeds", [False, True])
def test_mori_publication_does_not_flush_on_the_worker_thread(
    socket_dir, flush_succeeds
):
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
    handle = _MoriWorkerHandle(client, "flush", RankTopology(), socket_dir, 1, 5)
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


def test_backend_timeout_fails_the_finished_job(socket_dir):
    """A TimeoutError from MORI itself must not read as buffers still in use."""

    def get(*args):
        raise TimeoutError("backend already stopped")

    handle = _MoriWorkerHandle(
        SimpleNamespace(batch_get_ranges_into_ptr=get, close=lambda: None),
        "timeout",
        RankTopology(),
        socket_dir,
        1,
        5,
    )
    try:
        job = handle.load([BlockTransferPlan("failed", 0)])
        assert handle.wait(job).status is TransferJobStatus.FAILED
        assert handle.poll(job) is job
    finally:
        handle.close()


@pytest.fixture(params=["load", "store", "flush"])
def blocked_transfer(request, monkeypatch, socket_dir):
    started = threading.Event()
    release = threading.Event()
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(embedded.time, "monotonic", lambda: clock.now)

    def block(stage):
        if request.param == stage:
            started.set()
            assert release.wait(5)

    def get(keys, *args):
        block("load")
        return [True] * len(keys)

    def put(keys, *args):
        block("store")
        return [True] * len(keys)

    def flush():
        block("flush")
        return True

    client = SimpleNamespace(
        batch_get_ranges_into_ptr=get,
        batch_put_ranges_from_ptr=put,
        flush=flush,
        close=lambda: None,
    )
    handle = _MoriWorkerHandle(client, "blocked", RankTopology(), socket_dir, 1, 0.1)
    transfer = handle.load if request.param == "load" else handle.store
    try:
        job = transfer([BlockTransferPlan("running", 0)])
        assert started.wait(5)
        yield handle, job, transfer, release, clock
    finally:
        release.set()
        handle.close()


@pytest.mark.parametrize("operation", ["poll", "wait", "cancel"])
def test_transfer_timeout_keeps_buffers_owned(blocked_transfer, operation):
    handle, job, _, release, clock = blocked_transfer
    assert handle.poll(job) is None
    clock.now += 1
    with pytest.raises(TimeoutError, match="buffers still in use"):
        getattr(handle, operation)(job)
    assert job.status is TransferJobStatus.RUNNING

    release.set()
    handle.close()
    assert handle.poll(job).status is TransferJobStatus.COMPLETED


def test_timeout_does_not_count_time_waiting_for_a_worker(blocked_transfer):
    handle, _, transfer, release, clock = blocked_transfer
    queued = transfer([BlockTransferPlan("queued", 1)])
    finished = threading.Event()
    handle._futures[id(queued)].add_done_callback(lambda _: finished.set())
    clock.now += 1
    assert handle.poll(queued) is None

    release.set()
    assert finished.wait(5)
    assert handle.poll(queued).status is TransferJobStatus.COMPLETED


def test_lookup_server_close_does_not_unlink_a_replacement(socket_dir):
    client = SimpleNamespace(batch_exists=lambda keys: [True] * len(keys))
    path = _lookup_socket_path("replaced", (0, 0, 0, 0), socket_dir)
    first = _MoriLookupServer(path, client)
    first.start()
    first.close()
    second = _MoriLookupServer(path, client)
    second.start()
    try:
        first.close()
        scheduler = _MoriSchedulerHandle("replaced", RankTopology(), socket_dir)
        assert scheduler.lookup(["k"]) == [True]
    finally:
        second.close()


def _engine_step(scheduler, worker, before_load=lambda: None):
    """One engine step with the worker connector driven as the model runner does."""
    output = scheduler.schedule()
    before_load()
    worker.bind_connector_metadata(output.kv_connector_metadata)
    worker.start_load_kv(None)
    worker.wait_for_save()
    results = worker.get_transfer_results(output.finished_req_ids)
    runner_output = create_model_runner_output(
        [scheduler.requests[req_id] for req_id in output.num_scheduled_tokens],
        finished_sending=results.finished_sending,
        finished_recving=results.finished_recving,
        failed_recving=results.failed_recving,
        invalid_block_ids=worker.get_block_ids_with_load_errors(),
        kv_connector_worker_meta=worker.build_connector_worker_meta(),
    )
    worker.clear_connector_metadata()
    scheduler.update_from_output(output, runner_output)
    return output


def _run_to_completion(scheduler, worker) -> dict[str, int]:
    scheduled: dict[str, int] = {}
    for _ in range(10):
        if not scheduler.has_requests():
            return scheduled
        output = _engine_step(scheduler, worker)
        for req_id, count in output.num_scheduled_tokens.items():
            scheduled[req_id] = scheduled.get(req_id, 0) + count
    raise AssertionError("the engine did not drain")


def _attention_and_sliding_window_config() -> KVCacheConfig:
    shape = dict(block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float16)
    groups = [
        KVCacheGroupSpec(["full"], FullAttentionSpec(**shape)),
        KVCacheGroupSpec(["window"], SlidingWindowSpec(**shape, sliding_window=64)),
    ]
    page_bytes = groups[0].kv_cache_spec.page_size_bytes
    return KVCacheConfig(
        num_blocks=16,
        kv_cache_groups=groups,
        kv_cache_tensors=[
            KVCacheTensor(
                size=16 * page_bytes,
                layers=group.layer_names,
                layer_stride=16 * page_bytes,
                block_stride=page_bytes,
            )
            for group in groups
        ],
    )


@pytest.mark.parametrize(
    ("num_groups", "load_async"), [(1, True), (1, False), (2, True)]
)
@pytest.mark.parametrize("policy", ["fail", "recompute"])
def test_object_evicted_after_lookup_follows_failure_policy(
    monkeypatch, socket_dir, num_groups, load_async, policy
):
    """A hit that is gone by load time must fail or recompute, never reuse KV."""
    runtime = install_memory_runtime(monkeypatch)
    if num_groups == 1:
        kv_cache_config = _kv_cache_config()
        kv_caches = make_kv_caches()
    else:
        kv_cache_config = _attention_and_sliding_window_config()
        kv_caches = {
            name: torch.zeros(
                (16, kv_cache_config.kv_cache_tensors[0].block_stride),
                dtype=torch.uint8,
            )
            for name in ("full", "window")
        }
    vllm_config = create_vllm_config(
        kv_connector="UMBPStoreConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "mode": "embedded",
            "lookup_dir": socket_dir,
            "load_async": load_async,
        },
        kv_load_failure_policy=policy,
    )
    scheduler = create_scheduler(
        vllm_config,
        num_blocks=kv_cache_config.num_blocks,
        kv_cache_config=kv_cache_config,
    )
    worker = UMBPStoreConnector(vllm_config, KVConnectorRole.WORKER, kv_cache_config)
    worker.register_kv_caches(kv_caches)
    handshake(scheduler.connector, worker)
    num_tokens = 3 * scheduler.block_size + 1
    num_hit_tokens = 3 * scheduler.block_size

    def same_prompt():
        return create_request(
            num_tokens=num_tokens, common_prefix_len=num_tokens, max_tokens=1
        )

    scheduler.add_request(same_prompt())
    _run_to_completion(scheduler, worker)
    assert scheduler.reset_prefix_cache()

    request = same_prompt()
    scheduler.add_request(request)
    free_blocks = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()

    def evict_after_lookup():
        assert request.prefill_stats.num_external_cached_tokens == num_hit_tokens
        runtime.evict_all()

    _engine_step(scheduler, worker, before_load=evict_after_lookup)
    if policy == "fail":
        assert request.status == RequestStatus.FINISHED_ERROR
        _run_to_completion(scheduler, worker)
        assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == (
            free_blocks
        )
    else:
        scheduled = _run_to_completion(scheduler, worker)
        assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
        assert scheduled[request.request_id] >= num_tokens
    # Blocks the failed load left unfilled must never serve a prefix hit.
    _, num_cached, _ = scheduler.kv_cache_manager.get_computed_blocks(same_prompt())
    assert num_cached == (0 if policy == "fail" else num_hit_tokens)
    worker.shutdown()
