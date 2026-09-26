# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.connector import (
    UMBPStoreConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.data import (
    BlockIdentityCodec,
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
)

from .test_umbp_shared import _kv_cache_config, _vllm_config


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


def test_runtime_config_resolves_total_embedded_capacity_per_rank():
    config = UMBPRuntimeConfig.from_vllm(
        _vllm_config({"mode": "embedded", "total_capacity_bytes": 4096})
    ).resolve_for_rank_count(4)

    assert config.options["capacity_bytes"] == 1024
    assert config.options["_configured_total_capacity_bytes"] == 4096


def test_runtime_config_rejects_ambiguous_embedded_capacity():
    with pytest.raises(ValueError, match="mutually exclusive"):
        UMBPRuntimeConfig.from_vllm(
            _vllm_config(
                {
                    "mode": "embedded",
                    "capacity_bytes": 1024,
                    "total_capacity_bytes": 4096,
                }
            )
        )


def test_embedded_connector_rejects_pipeline_parallelism():
    with pytest.raises(NotImplementedError, match="pipeline parallelism"):
        UMBPStoreConnector(
            _vllm_config(
                {"mode": "embedded", "backend": "memory"},
                pipeline_parallel_size=2,
            ),
            KVConnectorRole.SCHEDULER,
            _kv_cache_config(),
        )


def test_mori_scheduler_routes_lookup_to_owning_rank(tmp_path):
    class _Client:
        def __init__(self, keys):
            self.keys = set(keys)

        def batch_exists(self, keys):
            return [key in self.keys for key in keys]

        def clear(self):
            return True

    namespace = "routed-lookup"
    topology = RankTopology(tp_size=2)
    codec0 = BlockIdentityCodec(UMBPNamespace(namespace), tp_rank=0)
    codec1 = BlockIdentityCodec(UMBPNamespace(namespace), tp_rank=1)
    keys = [codec0.key(b"a"), codec1.key(b"a")]
    servers = []
    for rank, key in zip(topology.all_namespaces(), keys, strict=True):
        server = _MoriLookupServer(
            _lookup_socket_path(namespace, rank, str(tmp_path)),
            _Client([key]),
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
        servers[1].close()
        assert scheduler.lookup(keys) == [True, False]
        assert scheduler.last_lookup_diagnostics["unavailable_ranks"] == ((1, 0, 0, 0),)
        assert scheduler.last_lookup_diagnostics["missing_keys"] == (keys[1],)
    finally:
        for server in servers:
            server.close()


def test_embedded_round_trip_restores_all_layer_ranges():
    kv_config = _kv_cache_config()
    vllm_config = _vllm_config(
        {
            "mode": "embedded",
            "backend": "memory",
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
    store_meta = scheduler.build_connector_meta(_scheduler_output(producer))
    worker.bind_connector_metadata(store_meta)
    worker.wait_for_save()

    consumer = _request("consumer", [5, 6])
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


def test_embedded_missing_object_reports_target_block_for_recompute():
    kv_config = _kv_cache_config()
    vllm_config = _vllm_config(
        {
            "mode": "embedded",
            "backend": "memory",
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
        load_plans=[plan],
        load_requests={"request": [plan]},
    )
    worker.bind_connector_metadata(metadata)
    worker.start_load_kv(None)
    worker.wait_for_layer_load("layer0")

    assert worker.get_block_ids_with_load_errors() == {7}


def test_embedded_publish_makes_object_visible_atomically():
    runtime = EmbeddedRuntime.from_config(
        UMBPRuntimeConfig("embedded", {"backend": "memory"})
    )
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
    runtime = EmbeddedRuntime.from_config(
        UMBPRuntimeConfig("embedded", {"backend": "memory"})
    )
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


def test_embedded_partial_tail_round_trip():
    topology = RankTopology()
    descriptor = KVLayoutDescriptor(
        regions=(KVRegion("layer0", 0, 16, 16, 0, block_size=16),),
        topology=topology,
    )
    runtime = EmbeddedRuntime.from_config(
        UMBPRuntimeConfig("embedded", {"backend": "memory"})
    )
    worker = runtime.create_worker_handle("partial-tail", topology, descriptor)
    scheduler = runtime.create_scheduler_handle("partial-tail", topology, descriptor)
    source = torch.arange(32, dtype=torch.uint8).reshape(2, 16)
    destination = torch.zeros((2, 16), dtype=torch.uint8)
    planner = KVLayoutPlanner(descriptor.regions)

    planner.register_kv_caches({"layer0": source})
    store_plan = planner.plan_registered_block(
        "partial-tail-key",
        block_id=0,
        token_start=0,
        token_end=12,
    )
    worker.register_buffers({"layer0": source})
    store_job = worker.store([store_plan])
    worker.publish(worker.wait(store_job))
    assert scheduler.lookup(["partial-tail-key"]) == [True]

    planner.register_kv_caches({"layer0": destination})
    load_plan = planner.plan_registered_block(
        "partial-tail-key",
        block_id=1,
        token_start=0,
        token_end=12,
    )
    worker.register_buffers({"layer0": destination})
    load_result = worker.wait(worker.load([load_plan]))

    assert load_result.status.value == "completed"
    assert torch.equal(source[0, :12], destination[1, :12])
    assert torch.equal(destination[0], torch.zeros(16, dtype=torch.uint8))
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


def test_mori_store_cancellation_waits_before_source_reuse(tmp_path):
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
            self.started.set()
            assert self.release.wait(5)
            return [True] * len(keys)

        def batch_get_ranges_into_ptr(self, keys, *args):
            return [False] * len(keys)

        def flush(self):
            return True

        def clear(self):
            return True

        def close(self):
            pass

    client = _Client()
    source = torch.zeros(1024, dtype=torch.uint8)
    handle = _MoriWorkerHandle(
        client,
        "cancel-reuse",
        RankTopology(),
        str(tmp_path),
        1,
        5,
    )
    handle.register_buffers({"layer0": source})
    job = handle.store(
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
    assert client.started.wait(5)
    result = []
    cancelled = threading.Thread(target=lambda: result.append(handle.cancel(job)))
    cancelled.start()
    cancelled.join(0.1)
    assert cancelled.is_alive()

    client.release.set()
    cancelled.join(5)
    assert not cancelled.is_alive()
    assert result[0].status is TransferJobStatus.COMPLETED

    source.fill_(1)
    handle.close()
