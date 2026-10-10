# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import enum
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tests.v1.kv_connector.umbp_test_utils import _kv_cache_config, _vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.connector import (
    UMBPStoreConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.data import (
    TRANSFER_NOT_ATTEMPTED,
    BlockTransferPlan,
    KVLayoutDescriptor,
    KVRange,
    KVRegion,
    RankTopology,
    TransferJobState,
    TransferJobStatus,
    UMBPConnectorMetadata,
    UMBPConnectorWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime import (
    StandaloneRuntime,
    UMBPRuntimeConfig,
    UMBPRuntimeFactory,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime.standalone import (
    normalize_endpoint,
)

ENDPOINT = "/run/umbp/test.grpc.sock"


class _DeploymentMode(enum.Enum):
    Local = 0
    StandaloneProcess = 1
    Distributed = 2


class _Server:
    """One host-wide pool shared by every fake client."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.up = True
        self.accept_puts = True
        self.clients: list[_Client] = []

    def check(self) -> None:
        if not self.up:
            raise RuntimeError("standalone server unavailable")


class _Client:
    def __init__(self, server: _Server, config) -> None:
        server.check()
        self.server = server
        self.config = config
        self.mode = _DeploymentMode.StandaloneProcess
        self.registered: dict[int, int] = {}
        self.flush_ok = True
        self.closed = False
        server.clients.append(self)

    def get_deployment_mode(self):
        return self.mode

    def register_memory(self, ptr, size, location, device):
        self.server.check()
        self.registered[ptr] = size
        return True

    def deregister_memory(self, ptr):
        self.server.check()
        self.registered.pop(ptr, None)
        return True

    def batch_exists(self, keys):
        self.server.check()
        return [key in self.server.objects for key in keys]

    def batch_put_ranges_from_ptr(self, keys, object_sizes, pointers, sizes, offsets):
        self.server.check()
        if not self.server.accept_puts:
            return [False] * len(keys)
        for key, size, ptrs, lengths, offs in zip(
            keys, object_sizes, pointers, sizes, offsets, strict=True
        ):
            payload = bytearray(size)
            for ptr, length, offset in zip(ptrs, lengths, offs, strict=True):
                payload[offset : offset + length] = ctypes.string_at(ptr, length)
            self.server.objects.setdefault(key, bytes(payload))
        return [True] * len(keys)

    def batch_get_ranges_into_ptr(self, keys, pointers, sizes, offsets):
        self.server.check()
        results = []
        for key, ptrs, lengths, offs in zip(
            keys, pointers, sizes, offsets, strict=True
        ):
            payload = self.server.objects.get(key)
            if payload is None:
                results.append(False)
                continue
            for ptr, length, offset in zip(ptrs, lengths, offs, strict=True):
                ctypes.memmove(ptr, payload[offset : offset + length], length)
            results.append(True)
        return results

    def flush(self):
        self.server.check()
        return self.flush_ok

    def clear(self):
        self.server.check()
        self.server.objects.clear()
        return True

    def close(self):
        self.closed = True


@pytest.fixture
def server(monkeypatch) -> _Server:
    pool = _Server()
    cpp = SimpleNamespace(
        UMBPClient=lambda config: _Client(pool, config),
        UMBPConfig=lambda: SimpleNamespace(dram=SimpleNamespace()),
        UMBPStandaloneProcessConfig=SimpleNamespace,
        UMBPDeploymentMode=_DeploymentMode,
        MemoryLocationType=SimpleNamespace(CPU="cpu", GPU="gpu"),
    )
    mori = sys.modules.get("mori") or ModuleType("mori")
    monkeypatch.setitem(sys.modules, "mori", mori)
    monkeypatch.setitem(sys.modules, "mori.cpp", cpp)
    monkeypatch.setattr(mori, "cpp", cpp, raising=False)
    return pool


def _runtime(**options) -> StandaloneRuntime:
    return StandaloneRuntime(
        UMBPRuntimeConfig("standalone", {"endpoint": ENDPOINT, **options})
    )


def _standalone_config(**options):
    return _vllm_config(
        {
            "mode": "standalone",
            "endpoint": ENDPOINT,
            "load_async": False,
            "key_namespace": "standalone-test",
            **options,
        }
    )


def _caches(fill: bool) -> dict[str, torch.Tensor]:
    caches = {
        name: torch.empty_strided((8, 2, 16, 8), (512, 256, 8, 1), dtype=torch.float16)
        for name in ("layer1", "layer2")
    }
    for index, cache in enumerate(caches.values()):
        if fill:
            cache.copy_(
                torch.arange(cache.numel(), dtype=torch.float16).reshape(cache.shape)
                + index
            )
        else:
            cache.zero_()
    return caches


def _request(request_id: str, block_hashes: list[bytes]) -> SimpleNamespace:
    # One token past the two cached blocks, which the scheduler must still run.
    return SimpleNamespace(
        request_id=request_id,
        req_id=request_id,
        num_tokens=33,
        block_hashes=block_hashes,
        block_ids=([1, 2],),
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


def _engine(vllm_config):
    kv_config = _kv_cache_config()
    scheduler = UMBPStoreConnector(vllm_config, KVConnectorRole.SCHEDULER, kv_config)
    worker = UMBPStoreConnector(vllm_config, KVConnectorRole.WORKER, kv_config)
    return scheduler, worker


def _store(scheduler, worker, request) -> UMBPConnectorWorkerMetadata:
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: ([1, 2],)),
        0,
    )
    worker.bind_connector_metadata(
        scheduler.build_connector_meta(_scheduler_output(request))
    )
    worker.wait_for_save()
    worker.connector_worker._drain_store_jobs(wait=True)
    metadata = worker.build_connector_worker_meta()
    scheduler.update_connector_output(
        SimpleNamespace(kv_connector_worker_meta=metadata, kv_cache_events=None)
    )
    return metadata


def _load(scheduler, worker, request) -> None:
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: ([5, 6],)),
        32,
    )
    worker.bind_connector_metadata(
        scheduler.build_connector_meta(_scheduler_output(request))
    )
    worker.start_load_kv(None)
    worker.wait_for_layer_load("layer0")


@pytest.mark.parametrize(
    ("endpoint", "address"),
    [
        ("/run/umbp/a.grpc.sock", "unix:///run/umbp/a.grpc.sock"),
        ("unix:///run/umbp/a.grpc.sock", "unix:///run/umbp/a.grpc.sock"),
    ],
)
def test_endpoint_normalizes_to_a_unix_address(endpoint, address):
    assert normalize_endpoint(endpoint) == address


@pytest.mark.parametrize("endpoint", ["tcp://127.0.0.1:5555", "run/umbp/a.sock"])
def test_endpoint_rejects_non_socket_paths(endpoint):
    with pytest.raises(ValueError, match="endpoint"):
        normalize_endpoint(endpoint)


def _build(vllm_config):
    return UMBPRuntimeFactory.build(UMBPRuntimeConfig.from_vllm(vllm_config))


@pytest.mark.parametrize(
    "option",
    [
        {"capacity_bytes": 1024},
        {"total_capacity_bytes": 1024},
        {"dram_prefault": False},
        {"auto_start": True},
    ],
)
def test_config_rejects_options_the_server_owns(option):
    with pytest.raises(ValueError, match="owns its DRAM pool"):
        _build(_standalone_config(**option))


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"startup_timeout_ms": 0}, "startup_timeout_ms"),
        ({"master_address": "x"}, "distributed-only"),
        ({"endpoint": "tcp://127.0.0.1:5555"}, "endpoint"),
    ],
)
def test_config_rejects_invalid_standalone_options(options, match):
    with pytest.raises(ValueError, match=match):
        _build(_standalone_config(**options))


def test_factory_builds_the_standalone_runtime():
    runtime = _build(_standalone_config())
    assert isinstance(runtime, StandaloneRuntime)


def test_client_targets_the_server_without_sizing_a_local_pool(server):
    runtime = _runtime(startup_timeout_ms=500)
    handle = runtime.create_scheduler_handle("ns", RankTopology(), None)

    config = server.clients[0].config
    assert config.standalone_process.address == "unix://" + ENDPOINT
    assert config.standalone_process.auto_start is False
    assert config.standalone_process.startup_timeout_ms == 500
    assert vars(config.dram) == {}
    handle.close()
    assert server.clients[0].closed


def test_unreachable_server_fails_at_startup(server):
    server.up = False
    runtime = _runtime()
    with pytest.raises(RuntimeError, match="cannot attach"):
        runtime.create_scheduler_handle("ns", RankTopology(), None)


def test_client_in_the_wrong_deployment_mode_is_rejected(server, monkeypatch):
    original = sys.modules["mori.cpp"].UMBPClient

    def local_client(config):
        client = original(config)
        client.mode = _DeploymentMode.Local
        return client

    monkeypatch.setattr(sys.modules["mori.cpp"], "UMBPClient", local_client)
    runtime = _runtime()
    with pytest.raises(RuntimeError, match="expected StandaloneProcess"):
        runtime.create_worker_handle("ns", RankTopology(), None)


def test_worker_does_not_serve_a_lookup_socket(server):
    runtime = _runtime()
    worker = runtime.create_worker_handle("ns", RankTopology(), None)
    cache = torch.zeros(64, dtype=torch.uint8)

    worker.register_buffers({"layer0": cache})

    assert worker._lookup_server is None
    assert server.clients[0].registered == {
        cache.untyped_storage().data_ptr(): cache.untyped_storage().nbytes()
    }
    worker.close()
    assert server.clients[0].registered == {}


def test_close_tolerates_a_server_that_is_already_gone(server):
    runtime = _runtime()
    worker = runtime.create_worker_handle("ns", RankTopology(), None)
    scheduler = runtime.create_scheduler_handle("ns", RankTopology(), None)
    worker.register_buffers({"layer0": torch.zeros(64, dtype=torch.uint8)})
    server.up = False

    worker.close()
    scheduler.close()

    assert all(client.closed for client in server.clients)


def test_failed_clear_reports_false(server, monkeypatch):
    runtime = _runtime()
    handle = runtime.create_scheduler_handle("ns", RankTopology(), None)
    monkeypatch.setattr(server.clients[0], "clear", lambda: False)

    assert handle.clear() is False


def test_engines_share_objects_through_one_server(server):
    producer_scheduler, producer_worker = _engine(_standalone_config())
    consumer_scheduler, consumer_worker = _engine(_standalone_config())
    source = _caches(fill=True)
    destination = _caches(fill=False)
    producer_worker.register_kv_caches(source)
    consumer_worker.register_kv_caches(destination)
    hashes = [b"shared-a", b"shared-b"]

    _store(producer_scheduler, producer_worker, _request("producer", hashes))
    consumer = _request("consumer", hashes)
    assert consumer_scheduler.get_num_new_matched_tokens(consumer, 0) == (32, False)
    _load(consumer_scheduler, consumer_worker, consumer)

    for name in source:
        assert torch.equal(source[name][1], destination[name][5])
        assert torch.equal(source[name][2], destination[name][6])
    assert consumer_worker.get_block_ids_with_load_errors() == set()


def test_reset_cache_clears_the_pool_for_every_engine(server):
    scheduler_a, worker_a = _engine(_standalone_config())
    scheduler_b, _ = _engine(_standalone_config())
    worker_a.register_kv_caches(_caches(fill=True))
    hashes = [b"reset-a", b"reset-b"]
    _store(scheduler_a, worker_a, _request("producer", hashes))
    assert scheduler_b.get_num_new_matched_tokens(_request("r1", hashes), 0) == (
        32,
        False,
    )

    assert scheduler_a.reset_cache() is True

    assert scheduler_b.get_num_new_matched_tokens(_request("r2", hashes), 0) == (
        0,
        False,
    )


def test_lookup_degrades_to_a_miss_when_the_server_is_gone(server):
    runtime = _runtime()
    handle = runtime.create_scheduler_handle("ns", RankTopology(), None)
    server.objects["k"] = b"x"
    assert handle.lookup(["k"]) == [True]

    server.up = False

    assert handle.lookup(["k", "j"]) == [False, False]
    assert handle.clear() is False


def test_lookup_rejects_a_short_result(server, monkeypatch):
    runtime = _runtime()
    handle = runtime.create_scheduler_handle("ns", RankTopology(), None)
    monkeypatch.setattr(server.clients[0], "batch_exists", lambda keys: [True])

    assert handle.lookup(["a", "b"]) == [False, False]


def test_load_from_a_dead_server_marks_blocks_for_recompute(server):
    _, worker = _engine(_standalone_config())
    worker.register_kv_caches(_caches(fill=False))
    worker_impl = worker.connector_worker
    plan = worker_impl.layout.plan_registered_block("gone-key", 7)
    server.up = False

    worker.bind_connector_metadata(UMBPConnectorMetadata(load_requests={"r": [plan]}))
    worker.start_load_kv(None)
    worker.wait_for_layer_load("layer0")

    assert worker.get_block_ids_with_load_errors() == {7}


@pytest.mark.parametrize("failure", ["server-down", "puts-rejected"])
def test_failed_store_is_not_reported_as_a_load_error(server, failure):
    # vLLM fails or recomputes requests owning blocks reported as load errors.
    # A failed store leaves the source KV intact, so it must not report any.
    scheduler, worker = _engine(_standalone_config())
    worker.register_kv_caches(_caches(fill=True))
    if failure == "server-down":
        server.up = False
    else:
        server.accept_puts = False

    metadata = _store(scheduler, worker, _request("producer", [b"dead-a", b"dead-b"]))

    assert worker.get_block_ids_with_load_errors() == set()
    (result,) = metadata.store_events.values()
    assert result.completed_workers == 1


def test_publish_failure_is_not_raised(server):
    runtime = _runtime()
    topology = RankTopology()
    layout = KVLayoutDescriptor(
        regions=(KVRegion("layer0", 0, 16, 16, 0),), topology=topology
    )
    worker = runtime.create_worker_handle("ns", topology, layout)
    source = torch.arange(16, dtype=torch.uint8)
    worker.register_buffers({"layer0": source})
    plan = BlockTransferPlan(
        "publish-key",
        0,
        ranges=(KVRange("layer0", 0, 0, source.data_ptr(), 16, 16, 0),),
    )
    job = worker.wait(worker.store([plan]))
    server.clients[0].flush_ok = False

    worker.publish(job)

    assert "publish-key" in server.objects


def _stalled_store(server, monkeypatch, release: threading.Event):
    client = server.clients[0]
    put = client.batch_put_ranges_from_ptr

    def blocking_put(*args):
        release.wait(10)
        return put(*args)

    monkeypatch.setattr(client, "batch_put_ranges_from_ptr", blocking_put)


def _block_plan(key: str, source: torch.Tensor) -> BlockTransferPlan:
    return BlockTransferPlan(
        key,
        0,
        ranges=(KVRange("layer0", 0, 0, source.data_ptr(), 16, 16, 0),),
    )


def test_stalled_store_stays_pending_and_refuses_new_transfers(server, monkeypatch):
    worker = _runtime(timeout_ms=20).create_worker_handle("ns", RankTopology(), None)
    source = torch.arange(16, dtype=torch.uint8)
    worker.register_buffers({"layer0": source})
    release = threading.Event()
    _stalled_store(server, monkeypatch, release)

    job = worker.store([_block_plan("slow", source)])
    time.sleep(0.05)

    # MORI still reads the source, so the job must not settle as failed.
    assert worker.poll(job) is None
    refused = worker.store([_block_plan("next", source)])
    assert refused.status == TransferJobStatus.FAILED
    assert refused.error.startswith(TRANSFER_NOT_ATTEMPTED)
    refused = worker.store_blocks([BlockTransferPlan("next", 0, group_id=0)])
    assert refused is not None and refused.status == TransferJobStatus.FAILED
    load = worker.load([_block_plan("slow", source)])
    assert load.status == TransferJobStatus.FAILED

    release.set()
    deadline = time.monotonic() + 5
    result = worker.poll(job)
    while result is None and time.monotonic() < deadline:
        time.sleep(0.01)
        result = worker.poll(job)
    assert result is not None and result.status == TransferJobStatus.COMPLETED
    assert "slow" in server.objects
    assert worker.store([_block_plan("after", source)]).status in (
        TransferJobStatus.RUNNING,
        TransferJobStatus.COMPLETED,
    )
    worker.close()


def test_close_does_not_wait_for_a_stalled_store(server, monkeypatch):
    worker = _runtime(timeout_ms=20).create_worker_handle("ns", RankTopology(), None)
    source = torch.arange(16, dtype=torch.uint8)
    worker.register_buffers({"layer0": source})
    release = threading.Event()
    _stalled_store(server, monkeypatch, release)
    worker.store([_block_plan("slow", source)])

    started = time.monotonic()
    worker.close()

    assert time.monotonic() - started < 2
    release.set()


def test_publish_rejects_an_incomplete_job(server):
    runtime = _runtime()
    worker = runtime.create_worker_handle("ns", RankTopology(), None)
    job = TransferJobState((BlockTransferPlan("k", 0),))

    with pytest.raises(RuntimeError, match="incomplete"):
        worker.publish(job)


def test_standalone_connector_rejects_pipeline_parallelism(server):
    with pytest.raises(NotImplementedError, match="pipeline parallelism"):
        UMBPStoreConnector(
            _vllm_config(
                {"mode": "standalone", "endpoint": ENDPOINT},
                pipeline_parallel_size=2,
            ),
            KVConnectorRole.SCHEDULER,
            _kv_cache_config(),
        )
