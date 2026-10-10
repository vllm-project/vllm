# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import enum
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.connector import (
    UMBPStoreConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.data import (
    TRANSFER_NOT_ATTEMPTED,
    BlockIdentityCodec,
    KVLayoutDescriptor,
    KVRegion,
    RankTopology,
    UMBPConnectorMetadata,
    UMBPConnectorWorkerMetadata,
    UMBPNamespace,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime import (
    DistributedRuntime,
    UMBPRuntimeConfig,
    UMBPRuntimeFactory,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime import (
    distributed as distributed_runtime,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.scheduler import (
    UMBPStoreConnectorScheduler,
)

from .test_umbp_shared import _kv_cache_config, _SchedulerHandle, _vllm_config

MASTER = "10.0.0.1:15558"
BASE_OPTIONS = {"master_address": MASTER, "peer_service_port": 17000}


def _dist(options: dict, rank_count: int = 1) -> DistributedRuntime:
    return DistributedRuntime(UMBPRuntimeConfig("distributed", options, rank_count))


def _build(vllm_config):
    return UMBPRuntimeFactory.build(UMBPRuntimeConfig.from_vllm(vllm_config))


class _DeploymentMode(enum.Enum):
    Local = 0
    StandaloneProcess = 1
    Distributed = 2


class _Master:
    """Cluster-wide index; objects stay in the pool of the node that wrote them."""

    def __init__(self) -> None:
        self.up = True
        self.nodes: dict[str, _Client] = {}
        self.index: dict[str, str] = {}
        self.clients: list[_Client] = []

    def check(self) -> None:
        if not self.up:
            raise RuntimeError("master unavailable")


class _Client:
    def __init__(self, master: _Master, config) -> None:
        dist = config.distributed
        self.master = master
        self.config = config
        self.node_id = dist.master_config.node_id
        self.mode = _DeploymentMode.Distributed
        self.registered: dict[int, int] = {}
        self.objects: dict[str, bytes] = {}
        self.unpublished: set[str] = set()
        self.alive = True
        if not master.up:
            raise RuntimeError("DistributedClient: PoolClient::Init() failed")
        if self.node_id in master.nodes:
            raise RuntimeError("node is already alive and cannot be re-registered")
        master.nodes[self.node_id] = self
        master.clients.append(self)

    def get_deployment_mode(self):
        return self.mode

    def register_memory(self, ptr, size, location, device):
        self.registered[ptr] = size
        return True

    def deregister_memory(self, ptr):
        self.registered.pop(ptr, None)

    def batch_exists(self, keys):
        # MORI answers from local media first and reports a master RPC failure
        # as all-miss for the rest instead of raising.
        local = [key in self.objects for key in keys]
        if not self.master.up:
            return local
        return [
            hit or self.master.index.get(key) is not None
            for key, hit in zip(keys, local, strict=True)
        ]

    def batch_put_ranges_from_ptr(self, keys, object_sizes, pointers, sizes, offsets):
        if not self.master.up:
            return [False] * len(keys)
        results = []
        for key, size, ptrs, lengths, offs in zip(
            keys, object_sizes, pointers, sizes, offsets, strict=True
        ):
            if key in self.master.index:
                results.append(True)
                continue
            payload = bytearray(size)
            for ptr, length, offset in zip(ptrs, lengths, offs, strict=True):
                payload[offset : offset + length] = ctypes.string_at(ptr, length)
            self.objects[key] = bytes(payload)
            self.unpublished.add(key)
            results.append(True)
        return results

    def batch_get_ranges_into_ptr(self, keys, pointers, sizes, offsets):
        results = []
        for key, ptrs, lengths, offs in zip(
            keys, pointers, sizes, offsets, strict=True
        ):
            payload = self.objects.get(key)
            if payload is None and self.master.up:
                owner = self.master.nodes.get(self.master.index.get(key, ""))
                if owner is not None and owner.alive:
                    payload = owner.objects.get(key)
            if payload is None:
                results.append(False)
                continue
            for ptr, length, offset in zip(ptrs, lengths, offs, strict=True):
                ctypes.memmove(ptr, payload[offset : offset + length], length)
            results.append(True)
        return results

    def flush(self):
        # A heartbeat flush; MORI reports success even when the master is down.
        if self.master.up:
            for key in self.unpublished:
                self.master.index.setdefault(key, self.node_id)
            self.unpublished.clear()
        return True

    def clear(self):
        self.objects.clear()
        return True

    def crash(self) -> None:
        """Die without unregistering, leaving stale index entries behind."""
        self.alive = False


@pytest.fixture
def master(monkeypatch) -> _Master:
    cluster = _Master()
    cpp: Any = ModuleType("mori.cpp")
    cpp.UMBPClient = lambda config: _Client(cluster, config)
    cpp.UMBPConfig = lambda: SimpleNamespace(dram=SimpleNamespace())
    cpp.UMBPDistributedConfig = lambda: SimpleNamespace(
        master_config=SimpleNamespace(), io_engine=SimpleNamespace()
    )
    cpp.UMBPDeploymentMode = _DeploymentMode
    cpp.MemoryLocationType = SimpleNamespace(CPU="cpu", GPU="gpu")
    mori = sys.modules.get("mori") or ModuleType("mori")
    monkeypatch.setitem(sys.modules, "mori", mori)
    monkeypatch.setitem(sys.modules, "mori.cpp", cpp)
    monkeypatch.setattr(mori, "cpp", cpp, raising=False)
    monkeypatch.setattr(distributed_runtime, "_physical_device_index", lambda: 0)
    return cluster


def _distributed_config(**options):
    return _vllm_config(
        {
            "mode": "distributed",
            **BASE_OPTIONS,
            "node_address": "10.0.0.2",
            "capacity_bytes": 1 << 20,
            "load_async": False,
            "key_namespace": "distributed-test",
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
        SimpleNamespace(kv_connector_worker_meta=metadata)
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


def _layout(object_size: int) -> KVLayoutDescriptor:
    return KVLayoutDescriptor(
        regions=(KVRegion("layer0", 0, object_size, object_size, 0),),
        topology=RankTopology(),
    )


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({}, "master_address"),
        ({"master_address": "10.0.0.1"}, "master_address"),
        ({"master_address": "10.0.0.1:0"}, "master_address"),
        ({"master_address": MASTER}, "peer_service_port"),
        ({**BASE_OPTIONS, "peer_service_port": 70000}, "peer_service_port"),
        ({**BASE_OPTIONS, "io_engine_port": "16000"}, "io_engine_port"),
        ({**BASE_OPTIONS, "node_address": ""}, "node_address"),
        ({**BASE_OPTIONS, "dram_page_size": 0}, "dram_page_size"),
        ({**BASE_OPTIONS, "local_first": "yes"}, "local_first"),
        ({**BASE_OPTIONS, "dram_high_watermark": 0.9}, "master decides eviction"),
        ({**BASE_OPTIONS, "dram_use_shared_memory": True}, "does not use"),
        ({**BASE_OPTIONS, "dram_prefault": "no"}, "dram_prefault"),
    ],
)
def test_config_rejects_invalid_distributed_options(options, match):
    with pytest.raises(ValueError, match=match):
        _build(_vllm_config({"mode": "distributed", **options}))


@pytest.mark.parametrize("mode", ["embedded", "standalone"])
@pytest.mark.parametrize("option", ["peer_service_port", "dram_page_size", "node_id"])
def test_local_modes_reject_distributed_only_options(mode, option):
    extra = {"mode": mode, option: 1}
    if mode == "standalone":
        extra["endpoint"] = "/run/umbp.sock"
    with pytest.raises(ValueError):
        _build(_vllm_config(extra))


def test_total_capacity_is_split_across_the_engine_ranks():
    runtime = _dist({**BASE_OPTIONS, "total_capacity_bytes": 8}, rank_count=4)
    assert runtime._options["capacity_bytes"] == 2


def test_factory_builds_the_distributed_runtime():
    runtime = _build(_vllm_config({"mode": "distributed", **BASE_OPTIONS}))
    assert isinstance(runtime, DistributedRuntime)
    assert runtime.capabilities.layerwise_load
    assert runtime.capabilities.partial_hash_hits


def test_scheduler_client_owns_no_pool_and_serves_nothing(master):
    runtime = _dist({**BASE_OPTIONS, "node_address": "10.0.0.2"})
    handle = runtime.create_scheduler_handle("ns", RankTopology(), None)

    config = master.clients[0].config
    dist = config.distributed
    assert config.dram.capacity_bytes == 0
    assert dist.master_config.master_address == MASTER
    assert dist.master_config.node_address == "10.0.0.2"
    assert "-lookup-" in dist.master_config.node_id
    assert not hasattr(dist.io_engine, "host")
    assert not hasattr(dist, "peer_service_port")
    assert not hasattr(dist, "ranged_scratch_size")
    assert dist.cache_remote_fetches is False
    handle.close()


def test_worker_binds_ports_offset_by_its_physical_gpu(master, monkeypatch):
    monkeypatch.setattr(distributed_runtime, "_physical_device_index", lambda: 3)
    runtime = _dist(
        {
            **BASE_OPTIONS,
            "io_engine_port": 16000,
            "node_address": "10.0.0.2",
            "capacity_bytes": 4096,
            "dram_page_size": 65536,
            "cache_remote_fetches": False,
            "backend_policy_path": "/etc/umbp/policy.json",
        }
    )
    runtime.create_worker_handle("ns", RankTopology(), None)

    config = master.clients[0].config
    dist = config.distributed
    assert config.dram.capacity_bytes == 4096
    assert dist.peer_service_port == 17003
    assert dist.io_engine.port == 16003
    assert dist.io_engine.host == "10.0.0.2"
    assert dist.ranged_scratch_size == distributed_runtime.DEFAULT_RANGED_SCRATCH_BYTES
    assert dist.dram_page_size == 65536
    assert dist.cache_remote_fetches is False
    assert dist.backend_policy_path == "/etc/umbp/policy.json"
    assert "-gpu3-" in dist.master_config.node_id


def test_worker_defaults_to_this_host_address_and_a_free_io_port(master, monkeypatch):
    import vllm.utils.network_utils as network_utils

    monkeypatch.setattr(network_utils, "get_ip", lambda: "10.9.9.9")
    _dist(dict(BASE_OPTIONS)).create_worker_handle("ns", RankTopology(), None)

    dist = master.clients[0].config.distributed
    assert dist.master_config.node_address == "10.9.9.9"
    assert dist.io_engine.host == "10.9.9.9"
    assert dist.io_engine.port == 0


def test_every_client_registers_a_distinct_identity(master):
    runtime = _dist(dict(BASE_OPTIONS, node_address="10.0.0.2"))
    runtime.create_worker_handle("ns", RankTopology(), None)
    runtime.create_worker_handle("ns", RankTopology(), None)
    runtime.create_scheduler_handle("ns", RankTopology(), None)

    assert len(master.nodes) == 3


def test_scratch_smaller_than_one_object_fails_at_startup(master):
    runtime = _dist(dict(BASE_OPTIONS, ranged_scratch_size=1024))
    with pytest.raises(ValueError, match="ranged_scratch_size"):
        runtime.create_worker_handle("ns", RankTopology(), _layout(4096))
    assert master.clients == []


@pytest.mark.parametrize(
    ("options", "object_size", "expected"),
    [
        ({"capacity_bytes": 1 << 20}, 3 << 20, ["padding"]),
        ({"capacity_bytes": 1 << 20, "dram_page_size": 1 << 20}, 3 << 20, []),
        ({"capacity_bytes": 4 << 30}, 2 << 20, ["hugepages"]),
        ({"capacity_bytes": 4 << 30, "dram_use_hugepages": True}, 2 << 20, []),
    ],
)
def test_pool_sizing_warnings(master, monkeypatch, options, object_size, expected):
    warnings: list[str] = []
    monkeypatch.setattr(
        distributed_runtime.logger,
        "warning",
        lambda message, *args: warnings.append(message % args),
    )
    runtime = _dist(dict(BASE_OPTIONS, node_address="10.0.0.2", **options))
    runtime.create_worker_handle("ns", RankTopology(), _layout(object_size))

    assert len(warnings) == len(expected)
    for warning, word in zip(warnings, expected):
        assert word in warning


def test_unreachable_master_fails_at_startup(master):
    master.up = False
    runtime = _dist(dict(BASE_OPTIONS, node_address="10.0.0.2"))
    with pytest.raises(RuntimeError, match="cannot join the UMBP master"):
        runtime.create_scheduler_handle("ns", RankTopology(), None)


def test_client_in_the_wrong_deployment_mode_is_rejected(master, monkeypatch):
    original = sys.modules["mori.cpp"].UMBPClient

    def local_client(config):
        client = original(config)
        client.mode = _DeploymentMode.Local
        return client

    monkeypatch.setattr(sys.modules["mori.cpp"], "UMBPClient", local_client)
    runtime = _dist(dict(BASE_OPTIONS, node_address="10.0.0.2"))
    with pytest.raises(RuntimeError, match="expected Distributed"):
        runtime.create_worker_handle("ns", RankTopology(), None)


def test_engines_share_objects_across_nodes(master):
    producer_scheduler, producer_worker = _engine(_distributed_config())
    consumer_scheduler, consumer_worker = _engine(
        _distributed_config(node_address="10.0.0.3")
    )
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
    owners = {master.index[key] for key in master.index}
    assert owners == {producer_worker.connector_worker.runtime.client.node_id}


def test_reset_cache_leaves_the_shared_pool_alone(master):
    scheduler_a, worker_a = _engine(_distributed_config())
    scheduler_b, _ = _engine(_distributed_config())
    worker_a.register_kv_caches(_caches(fill=True))
    hashes = [b"reset-a", b"reset-b"]
    _store(scheduler_a, worker_a, _request("producer", hashes))

    assert scheduler_a.reset_cache() is False

    assert scheduler_b.get_num_new_matched_tokens(_request("r", hashes), 0) == (
        32,
        False,
    )


def test_master_outage_degrades_to_recomputation(master):
    scheduler_a, worker_a = _engine(_distributed_config())
    scheduler_b, worker_b = _engine(_distributed_config())
    worker_a.register_kv_caches(_caches(fill=True))
    worker_b.register_kv_caches(_caches(fill=False))
    hashes = [b"outage-a", b"outage-b"]
    _store(scheduler_a, worker_a, _request("producer", hashes))
    request = _request("consumer", hashes)
    assert scheduler_b.get_num_new_matched_tokens(request, 0) == (32, False)

    master.up = False

    assert scheduler_b.get_num_new_matched_tokens(_request("r2", hashes), 0) == (
        0,
        False,
    )
    _load(scheduler_b, worker_b, request)
    assert worker_b.get_block_ids_with_load_errors() == {5, 6}


def test_store_during_a_master_outage_is_not_a_load_error(master):
    scheduler, worker = _engine(_distributed_config())
    worker.register_kv_caches(_caches(fill=True))
    master.up = False

    metadata = _store(scheduler, worker, _request("producer", [b"down-a", b"down-b"]))

    assert metadata.failed_block_ids == set()
    assert worker.get_block_ids_with_load_errors() == set()
    (result,) = metadata.store_events.values()
    assert result.failed_tokens


def _crashed_producer_and_reader(**options):
    scheduler_a, worker_a = _engine(_distributed_config())
    scheduler_b, worker_b = _engine(_distributed_config(**options))
    worker_a.register_kv_caches(_caches(fill=True))
    worker_b.register_kv_caches(_caches(fill=False))
    hashes = [b"dead-a", b"dead-b"]
    _store(scheduler_a, worker_a, _request("producer", hashes))
    worker_a.connector_worker.runtime.client.crash()
    return scheduler_b, worker_b, hashes


def _report_worker_output(scheduler, worker) -> None:
    scheduler.update_connector_output(
        SimpleNamespace(kv_connector_worker_meta=worker.build_connector_worker_meta())
    )


def test_objects_of_a_dead_node_fail_the_load_for_recompute(master):
    scheduler, worker, hashes = _crashed_producer_and_reader()

    request = _request("consumer", hashes)
    # The master still lists the objects until it reaps the dead node.
    assert scheduler.get_num_new_matched_tokens(request, 0) == (32, False)
    _load(scheduler, worker, request)

    assert worker.get_block_ids_with_load_errors() == {5, 6}


def test_a_failed_load_is_not_retried_while_quarantined(master, monkeypatch):
    from vllm.distributed.kv_transfer.kv_connector.v1.umbp import (
        scheduler as scheduler_module,
    )

    now = [1000.0]
    monkeypatch.setattr(scheduler_module.time, "monotonic", lambda: now[0])
    scheduler, worker, hashes = _crashed_producer_and_reader(
        load_failure_quarantine_ms=5000
    )
    request = _request("consumer", hashes)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (32, False)
    _load(scheduler, worker, request)
    _report_worker_output(scheduler, worker)

    # The rescheduled request recomputes instead of loading the dead copy again.
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    now[0] += 6
    assert scheduler.get_num_new_matched_tokens(request, 0) == (32, False)


def test_quarantine_can_be_disabled(master):
    scheduler, worker, hashes = _crashed_producer_and_reader(
        load_failure_quarantine_ms=0
    )
    request = _request("consumer", hashes)
    scheduler.get_num_new_matched_tokens(request, 0)
    _load(scheduler, worker, request)
    _report_worker_output(scheduler, worker)

    assert scheduler.get_num_new_matched_tokens(request, 0) == (32, False)


def test_a_refused_load_is_not_quarantined(master):
    scheduler, worker, hashes = _crashed_producer_and_reader()
    request = _request("consumer", hashes)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (32, False)
    keys = scheduler.connector_scheduler._build_lookup_context(
        request, 0, tuple(hashes)
    ).keys
    scheduler.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=UMBPConnectorWorkerMetadata(
                failed_loads={key: f"{TRANSFER_NOT_ATTEMPTED}: blocked" for key in keys}
            )
        )
    )

    assert scheduler.get_num_new_matched_tokens(request, 0) == (32, False)


def test_quarantine_is_off_by_default_outside_distributed_mode():
    codec = BlockIdentityCodec(UMBPNamespace("local-quarantine"))
    hashes = [b"local-a", b"local-b"]
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded"}),
        _kv_cache_config(),
        _SchedulerHandle({codec.key(block_hash, 0): True for block_hash in hashes}),
        codec,
    )
    request = _request("local", hashes)
    scheduler.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=UMBPConnectorWorkerMetadata(
                failed_loads={codec.key(hashes[0], 0): "load failed"}
            )
        )
    )

    assert scheduler.get_num_new_matched_tokens(request, 0) == (32, True)
    scheduler.close()


def test_config_rejects_a_negative_quarantine(master):
    with pytest.raises(ValueError, match="load_failure_quarantine_ms"):
        UMBPStoreConnector(
            _distributed_config(load_failure_quarantine_ms=-1),
            KVConnectorRole.SCHEDULER,
            _kv_cache_config(),
        )


class _Gate:
    """Makes a fake MORI call block until released, like a hung master."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.calls = 0

    def wrap(self, fn):
        def blocked(*args, **kwargs):
            self.calls += 1
            self.release.wait(30)
            return fn(*args, **kwargs)

        return blocked


def test_lookup_against_a_hung_master_is_bounded(master):
    runtime = _dist(dict(BASE_OPTIONS, node_address="10.0.0.2", lookup_timeout_ms=100))
    handle = runtime.create_scheduler_handle("ns", RankTopology(), None)
    master.index["k"] = "elsewhere"
    client = master.clients[0]
    gate = _Gate()
    original = client.batch_exists
    client.batch_exists = gate.wrap(original)

    started = time.monotonic()
    assert handle.lookup(["k"]) == [False]
    assert time.monotonic() - started < 5
    # While that call is blocked, lookups do not queue more calls behind it.
    assert handle.lookup(["k"]) == [False]
    assert gate.calls == 1

    gate.release.set()
    client.batch_exists = original
    deadline = time.monotonic() + 5
    while handle.lookup(["k"]) != [True] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert handle.lookup(["k"]) == [True]
    handle.close()


def test_load_of_a_missing_object_marks_blocks_for_recompute(master):
    _, worker = _engine(_distributed_config())
    worker.register_kv_caches(_caches(fill=False))
    plan = worker.connector_worker.layout.plan_registered_block("gone-key", 7)

    worker.bind_connector_metadata(UMBPConnectorMetadata(load_requests={"r": [plan]}))
    worker.start_load_kv(None)
    worker.wait_for_layer_load("layer0")

    assert worker.get_block_ids_with_load_errors() == {7}


def test_shutdown_releases_both_clients(master):
    scheduler, worker = _engine(_distributed_config())
    cache = _caches(fill=True)
    worker.register_kv_caches(cache)
    worker_client = worker.connector_worker.runtime.client
    assert worker_client.registered

    scheduler.shutdown()
    worker.shutdown()

    assert worker_client.registered == {}
    assert worker.connector_worker.runtime.client is None
    assert scheduler.connector_scheduler.runtime._client is None


def test_distributed_connector_rejects_pipeline_parallelism(master):
    with pytest.raises(NotImplementedError, match="pipeline parallelism"):
        UMBPStoreConnector(
            _vllm_config(
                {"mode": "distributed", **BASE_OPTIONS}, pipeline_parallel_size=2
            ),
            KVConnectorRole.SCHEDULER,
            _kv_cache_config(),
        )
