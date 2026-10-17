# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AsyncExitStack
from functools import partial
from types import SimpleNamespace
from typing import Any, cast

import msgspec
import openai  # use the official client for correctness check
import pytest
import pytest_asyncio
import requests
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from tests.utils import RemoteOpenAIServer
from vllm.distributed.elastic_ep.external_elastic_ep import (
    ExternalElasticEPInstanceMismatch,
    ExternalElasticEPScaleConflict,
    ExternalElasticEPScaleCoordinator,
    ExternalElasticEPScalePhase,
)
from vllm.entrypoints.serve.elastic_ep.api_router import attach_router
from vllm.entrypoints.serve.elastic_ep.middleware import (
    ScalingMiddleware,
    get_scaling_elastic_ep,
    set_elastic_ep_rank_retired,
    set_scaling_elastic_ep,
)
from vllm.platforms import current_platform
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core_client import DPAsyncMPClient

MODEL_NAME = os.getenv("MODEL_NAME", "ibm-research/PowerMoE-3b")
ELASTIC_EP_MODEL_NAME = "deepseek-ai/DeepSeek-V2-Lite-Chat"

# Number of data parallel ranks for external LB testing
DP_SIZE = int(os.getenv("DP_SIZE", "2"))
# Default tensor parallel size to use
TP_SIZE = int(os.getenv("TP_SIZE", "1"))


class DictStore:
    def __init__(self, values: dict[str, bytes]):
        self.values = values
        self.writes: list[str] = []

    def check(self, keys: list[str]) -> bool:
        return all(key in self.values for key in keys)

    def get(self, key: str) -> bytes:
        return self.values[key]

    def set(self, key: str, value: bytes) -> None:
        self.writes.append(key)
        self.values[key] = value


def test_external_elastic_ep_visible_epoch_has_bootstrap_and_phase(monkeypatch):
    client = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=2,
                data_parallel_master_port=12345,
                _data_parallel_master_port_list=[12346, 12347],
            )
        )
    )
    coordinator = ExternalElasticEPScaleCoordinator(cast("DPAsyncMPClient", client))
    control_store, reconfig_store = DictStore({}), DictStore({})
    coordinator.reconfig_store_ref = reconfig_store
    monkeypatch.setattr(
        coordinator, "_setup_reconfig_bootstrap", lambda: ("127.0.0.1", 12348)
    )
    epoch, bootstrap = coordinator._prepare_reconfig_bootstrap(
        control_store, 2, 3, "scale-1"
    )

    for store in (control_store, reconfig_store):
        # DictStore preserves insertion order: metadata must precede the epoch.
        assert list(store.values)[-1] == coordinator.key("current_epoch")
        assert store.get(coordinator.key(epoch, "bootstrap")) == msgspec.msgpack.encode(
            bootstrap
        )
        assert store.get(coordinator.key(epoch, "phase")) == b"preparing"
        assert store.get(coordinator.key("current_epoch")) == epoch.encode()
        monkeypatch.setattr(
            coordinator, "_get_reconfig_store", lambda store=store: store
        )
        assert coordinator.get_status().requested_data_parallel_size == 3


def _external_eep_coordinator(dp_size: int = 2, dp_rank: int = 0):
    client = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_rank=dp_rank,
                data_parallel_size=dp_size,
                data_parallel_external_lb=True,
            )
        )
    )
    return ExternalElasticEPScaleCoordinator(cast("DPAsyncMPClient", client))


def _operation_store(
    coordinator: ExternalElasticEPScaleCoordinator,
    *,
    phase: ExternalElasticEPScalePhase,
    operation_id: str = "scale-1",
    target_dp_size: int = 3,
    effective_dp_size: int = 2,
    topology_known: bool = True,
    error: str | None = None,
) -> DictStore:
    epoch = "epoch-1"
    bootstrap = ReconfigureDistributedRequest(
        new_data_parallel_size=target_dp_size,
        new_data_parallel_rank=ReconfigureRankType.KEEP_CURRENT_RANK,
        new_data_parallel_rank_local=ReconfigureRankType.KEEP_CURRENT_RANK,
        new_data_parallel_master_ip="127.0.0.1",
        new_data_parallel_master_port=1234,
        new_data_parallel_master_port_list=[1234],
        coord_store_port=4321,
    )
    values = {
        coordinator.key("current_epoch"): epoch.encode(),
        coordinator.key(epoch, "bootstrap"): msgspec.msgpack.encode(bootstrap),
        coordinator.key(epoch, "operation_id"): operation_id.encode(),
        coordinator.key(epoch, "effective_dp_size"): str(effective_dp_size).encode(),
        coordinator.key(epoch, "topology_known"): (b"1" if topology_known else b"0"),
        coordinator.key(epoch, "phase"): phase.value.encode(),
    }
    if phase == ExternalElasticEPScalePhase.COMPLETED:
        values[coordinator.key(epoch, "completed")] = b"1"
    if error is not None:
        values[coordinator.key(epoch, "error")] = error.encode()
    return DictStore(values)


def test_external_elastic_ep_status_reports_operation_state(monkeypatch):
    coordinator = _external_eep_coordinator()
    store = _operation_store(
        coordinator,
        phase=ExternalElasticEPScalePhase.COMMITTING,
        target_dp_size=2,
        effective_dp_size=3,
        topology_known=False,
    )
    monkeypatch.setattr(coordinator, "_get_reconfig_store", lambda: store)

    status = coordinator.get_status()

    assert status.operation_id == "scale-1"
    assert status.epoch == "epoch-1"
    assert status.phase == ExternalElasticEPScalePhase.COMMITTING
    assert status.requested_data_parallel_size == 2
    assert status.effective_data_parallel_size == 3
    assert not status.topology_known
    assert status.error is None


def test_external_elastic_ep_status_reports_failed_unknown_topology(monkeypatch):
    coordinator = _external_eep_coordinator()
    store = _operation_store(
        coordinator,
        phase=ExternalElasticEPScalePhase.FAILED,
        topology_known=False,
        error="commit failed",
    )
    monkeypatch.setattr(coordinator, "_get_reconfig_store", lambda: store)

    status = coordinator.get_status()

    assert status.phase == ExternalElasticEPScalePhase.FAILED
    assert status.error == "commit failed"
    assert not status.topology_known


def test_external_elastic_ep_unknown_topology_allows_only_same_operation(monkeypatch):
    coordinator = _external_eep_coordinator()
    store = _operation_store(
        coordinator,
        phase=ExternalElasticEPScalePhase.FAILED,
        topology_known=False,
        error="commit failed",
    )
    monkeypatch.setattr(coordinator, "_get_reconfig_store", lambda: store)

    status = coordinator.validate_request(3, "scale-1", coordinator.instance_id)
    assert status.phase == ExternalElasticEPScalePhase.FAILED

    with pytest.raises(ExternalElasticEPScaleConflict, match="restart or recover"):
        coordinator.validate_request(3, "scale-2", coordinator.instance_id)


def test_external_elastic_ep_bootstrap_rejects_unknown_topology():
    coordinator = _external_eep_coordinator()
    store = _operation_store(
        coordinator,
        phase=ExternalElasticEPScalePhase.FAILED,
        topology_known=False,
        error="commit failed",
    )

    with pytest.raises(ExternalElasticEPScaleConflict, match="restart or recover"):
        coordinator._prepare_reconfig_bootstrap(store, 2, 3, "scale-2")


def test_external_elastic_ep_publishes_evidence_before_completion():
    coordinator = _external_eep_coordinator()
    stores = tuple(
        _operation_store(
            coordinator,
            phase=ExternalElasticEPScalePhase.COMMITTING,
            topology_known=False,
        )
        for _ in range(2)
    )

    coordinator._publish_completed(stores, "epoch-1", 3)

    expected_writes = [
        coordinator.key("epoch-1", "effective_dp_size"),
        coordinator.key("epoch-1", "topology_known"),
        coordinator.key("epoch-1", "phase"),
        coordinator.key("epoch-1", "completed"),
    ]
    for store in stores:
        assert store.writes == expected_writes
        status = coordinator._read_status(store)
        assert status.phase == ExternalElasticEPScalePhase.COMPLETED
        assert status.effective_data_parallel_size == 3
        assert status.topology_known


def test_external_elastic_ep_completion_during_poll_has_current_topology():
    coordinator = _external_eep_coordinator()
    completed_key = coordinator.key("epoch-1", "completed")

    class CompletingStore(DictStore):
        def check(self, keys):
            if completed_key in keys and completed_key not in self.values:
                coordinator._publish_completed((self,), "epoch-1", 3)
            return super().check(keys)

    store = CompletingStore(
        _operation_store(
            coordinator,
            phase=ExternalElasticEPScalePhase.COMMITTING,
            topology_known=False,
        ).values
    )

    status = coordinator._read_status(store)

    assert status.phase == ExternalElasticEPScalePhase.COMPLETED
    assert status.effective_data_parallel_size == 3
    assert status.topology_known


@pytest.mark.parametrize(
    "phase",
    [
        ExternalElasticEPScalePhase.PREPARING,
        ExternalElasticEPScalePhase.COMPLETED,
        ExternalElasticEPScalePhase.FAILED,
    ],
)
def test_external_elastic_ep_accepts_retry_of_same_operation(monkeypatch, phase):
    coordinator = _external_eep_coordinator()
    store = _operation_store(
        coordinator,
        phase=phase,
    )
    monkeypatch.setattr(coordinator, "_get_reconfig_store", lambda: store)

    status = coordinator.validate_request(3, "scale-1", coordinator.instance_id)

    assert status.operation_id == "scale-1"
    assert status.requested_data_parallel_size == 3
    assert status.phase == phase


@pytest.mark.parametrize(
    ("target_dp_size", "operation_id"),
    [(4, "scale-1"), (3, "scale-2")],
)
def test_external_elastic_ep_rejects_conflicting_operation(
    monkeypatch,
    target_dp_size: int,
    operation_id: str,
):
    coordinator = _external_eep_coordinator()
    store = _operation_store(
        coordinator,
        phase=ExternalElasticEPScalePhase.PREPARING,
    )
    monkeypatch.setattr(coordinator, "_get_reconfig_store", lambda: store)

    with pytest.raises(ExternalElasticEPScaleConflict):
        coordinator.validate_request(
            target_dp_size,
            operation_id,
            coordinator.instance_id,
        )


def test_external_elastic_ep_rejects_stale_instance(monkeypatch):
    coordinator = _external_eep_coordinator()
    store = _operation_store(
        coordinator,
        phase=ExternalElasticEPScalePhase.PREPARING,
    )
    monkeypatch.setattr(coordinator, "_get_reconfig_store", lambda: store)

    with pytest.raises(ExternalElasticEPInstanceMismatch):
        coordinator.validate_request(3, "scale-1", "stale-instance")


@pytest.mark.asyncio
async def test_external_elastic_ep_bootstrap_rejects_other_operation():
    coordinator = _external_eep_coordinator(dp_rank=1)
    store = _operation_store(
        coordinator,
        phase=ExternalElasticEPScalePhase.PREPARING,
    )
    store.set(coordinator.key("epoch-1", "prepared"), b"1")

    with pytest.raises(ExternalElasticEPScaleConflict):
        await coordinator._wait_for_bootstrap(
            store,
            requested_new_dp_size=3,
            operation_id="scale-2",
        )


@pytest.mark.asyncio
async def test_external_elastic_ep_commit_failure_keeps_admission_closed(monkeypatch):
    class FailingEngineCore:
        async def prepare_elastic_ep(self, *_args):
            pass

        async def commit_elastic_ep(self):
            raise RuntimeError("commit failed")

        def shutdown(self, timeout=None):
            pass

    llm = cast(Any, object.__new__(AsyncLLM))
    llm.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=2,
            data_parallel_external_lb=True,
            data_parallel_rank=0,
        )
    )
    llm.engine_core = FailingEngineCore()
    llm.log_stats = False
    monkeypatch.setattr(
        "vllm.v1.engine.async_llm.envs.VLLM_ELASTIC_EP_DRAIN_REQUESTS",
        False,
    )
    set_scaling_elastic_ep(False)

    try:
        with pytest.raises(RuntimeError, match="commit failed"):
            await llm._scale_elastic_ep(3, 120, "scale-1")
        assert get_scaling_elastic_ep()
    finally:
        set_scaling_elastic_ep(False)


@pytest.mark.asyncio
async def test_external_elastic_ep_http_replays_do_not_repeat_commit(monkeypatch):
    coordinator = _external_eep_coordinator()
    store = DictStore({})
    coordinator.reconfig_store_ref = store
    monkeypatch.setattr(coordinator, "_get_reconfig_store", lambda: store)
    monkeypatch.setattr(
        coordinator, "_setup_reconfig_bootstrap", lambda: ("127.0.0.1", 12348)
    )
    prepared, finish_commit = asyncio.Event(), asyncio.Event()

    async def prepare(size, operation_id):
        coordinator._prepare_reconfig_bootstrap(store, 2, size, operation_id)
        prepared.set()

    async def commit():
        epoch = store.get(coordinator.key("current_epoch")).decode()
        store.set(coordinator.key(epoch, "topology_known"), b"0")
        coordinator._set_phase(store, epoch, ExternalElasticEPScalePhase.COMMITTING)
        await finish_commit.wait()
        coordinator._publish_completed((store,), epoch, 3)

    core = SimpleNamespace(
        external_eep_coordinator=coordinator,
        prepare_elastic_ep=prepare,
        commit_elastic_ep=commit,
        shutdown=lambda timeout=None: None,
    )
    core.validate_external_elastic_ep_request = partial(
        DPAsyncMPClient.validate_external_elastic_ep_request, core
    )
    core.get_external_elastic_ep_status = partial(
        DPAsyncMPClient.get_external_elastic_ep_status, core
    )
    llm = cast(Any, object.__new__(AsyncLLM))
    llm.vllm_config = coordinator.client.vllm_config
    llm.engine_core = core
    llm.log_stats = False
    llm._elastic_ep_lock = asyncio.Lock()
    monkeypatch.setattr(
        "vllm.v1.engine.async_llm.envs.VLLM_ELASTIC_EP_DRAIN_REQUESTS", False
    )
    monkeypatch.setattr(
        "vllm.entrypoints.serve.elastic_ep.middleware._scaling_elastic_ep", False
    )
    monkeypatch.setattr(
        "vllm.entrypoints.serve.elastic_ep.middleware._elastic_ep_rank_retired", False
    )
    app = FastAPI()
    app.state.engine_client = llm
    attach_router(app)
    app.add_middleware(ScalingMiddleware)

    @app.post("/v1/completions")
    async def completion():
        return {}

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        initial = (await client.post("/is_scaling_elastic_ep")).json()
        payload = {
            "new_data_parallel_size": 3,
            "operation_id": "scale-1",
            "expected_instance_id": initial["instance_id"],
        }
        request = asyncio.create_task(client.post("/scale_elastic_ep", json=payload))
        try:
            await asyncio.wait_for(prepared.wait(), 5)
            replay = await client.post("/scale_elastic_ep", json=payload)
            assert replay.status_code == 200
            active = replay.json()
            assert active["operation_id"] == "scale-1"
            assert active["phase"] == "committing"
            assert (await client.post("/v1/completions")).status_code == 503
            for changes in (
                {"operation_id": "scale-2"},
                {"new_data_parallel_size": 4},
                {"expected_instance_id": "stale-instance"},
            ):
                conflict = await client.post(
                    "/scale_elastic_ep", json=payload | changes
                )
                assert conflict.status_code == 409
            legacy = await client.post(
                "/scale_elastic_ep", json={"new_data_parallel_size": 3}
            )
            assert legacy.status_code == 503
        finally:
            finish_commit.set()
            result = await request
        assert result.status_code == 200
        completed = result.json()
        assert completed["phase"] == "completed"
        assert completed["effective_data_parallel_size"] == 3
        assert completed["topology_known"]
        replay = await client.post("/scale_elastic_ep", json=payload)
        assert replay.status_code == 200
        assert replay.json()["epoch"] == active["epoch"]
        assert (await client.post("/v1/completions")).status_code == 200
        set_elastic_ep_rank_retired(True)
        assert (await client.post("/scale_elastic_ep", json=payload)).status_code == 200
        mutation = await client.post(
            "/scale_elastic_ep",
            json=payload | {"operation_id": "scale-2", "new_data_parallel_size": 4},
        )
        assert mutation.status_code == 409
        assert (await client.post("/v1/completions")).status_code == 503
        assert (await client.post("/is_scaling_elastic_ep")).status_code == 200
        set_elastic_ep_rank_retired(False)
        set_scaling_elastic_ep(True)
        epoch = completed["epoch"]
        store.set(coordinator.key(epoch, "error"), b"commit failed")
        store.set(coordinator.key(epoch, "topology_known"), b"0")
        replay = await client.post("/scale_elastic_ep", json=payload)
        assert replay.status_code == 200
        assert replay.json()["phase"] == "failed"
        assert not replay.json()["topology_known"]
        mutation = await client.post(
            "/scale_elastic_ep", json=payload | {"operation_id": "scale-2"}
        )
        assert mutation.status_code == 409
        assert (await client.post("/v1/completions")).status_code == 503


def test_external_elastic_ep_calculates_target_expert_redundancy():
    client = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(get_num_experts=lambda: 128),
            parallel_config=SimpleNamespace(
                eplb_config=SimpleNamespace(num_redundant_experts=0)
            ),
        )
    )
    coordinator = ExternalElasticEPScaleCoordinator(cast("DPAsyncMPClient", client))

    assert coordinator._calculate_num_redundant_experts(2, 3) == 64


@pytest.mark.asyncio
@pytest.mark.parametrize("dp_rank", range(4))
async def test_validate_expert_capacity_on_every_rank(dp_rank, monkeypatch):
    """Validate expert capacity on every rank before waiting for an epoch."""
    client = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(get_num_experts=lambda: 128),
            parallel_config=SimpleNamespace(
                data_parallel_rank=dp_rank,
                elastic_ep_max_dp_size=4,
                data_parallel_master_ip="127.0.0.1",
                _coord_store_port=12345,
                eplb_config=SimpleNamespace(num_redundant_experts=0),
            ),
        )
    )
    coordinator = ExternalElasticEPScaleCoordinator(cast("DPAsyncMPClient", client))
    store = DictStore({})
    monkeypatch.setattr(
        "vllm.distributed.utils.get_cached_tcp_store_client", lambda *args: store
    )

    with pytest.raises(ValueError, match="cannot retain all model experts"):
        await asyncio.wait_for(coordinator.prepare(4, 2), timeout=1)

    assert not store.values
    assert coordinator.prepared_scale is None


@pytest.mark.asyncio
async def test_external_elastic_ep_late_rank_observes_epoch_error():
    coordinator = ExternalElasticEPScaleCoordinator(
        cast("DPAsyncMPClient", SimpleNamespace())
    )
    epoch = "failed-epoch"
    store = DictStore(
        {
            coordinator.key("current_epoch"): epoch.encode(),
            coordinator.key(epoch, "error"): b"scale failed",
        }
    )

    with pytest.raises(RuntimeError, match="scale failed"):
        await coordinator._wait_for_bootstrap(store, requested_new_dp_size=3)


class ExternalLBServerManager:
    """Manages data parallel vLLM server instances for external
    load balancer testing."""

    def __init__(
        self,
        model_name: str,
        dp_size: int,
        api_server_count: int,
        base_server_args: list,
        tp_size: int = TP_SIZE,
    ):
        self.model_name = model_name
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.api_server_count = api_server_count
        self.base_server_args = base_server_args
        self.servers: list[tuple[RemoteOpenAIServer, list[str]]] = []
        self.server_threads: list[threading.Thread] = []

    def start_rank(
        self,
        rank: int,
        dp_size: int | None = None,
        elastic_ep_scale_up_launch: bool = False,
    ) -> tuple[RemoteOpenAIServer, list[str]]:
        """Start one externally managed DP rank."""
        dp_size = self.dp_size if dp_size is None else dp_size
        server_args = self.base_server_args.copy()
        server_args.extend(
            [
                "--data-parallel-size",
                str(dp_size),
                "--data-parallel-rank",
                str(rank),
                "--data-parallel-size-local",
                "1",
                "--tensor-parallel-size",
                str(self.tp_size),
                "--port",
                str(8000 + rank),
                "--api-server-count",
                str(self.api_server_count),
            ]
        )

        env_dict = {
            "VLLM_SERVER_DEV_MODE": "1",
            current_platform.device_control_env_var: ",".join(
                str(current_platform.device_id_to_physical_device_id(i))
                for i in range(rank * self.tp_size, (rank + 1) * self.tp_size)
            ),
        }
        if elastic_ep_scale_up_launch:
            env_dict["VLLM_ELASTIC_EP_SCALE_UP_LAUNCH"] = "1"

        server = RemoteOpenAIServer(
            self.model_name,
            server_args,
            auto_port=False,
            env_dict=env_dict,
        )
        print(
            f"Server rank {rank} started successfully with "
            f"{self.api_server_count} API servers"
        )
        server_and_args = (server, server_args)
        self.servers.append(server_and_args)
        return server_and_args

    def __enter__(self) -> list[tuple[RemoteOpenAIServer, list[str]]]:
        """Start all initial server instances for external LB mode."""
        start_errors: list[Exception] = []
        for rank in range(self.dp_size):
            # Use a thread to start each server to allow parallel initialization
            def start_server(r: int):
                try:
                    self.start_rank(r)
                except Exception as e:
                    print(f"Failed to start server rank {r}: {e}")
                    start_errors.append(e)

            thread = threading.Thread(target=start_server, args=(rank,))
            thread.start()

            self.server_threads.append(thread)

        # Wait for all servers to start
        for thread in self.server_threads:
            thread.join()

        if start_errors:
            start_error = start_errors[0]
            raise RuntimeError("Failed to start external LB servers") from start_error

        # Give servers additional time to fully initialize and coordinate
        time.sleep(2)

        if len(self.servers) != self.dp_size:
            raise Exception("Servers failed to start")

        return self.servers

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop all server instances."""
        servers = [s for s, _ in self.servers]
        self.servers.clear()
        try:
            RemoteOpenAIServer.shutdown_many(servers)
        except Exception as e:
            print(f"Error stopping servers: {e}")


@pytest.fixture(scope="module")
def default_server_args():
    return [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "128",
        "--enforce-eager",
    ]


@pytest.fixture(
    scope="module",
    params=[(1, False), (4, False), (1, True)],
    ids=["api-1-standard", "api-4-standard", "api-1-ep-enabled"],
)
def server_manager(request, default_server_args):
    api_server_count, enable_expert_parallel = request.param
    server_args = default_server_args.copy()
    if enable_expert_parallel:
        server_args.append("--enable-expert-parallel")

    server_manager = ExternalLBServerManager(
        MODEL_NAME, DP_SIZE, api_server_count, server_args
    )

    with server_manager:
        yield server_manager


@pytest.fixture
def servers(server_manager):
    return server_manager.servers


@pytest_asyncio.fixture
async def clients(servers: list[tuple[RemoteOpenAIServer, list[str]]]):
    # Create a client for each server
    async with AsyncExitStack() as stack:
        yield [
            await stack.enter_async_context(server.get_async_client())
            for server, _ in servers
        ]


def _get_parallel_config(server: RemoteOpenAIServer):
    response = requests.get(server.url_for("server_info?config_format=json"))
    response.raise_for_status()

    vllm_config = response.json()["vllm_config"]
    return vllm_config["parallel_config"]


def _assert_expert_parallel_config(
    servers: list[tuple[RemoteOpenAIServer, list[str]]],
):
    for server, server_args in servers:
        expected = "--enable-expert-parallel" in server_args
        parallel_config = _get_parallel_config(server)
        assert parallel_config["enable_expert_parallel"] is expected


def _send_scale_command(
    server: RemoteOpenAIServer, new_dp_size: int
) -> requests.Response:
    return requests.post(
        server.url_for("scale_elastic_ep"),
        json={"new_data_parallel_size": new_dp_size},
        timeout=600,
    )


def _wait_for_scale_requests_to_start(
    scale_requests: list[Future[requests.Response]],
    timeout: float = 30,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for request in scale_requests:
            if request.done():
                response = request.result()
                pytest.fail(
                    "Scale request finished before the new rank was launched: "
                    f"{response.status_code} {response.text}"
                )

        if all(request.running() for request in scale_requests):
            return
        time.sleep(0.1)

    pytest.fail("Timed out waiting for all scale requests to start")


@pytest.mark.distributed(num_gpus=3)
@pytest.mark.skipif(
    not current_platform.is_cuda_alike()
    or TP_SIZE != 1
    or current_platform.device_count() < 3,
    reason="External Elastic EP scale-up requires three GPUs with TP_SIZE=1",
)
@pytest.mark.skipif(
    os.getenv("VLLM_USE_RUST_FRONTEND") == "1",
    reason="Elastic EP scaling is not supported by the Rust frontend",
)
def test_external_lb_elastic_ep_scale_up(default_server_args) -> None:
    server_args = [
        *default_server_args,
        "--enable-expert-parallel",
        "--all2all-backend",
        "allgather_reducescatter",
        "--attention-backend",
        "TRITON_MLA",
        "--enable-elastic-ep",
        "--elastic-ep-max-dp-size",
        "3",
        "--enable-eplb",
        "--eplb-config",
        '{"communicator":"pynccl","num_redundant_experts":0,"use_async":false}',
    ]
    server_manager = ExternalLBServerManager(
        ELASTIC_EP_MODEL_NAME,
        dp_size=2,
        api_server_count=1,
        base_server_args=server_args,
    )

    with server_manager:
        existing_servers = [server for server, _ in server_manager.servers]
        with ThreadPoolExecutor(max_workers=2) as executor:
            scale_requests = [
                executor.submit(_send_scale_command, server, 3)
                for server in existing_servers
            ]
            _wait_for_scale_requests_to_start(scale_requests)
            server_manager.start_rank(
                rank=2,
                dp_size=3,
                elastic_ep_scale_up_launch=True,
            )

            for request in scale_requests:
                response = request.result(timeout=600)
                assert response.status_code == 200, response.text

        for server, _ in server_manager.servers:
            parallel_config = _get_parallel_config(server)
            assert parallel_config["data_parallel_size"] == 3
            assert parallel_config["eplb_config"]["num_redundant_experts"] == 32
            with server.get_client() as client:
                completion = client.completions.create(
                    model=ELASTIC_EP_MODEL_NAME,
                    prompt="Hello, my name is",
                    max_tokens=5,
                    temperature=0.0,
                )
            assert completion.choices[0].finish_reason in ("length", "stop")


def test_external_lb_server_info(server_manager):
    servers = server_manager.servers
    api_server_count = server_manager.api_server_count

    for i, (server, _) in enumerate(servers):
        print(f"Testing {i=}")

        # Each request will hit one of the API servers
        # `n_reqs` is set so that there is a good chance each server
        # receives at least one request
        n_reqs = 2 * api_server_count * api_server_count
        parallel_configs = [_get_parallel_config(server) for _ in range(n_reqs)]
        api_process_counts = [c["_api_process_count"] for c in parallel_configs]
        api_process_ranks = [c["_api_process_rank"] for c in parallel_configs]

        assert all(c == api_server_count for c in api_process_counts), (
            api_process_counts
        )
        assert all(0 <= r < api_server_count for r in api_process_ranks), (
            api_process_ranks
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_external_lb_single_completion(
    clients: list[openai.AsyncOpenAI],
    servers: list[tuple[RemoteOpenAIServer, list[str]]],
    model_name: str,
) -> None:
    async def make_request(client: openai.AsyncOpenAI):
        completion = await client.completions.create(
            model=model_name, prompt="Hello, my name is", max_tokens=10, temperature=1.0
        )

        assert completion.id is not None
        assert completion.choices is not None and len(completion.choices) == 1

        choice = completion.choices[0]
        # The exact number of tokens can vary slightly with temperature=1.0,
        # so we check for a reasonable minimum length.
        assert len(choice.text) >= 1
        # Finish reason might not always be 'length' if the model finishes early
        # or due to other reasons, especially with high temperature.
        # So, we'll accept 'length' or 'stop'.
        assert choice.finish_reason in ("length", "stop")

        # Token counts can also vary, so we check they are positive.
        assert completion.usage.completion_tokens > 0
        assert completion.usage.prompt_tokens > 0
        assert completion.usage.total_tokens > 0
        return completion

    # Test single request to each server
    for i, client in enumerate(clients):
        result = await make_request(client)
        assert result is not None
        print(f"Server {i} handled single completion request successfully")

    await asyncio.sleep(0.5)

    # Send requests to all servers in round-robin fashion
    num_requests_per_server = 25  # Total 50 requests across 2 servers
    all_tasks = []

    for i, client in enumerate(clients):
        tasks = [make_request(client) for _ in range(num_requests_per_server)]
        all_tasks.extend(tasks)

    results = await asyncio.gather(*all_tasks)
    assert len(results) == num_requests_per_server * len(clients)
    assert all(completion is not None for completion in results)

    await asyncio.sleep(0.5)

    # Second burst of requests
    all_tasks = []
    for i, client in enumerate(clients):
        tasks = [make_request(client) for _ in range(num_requests_per_server)]
        all_tasks.extend(tasks)

    results = await asyncio.gather(*all_tasks)
    assert len(results) == num_requests_per_server * len(clients)
    assert all(completion is not None for completion in results)

    _, server_args = servers[0]
    api_server_count = (
        server_args.count("--api-server-count")
        and server_args[server_args.index("--api-server-count") + 1]
        or 1
    )
    print(
        f"Successfully completed external LB test with {len(clients)} servers "
        f"(API server count: {api_server_count})"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_external_lb_completion_streaming(
    clients: list[openai.AsyncOpenAI],
    servers: list[tuple[RemoteOpenAIServer, list[str]]],
    model_name: str,
) -> None:
    prompt = "What is an LLM?"

    async def make_streaming_request(client: openai.AsyncOpenAI):
        # Perform a non-streaming request to get the expected full output
        single_completion = await client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=5,
            temperature=0.0,
        )
        single_output = single_completion.choices[0].text

        # Perform the streaming request
        stream = await client.completions.create(
            model=model_name, prompt=prompt, max_tokens=5, temperature=0.0, stream=True
        )
        chunks: list[str] = []
        finish_reason_count = 0
        last_chunk = None
        async for chunk in stream:
            chunks.append(chunk.choices[0].text)
            if chunk.choices[0].finish_reason is not None:
                finish_reason_count += 1
            last_chunk = chunk  # Keep track of the last chunk

        # finish reason should only return in the last block for OpenAI API
        assert finish_reason_count == 1, "Finish reason should appear exactly once."
        assert last_chunk is not None, "Stream should have yielded at least one chunk."
        assert last_chunk.choices[0].finish_reason == "length", (
            "Finish reason should be 'length'."
        )
        # Check that the combined text matches the non-streamed version.
        assert "".join(chunks) == single_output, (
            "Streamed output should match non-streamed output."
        )
        return True  # Indicate success for this request

    # Test single request to each server
    for i, client in enumerate(clients):
        result = await make_streaming_request(client)
        assert result is not None
        print(f"Server {i} handled single streaming request successfully")

    await asyncio.sleep(0.5)

    # Send streaming requests to all servers in round-robin fashion
    num_requests_per_server = 25  # Total 50 requests across 2 servers
    all_tasks = []

    for i, client in enumerate(clients):
        tasks = [make_streaming_request(client) for _ in range(num_requests_per_server)]
        all_tasks.extend(tasks)

    results = await asyncio.gather(*all_tasks)
    assert len(results) == num_requests_per_server * len(clients)
    assert all(results), "Not all streaming requests completed successfully."

    await asyncio.sleep(0.5)

    # Second burst of streaming requests
    all_tasks = []
    for i, client in enumerate(clients):
        tasks = [make_streaming_request(client) for _ in range(num_requests_per_server)]
        all_tasks.extend(tasks)

    results = await asyncio.gather(*all_tasks)
    assert len(results) == num_requests_per_server * len(clients)
    assert all(results), "Not all streaming requests completed successfully."

    _, server_args = servers[0]
    api_server_count = (
        server_args.count("--api-server-count")
        and server_args[server_args.index("--api-server-count") + 1]
        or 1
    )
    print(
        f"Successfully completed external LB streaming test with "
        f"{len(clients)} servers (API server count: {api_server_count})"
    )
