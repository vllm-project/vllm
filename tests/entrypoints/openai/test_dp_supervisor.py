# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for DPSupervisor: unit tests and lifecycle integration tests.

Lifecycle integration tests replace child vLLM servers with lightweight
aiohttp "fake" servers controlled by the test, so the suite runs without GPUs.
_start_children is monkeypatched to install FakeProcess objects (with
controllable liveness/timing) alongside those fake HTTP servers.

Port allocation (kept far from default vLLM ports to avoid conflicts):
  Supervisor : 19256
  Children   : 18000, 18001
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import signal
import time
from types import SimpleNamespace

import aiohttp
import pytest
import uvicorn
from fastapi import FastAPI, Response

import vllm.entrypoints.openai.dp_supervisor as dp_sup
from vllm.entrypoints.openai.dp_supervisor import (
    CHILD_EXIT_GRACE_S,
    DPSupervisor,
    _build_vllm_dp_server_args,
    infer_multi_port_external_lb_start_rank,
    validate_multi_port_external_lb_args,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

_SUPERVISOR_PORT = 19256
_CHILD_PORT_BASE = 18000
_N_CHILDREN = 2
_PROBE_INTERVAL = 1.0
_POLL_INTERVAL = 1.0


# ---------------------------------------------------------------------------
# Args factories
# ---------------------------------------------------------------------------


def _make_unit_args(**overrides) -> argparse.Namespace:
    """Minimal args for unit tests (no real network activity)."""
    base = {
        "host": None,
        "port": 8000,
        "data_parallel_multi_port_external_lb": True,
        "data_parallel_supervisor_port": 9256,
        "dp_supervisor_probe_interval_s": 5.0,
        "dp_supervisor_probe_timeout_s": 5.0,
        "dp_supervisor_probe_failure_threshold": 3,
        "data_parallel_size": 8,
        "data_parallel_size_local": 4,
        "data_parallel_start_rank": None,
        "data_parallel_rank": None,
        "data_parallel_external_lb": False,
        "data_parallel_hybrid_lb": False,
        "api_server_count": None,
        "headless": False,
        "grpc": False,
        "uds": None,
        "ssl_keyfile": None,
        "ssl_certfile": None,
        "ssl_ca_certs": None,
        "ssl_cert_reqs": 0,
        "ssl_ciphers": None,
        "node_rank": 1,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "uvicorn_log_level": "info",
        "shutdown_timeout": 5.0,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _make_args(**overrides) -> argparse.Namespace:
    """Args for lifecycle integration tests (real loopback servers)."""
    base: dict = dict(
        host="127.0.0.1",
        port=_CHILD_PORT_BASE,
        data_parallel_multi_port_external_lb=True,
        data_parallel_supervisor_port=_SUPERVISOR_PORT,
        dp_supervisor_probe_interval_s=_PROBE_INTERVAL,
        dp_supervisor_probe_timeout_s=1.0,
        dp_supervisor_probe_failure_threshold=3,
        data_parallel_size=_N_CHILDREN,
        data_parallel_size_local=_N_CHILDREN,
        data_parallel_start_rank=0,
        data_parallel_rank=None,
        data_parallel_external_lb=False,
        data_parallel_hybrid_lb=False,
        api_server_count=None,
        headless=False,
        grpc=False,
        uds=None,
        ssl_keyfile=None,
        ssl_certfile=None,
        ssl_ca_certs=None,
        ssl_cert_reqs=0,
        ssl_ciphers=None,
        node_rank=0,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        uvicorn_log_level="warning",
        shutdown_timeout=0.0,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_infer_multi_port_external_lb_start_rank_uses_node_rank():
    args = _make_unit_args()
    assert infer_multi_port_external_lb_start_rank(args) == 4


def test_build_multi_port_external_lb_child_args_sets_external_rank_server():
    args = _make_unit_args(data_parallel_start_rank=8, api_server_count=None)
    child_args = _build_vllm_dp_server_args(args, local_rank=2)

    assert child_args.port == 8002
    assert child_args.data_parallel_rank == 10
    assert child_args.data_parallel_size_local == 1
    assert child_args.data_parallel_external_lb is True
    assert child_args.data_parallel_hybrid_lb is False
    assert child_args.data_parallel_multi_port_external_lb is False
    assert child_args.api_server_count == 1


def test_validate_multi_port_external_lb_args_allows_ssl():
    args = _make_unit_args(
        ssl_keyfile="/tmp/server.key",
        ssl_certfile="/tmp/server.crt",
        ssl_ca_certs="/tmp/ca.crt",
    )
    validate_multi_port_external_lb_args(args)


def test_aggregates_health():
    supervisor = DPSupervisor(_make_unit_args())
    supervisor._is_ready = True
    assert supervisor.is_ready is True


def test_handles_shutdown_event():
    supervisor = DPSupervisor(_make_unit_args())
    supervisor._is_ready = True
    supervisor._shutdown_event.set()
    assert supervisor.is_ready is False


@pytest.mark.asyncio
async def test_handles_child_exit(
    monkeypatch: pytest.MonkeyPatch,
):
    supervisor = DPSupervisor(_make_unit_args())
    supervisor._processes = [
        SimpleNamespace(
            name="APIServer_DPRank_4", exitcode=None, is_alive=lambda: True
        ),
        SimpleNamespace(name="APIServer_DPRank_5", exitcode=17, is_alive=lambda: False),
    ]

    async def fake_probe(*_args, **_kwargs) -> bool:
        return True

    monkeypatch.setattr(dp_sup, "_probe_endpoint", fake_probe)

    await supervisor._monitor_children()
    assert supervisor._is_ready is False


@pytest.mark.asyncio
async def test_handles_probe_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    supervisor = DPSupervisor(_make_unit_args(dp_supervisor_probe_interval_s=0.0))
    supervisor.child_ports = [8000]
    probe_results = iter([True, False])

    async def fake_probe(*_args, **_kwargs) -> bool:
        return next(probe_results)

    monkeypatch.setattr(dp_sup, "_probe_endpoint", fake_probe)

    await supervisor._monitor_children()
    assert supervisor._is_ready is False


@pytest.mark.asyncio
async def test_shutdown_if_supervisor_server_error_on_startup(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeLoop:
        def add_signal_handler(self, *_args, **_kwargs):
            pass

        def remove_signal_handler(self, *_args, **_kwargs):
            pass

    class FakeServer:
        def __init__(self, _config):
            self.started = False
            self.should_exit = False

        async def serve(self):
            raise ValueError("supervisor boom")

    async def fake_shutdown_children(self):
        return None

    monkeypatch.setattr(dp_sup.asyncio, "get_running_loop", lambda: FakeLoop())
    monkeypatch.setattr(dp_sup.uvicorn, "Server", FakeServer)
    monkeypatch.setattr(DPSupervisor, "_shutdown_children", fake_shutdown_children)

    supervisor = DPSupervisor(_make_unit_args())

    with pytest.raises(ValueError, match="supervisor boom"):
        await supervisor.run()


# ---------------------------------------------------------------------------
# Lifecycle integration tests – MockVLLMServer
# ---------------------------------------------------------------------------


class MockVLLMServer:
    """
    Minimal FastAPI server that mimics one vLLM replica.
    GET /health returns 200 when healthy, 503 otherwise.
    Health state is toggled by the test via set_healthy().
    """

    def __init__(self, port: int, drain_seconds: float = 0.0) -> None:
        self.port = port
        self._healthy = False
        self._drain_seconds = drain_seconds
        self._server: uvicorn.Server | None = None
        self._serve_task: asyncio.Task | None = None

    async def start(self) -> None:
        app = FastAPI()

        @app.get("/health")
        async def health() -> Response:
            print(f"MockServer {self.port}: /health: {self._healthy}")
            return Response(status_code=200 if self._healthy else 503)

        @app.get("/set_healthy")
        async def set_healthy() -> Response:
            print(f"MockServer {self.port}: /set_healthy")
            self._healthy = True
            return Response(status_code=200)

        @app.get("/set_unhealthy")
        async def set_unhealthy() -> Response:
            print(f"MockServer {self.port}: /set_unhealthy")
            self._healthy = False
            return Response(status_code=200)

        @app.get("/kill")
        async def kill() -> Response:
            print(f"MockServer {self.port}: /kill")
            os.kill(os.getpid(), signal.SIGKILL)

        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self.port,
            log_level="warning",
            lifespan="off",
        )
        self._server = uvicorn.Server(config)

        # Configure request draining if needed.
        # Uvicorn's capture_signals() installs signal.signal(SIGTERM, self.handle_exit),
        # which sets should_exit=True immediately. Override handle_exit on the instance
        # so capture_signals() picks up our version that drains first.
        if self._drain_seconds > 0:
            self._shutdown_event = asyncio.Event()
            loop = asyncio.get_running_loop()

            async def _drain_and_stop() -> None:
                await self._shutdown_event.wait()
                print(f"MockServer {self.port}: draining for {self._drain_seconds}s.")
                await asyncio.sleep(self._drain_seconds)
                print("Setting should_exit")
                if self._server is not None:
                    self._server.should_exit = True

            self._drain_task = asyncio.create_task(_drain_and_stop())

            def _custom_handle_exit(sig: int, frame: object) -> None:
                print("Got SIGTERM, setting shutdown.")
                if not self._shutdown_event.is_set():
                    loop.call_soon_threadsafe(self._shutdown_event.set)

            self._server.handle_exit = _custom_handle_exit

        self._serve_task = asyncio.create_task(self._server.serve())
        while not self._server.started:
            await asyncio.sleep(0.01)
        print(f"Mock DP Server on port {self.port} started")

        await self._serve_task


def launch_mock_vllm(child_args: argparse.Namespace, env_updates: dict[str, str]):
    logger.info("Launching mock vLLM on port %s", child_args.port)
    mock_vllm = MockVLLMServer(port=child_args.port)
    asyncio.run(mock_vllm.start())


def launch_mock_vllm_with_drain(
    child_args: argparse.Namespace, env_updates: dict[str, str]
):
    logger.info("Launching mock vLLM with 15s drain on port %s", child_args.port)
    mock_vllm = MockVLLMServer(port=child_args.port, drain_seconds=10.0)
    asyncio.run(mock_vllm.start())


# ---------------------------------------------------------------------------
# Lifecycle test helpers
# ---------------------------------------------------------------------------


async def _poll_supervisor_health(expected_status: int) -> bool:
    """
    Poll GET /health on the supervisor until expected_status is seen.
    A connection error is treated as 503-equivalent when expected_status != 200.
    """
    url = f"http://127.0.0.1:{_SUPERVISOR_PORT}/health"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as resp:
                if resp.status != expected_status:
                    print(f"expected: {expected_status=}, got: {resp.status=}")
                    return False
                return True
        except aiohttp.ClientError:
            if expected_status != -1:
                print(f"expected: {expected_status=}, got: aiohttp.ClientError")
                return False
            return True


async def _poll_until_api_server_running(port: int, retries: int = 10) -> None:
    url = f"http://127.0.0.1:{port}/health"
    async with aiohttp.ClientSession() as session:
        for _ in range(retries):
            try:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return
                await asyncio.sleep(1.0)
            except aiohttp.ClientError:
                print("Test detected not started yet, sleeping for 1s")
                await asyncio.sleep(1.0)


async def _set_healthy(port: int) -> None:
    url = f"http://127.0.0.1:{port}/set_healthy"
    async with aiohttp.ClientSession() as session, session.get(url) as resp:
        assert resp.status == 200


async def _set_unhealthy(port: int) -> None:
    url = f"http://127.0.0.1:{port}/set_unhealthy"
    async with aiohttp.ClientSession() as session, session.get(url) as resp:
        assert resp.status == 200


async def _kill_server(port: int) -> None:
    url = f"http://127.0.0.1:{port}/kill"
    try:
        async with aiohttp.ClientSession() as session, session.get(url) as resp:
            assert resp.status != 200
    except Exception as e:
        assert isinstance(e, aiohttp.ClientConnectorError)


@contextlib.asynccontextmanager
async def _run_supervisor(
    args: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
    launch_fn=None,
):
    if launch_fn is None:
        launch_fn = launch_mock_vllm
    monkeypatch.setattr(dp_sup, "_run_vllm_dp_server", launch_fn)
    supervisor = DPSupervisor(args)
    task = asyncio.create_task(supervisor.run())
    await asyncio.sleep(1.0)
    try:
        yield supervisor, task
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


# ---------------------------------------------------------------------------
# Lifecycle integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_basic_lifecycle(monkeypatch):
    """
    A) Supervisor /health returns 503 while children are unhealthy.
    B) /health returns 200 once every child reports healthy.
    C) SIGTERM and shutdown
    """
    args = _make_args()

    vllm_server_ports = [_CHILD_PORT_BASE + i for i in range(_N_CHILDREN)]

    async with _run_supervisor(args, monkeypatch) as (supervisor, _task):
        assert await _poll_supervisor_health(503)
        assert not supervisor.is_ready

        for port in vllm_server_ports:
            assert await _poll_supervisor_health(503)
            assert not supervisor.is_ready
            await _poll_until_api_server_running(port)

        await _set_healthy(vllm_server_ports[0])
        await asyncio.sleep(1.0)
        assert await _poll_supervisor_health(503)
        assert not supervisor.is_ready
        print("/health is 503 --- expected!")

        for port in vllm_server_ports:
            await _set_healthy(port)
        await asyncio.sleep(1.0)
        assert await _poll_supervisor_health(200)
        assert supervisor.is_ready
        print("/health is 200 --- expected!")

        await asyncio.sleep(1.0)
        assert await _poll_supervisor_health(200)
        assert supervisor.is_ready
        print("/health is 200 --- expected!")

        os.kill(os.getpid(), signal.SIGTERM)

        await asyncio.wait_for(_task, timeout=5.0)
        for p in supervisor._processes:
            assert not p.is_alive()
        print("everything was cleaned up!")


@pytest.mark.asyncio
async def test_failed_startup(monkeypatch):
    """
    A) One of the vLLM servers crashes during startup.
    B) DPSupervisor detects this, and cleans up resources.
    """
    args = _make_args()

    vllm_server_ports = [_CHILD_PORT_BASE + i for i in range(_N_CHILDREN)]

    async with _run_supervisor(args, monkeypatch) as (supervisor, _task):
        assert await _poll_supervisor_health(503)
        assert not supervisor.is_ready

        for port in vllm_server_ports:
            await _poll_until_api_server_running(port)

        await _kill_server(port)

        await asyncio.wait_for(_task, timeout=5.0)
        for p in supervisor._processes:
            assert not p.is_alive()


@pytest.mark.asyncio
async def test_becomes_unhealthy(monkeypatch):
    """
    A) Supervisor /health returns 503 while children are unhealthy.
    B) /health returns 200 once every child reports healthy.
    C) Child process becomes unhealthy.
    D) Detected and shutdown.
    """
    args = _make_args()

    vllm_server_ports = [_CHILD_PORT_BASE + i for i in range(_N_CHILDREN)]

    async with _run_supervisor(args, monkeypatch) as (supervisor, _task):
        assert await _poll_supervisor_health(503)
        assert not supervisor.is_ready

        for port in vllm_server_ports:
            assert await _poll_supervisor_health(503)
            assert not supervisor.is_ready
            await _poll_until_api_server_running(port)

        await _set_healthy(vllm_server_ports[0])
        await asyncio.sleep(1.0)
        assert await _poll_supervisor_health(503)
        assert not supervisor.is_ready
        print("/health is 503 --- expected!")

        for port in vllm_server_ports:
            await _set_healthy(port)
        await asyncio.sleep(1.0)
        assert await _poll_supervisor_health(200)
        assert supervisor.is_ready
        print("/health is 200 --- expected!")

        await _set_unhealthy(port)

        await asyncio.wait_for(_task, timeout=5.0)
        for p in supervisor._processes:
            assert not p.is_alive()
        print("everything was cleaned up!")


@pytest.mark.asyncio
async def test_dp_server_fails(monkeypatch):
    """
    A) Supervisor /health returns 503 while children are unhealthy.
    B) /health returns 200 once every child reports healthy.
    C) Child process fails.
    D) Detected and shutdown.
    """
    args = _make_args()

    vllm_server_ports = [_CHILD_PORT_BASE + i for i in range(_N_CHILDREN)]

    async with _run_supervisor(args, monkeypatch) as (supervisor, _task):
        assert await _poll_supervisor_health(503)
        assert not supervisor.is_ready

        for port in vllm_server_ports:
            assert await _poll_supervisor_health(503)
            assert not supervisor.is_ready
            await _poll_until_api_server_running(port)

        await _set_healthy(vllm_server_ports[0])
        await asyncio.sleep(1.0)
        assert await _poll_supervisor_health(503)
        assert not supervisor.is_ready
        print("/health is 503 --- expected!")

        for port in vllm_server_ports:
            await _set_healthy(port)
        await asyncio.sleep(1.0)
        assert await _poll_supervisor_health(200)
        assert supervisor.is_ready
        print("/health is 200 --- expected!")

        dp_mock_server_process = supervisor._processes[0]
        os.kill(dp_mock_server_process.pid, signal.SIGKILL)
        await asyncio.sleep(1.0)
        assert not dp_mock_server_process.is_alive()

        await asyncio.wait_for(_task, timeout=5.0)
        for p in supervisor._processes:
            assert not p.is_alive()
        print("everything was cleaned up!")


@pytest.mark.asyncio
async def test_shutdown_timeout(monkeypatch: pytest.MonkeyPatch):
    """
    Child mock servers delay shutdown by 10s on SIGTERM (simulating in-flight
    request drain).  The supervisor is configured with shutdown_timeout=10,
    so its total wait budget is 10 + CHILD_EXIT_GRACE_S seconds.  The
    children exit naturally within that window, so no force-kill should occur
    and the measured wall-clock time must be >= 10s.
    """
    _DRAIN_SECONDS = 10.0
    _SHUTDOWN_TIMEOUT = 10.0

    args = _make_args(shutdown_timeout=_SHUTDOWN_TIMEOUT)
    vllm_server_ports = [_CHILD_PORT_BASE + i for i in range(_N_CHILDREN)]

    async with _run_supervisor(
        args, monkeypatch, launch_fn=launch_mock_vllm_with_drain
    ) as (supervisor, _task):
        for port in vllm_server_ports:
            await _poll_until_api_server_running(port)

        for port in vllm_server_ports:
            await _set_healthy(port)
        await asyncio.sleep(1.0)
        assert await _poll_supervisor_health(200)
        assert supervisor.is_ready

        start_t = time.perf_counter()
        os.kill(os.getpid(), signal.SIGTERM)

        print(f"DRAINING FOR {_DRAIN_SECONDS}")
        await asyncio.wait_for(_task, timeout=_DRAIN_SECONDS + CHILD_EXIT_GRACE_S + 5.0)
        elapsed = time.perf_counter() - start_t

        assert elapsed >= _DRAIN_SECONDS, (
            f"Supervisor exited after only {elapsed:.1f}s; "
            f"expected >= {_DRAIN_SECONDS}s for request draining"
        )

        for p in supervisor._processes:
            assert not p.is_alive()
        print(f"Supervisor waited {elapsed:.1f}s for children to drain — expected!")
