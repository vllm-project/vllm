# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Lifecycle integration tests for DPSupervisor.

These tests replace child vLLM servers with lightweight aiohttp "fake" servers
controlled by the test, so the suite runs without GPUs.  _start_children is
monkeypatched to install FakeProcess objects (with controllable liveness/timing)
alongside those fake HTTP servers.

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

import aiohttp
import pytest
import uvicorn
from fastapi import FastAPI, Response

import vllm.entrypoints.openai.dp_supervisor as dp_sup
from vllm.entrypoints.openai.dp_supervisor import CHILD_EXIT_GRACE_S, DPSupervisor
from vllm.logger import init_logger

logger = init_logger(__name__)

_SUPERVISOR_PORT = 19256
_CHILD_PORT_BASE = 18000
_N_CHILDREN = 2
_PROBE_INTERVAL = 1.0  # smaller interval so tests run quickly
_POLL_INTERVAL = 1.0  # smaller interval so tests run quickly

# ---------------------------------------------------------------------------
# Monkeypatch to use mock vLLM server
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

        # Put the server onto the eventloop.
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
# Args factory
# ---------------------------------------------------------------------------


def _make_args(**overrides) -> argparse.Namespace:
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
        node_rank=0,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        uvicorn_log_level="warning",
        shutdown_timeout=0.0,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# Test helpers
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
            # -1 means expected client error
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


# ---------------------------------------------------------------------------
# Core context manager
# ---------------------------------------------------------------------------


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
# Test 1 – Basic lifecycle: not-ready → ready → sigterm
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
        # A: supervisor HTTP server is up but children are not healthy yet.
        assert await _poll_supervisor_health(503)
        assert not supervisor.is_ready

        # Wait until the vLLM servers finally started.
        for port in vllm_server_ports:
            assert await _poll_supervisor_health(503)
            assert not supervisor.is_ready
            await _poll_until_api_server_running(port)

        # B: flip the first server, should still not be ready.
        await _set_healthy(vllm_server_ports[0])
        await asyncio.sleep(1.0)
        assert await _poll_supervisor_health(503)
        assert not supervisor.is_ready
        print("/health is 503 --- expected!")

        # C: flip all the servers, should get a 200.
        for port in vllm_server_ports:
            await _set_healthy(port)
        await asyncio.sleep(1.0)
        assert await _poll_supervisor_health(200)
        assert supervisor.is_ready
        print("/health is 200 --- expected!")

        # D: sleep for a few seconds, then try again, should get 200s.
        await asyncio.sleep(1.0)
        assert await _poll_supervisor_health(200)
        assert supervisor.is_ready
        print("/health is 200 --- expected!")

        # E: Simulate K8s sending a SIGTERM (DPSupervisor runs in
        # the main process here, so we signal ourselves).
        os.kill(os.getpid(), signal.SIGTERM)

        await asyncio.wait_for(_task, timeout=5.0)
        for p in supervisor._processes:
            assert not p.is_alive()
        print("everything was cleaned up!")


# ---------------------------------------------------------------------------
# Test 2 – Failed Startup in one vLLM DP Server
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_startup(monkeypatch):
    """
    A) One of the vLLM servers crashes during startup.
    B) DPSupervisor detects this, and cleans up resources.
    """

    args = _make_args()

    vllm_server_ports = [_CHILD_PORT_BASE + i for i in range(_N_CHILDREN)]

    async with _run_supervisor(args, monkeypatch) as (supervisor, _task):
        # A: DPSupervisor is up but children are not healthy yet.
        assert await _poll_supervisor_health(503)
        assert not supervisor.is_ready

        # Wait until the apis servers start (but not ready).
        for port in vllm_server_ports:
            await _poll_until_api_server_running(port)

        # Kill DP server processes.
        await _kill_server(port)

        # B: DPSupervisor and all vLLM processes should shut down.
        # If the DPSupervisor does not shut down, it raises a TimeoutError.
        try:
            import time

            before = time.perf_counter()
            print("WAITING")
            await asyncio.wait_for(_task, timeout=5.0)
        except TimeoutError:
            then = time.perf_counter()
            print(f"GOT TIMEOUT ERROR {then - before}")
        for p in supervisor._processes:
            assert not p.is_alive()


# ---------------------------------------------------------------------------
# Test 3 – Failed /health in one vLLM DP Server after start
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_becomes_unhealthy(monkeypatch):
    """
    A) Supervisor /health returns 503 while children are unhealthy.
    B) /health returns 200 once every child reports healthy.
    C) Child process becomes unhealtly.
    D) Detected and shutdown.
    """

    args = _make_args()

    vllm_server_ports = [_CHILD_PORT_BASE + i for i in range(_N_CHILDREN)]

    async with _run_supervisor(args, monkeypatch) as (supervisor, _task):
        # A: supervisor HTTP server is up but children are not healthy yet.
        assert await _poll_supervisor_health(503)
        assert not supervisor.is_ready

        # Wait until the vLLM servers finally started.
        for port in vllm_server_ports:
            assert await _poll_supervisor_health(503)
            assert not supervisor.is_ready
            await _poll_until_api_server_running(port)

        # B: flip the first server, should still not be ready.
        await _set_healthy(vllm_server_ports[0])
        await asyncio.sleep(1.0)
        assert await _poll_supervisor_health(503)
        assert not supervisor.is_ready
        print("/health is 503 --- expected!")

        # C: flip all the servers, should get a 200.
        for port in vllm_server_ports:
            await _set_healthy(port)
        await asyncio.sleep(1.0)
        assert await _poll_supervisor_health(200)
        assert supervisor.is_ready
        print("/health is 200 --- expected!")

        # D: flip a server to unhealthy (e.g. hang.)
        await _set_unhealthy(port)

        # E: everything should clean up.
        await asyncio.wait_for(_task, timeout=5.0)
        for p in supervisor._processes:
            assert not p.is_alive()
        print("everything was cleaned up!")


# ---------------------------------------------------------------------------
# Test 4 – Process dies one vLLM DP Server after start
# ---------------------------------------------------------------------------


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
        # A: supervisor HTTP server is up but children are not healthy yet.
        assert await _poll_supervisor_health(503)
        assert not supervisor.is_ready

        # Wait until the vLLM servers finally started.
        for port in vllm_server_ports:
            assert await _poll_supervisor_health(503)
            assert not supervisor.is_ready
            await _poll_until_api_server_running(port)

        # B: flip the first server, should still not be ready.
        await _set_healthy(vllm_server_ports[0])
        await asyncio.sleep(1.0)
        assert await _poll_supervisor_health(503)
        assert not supervisor.is_ready
        print("/health is 503 --- expected!")

        # C: flip all the servers, should get a 200.
        for port in vllm_server_ports:
            await _set_healthy(port)
        await asyncio.sleep(1.0)
        assert await _poll_supervisor_health(200)
        assert supervisor.is_ready
        print("/health is 200 --- expected!")

        # D: fill one of the background processes.
        dp_mock_server_process = supervisor._processes[0]
        os.kill(dp_mock_server_process.pid, signal.SIGKILL)
        await asyncio.sleep(1.0)
        assert not dp_mock_server_process.is_alive()

        # E: everything should clean up.
        await asyncio.wait_for(_task, timeout=5.0)
        for p in supervisor._processes:
            assert not p.is_alive()
        print("everything was cleaned up!")


# ---------------------------------------------------------------------------
# Test 5 – Graceful drain on shutdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shutdown_timeout(monkeypatch: pytest.MonkeyPatch):
    """
    Child mock servers delay shutdown by 15s on SIGTERM (simulating in-flight
    request drain).  The supervisor is configured with shutdown_timeout=15,
    so its total wait budget is 15 + CHILD_EXIT_GRACE_S seconds.  The
    children exit naturally within that window, so no force-kill should occur
    and the measured wall-clock time must be >= 15s.
    """
    _DRAIN_SECONDS = 10.0
    _SHUTDOWN_TIMEOUT = 10.0

    args = _make_args(shutdown_timeout=_SHUTDOWN_TIMEOUT)
    vllm_server_ports = [_CHILD_PORT_BASE + i for i in range(_N_CHILDREN)]

    async with _run_supervisor(
        args, monkeypatch, launch_fn=launch_mock_vllm_with_drain
    ) as (supervisor, _task):
        # Wait for all mock servers to start (unhealthy but reachable).
        for port in vllm_server_ports:
            await _poll_until_api_server_running(port)

        # Mark all children healthy so the supervisor reaches is_ready.
        for port in vllm_server_ports:
            await _set_healthy(port)
        await asyncio.sleep(1.0)
        assert await _poll_supervisor_health(200)
        assert supervisor.is_ready

        # Trigger graceful shutdown and measure wall-clock time.
        start_t = time.perf_counter()
        os.kill(os.getpid(), signal.SIGTERM)

        # Children drain for 15s; total budget is 15 + CHILD_EXIT_GRACE_S.
        # Add a few extra seconds of slack for scheduling jitter.
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
