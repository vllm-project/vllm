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

import aiohttp
import pytest
import uvicorn
from fastapi import FastAPI, Response

import vllm.entrypoints.openai.dp_supervisor as dp_sup
from vllm.entrypoints.openai.dp_supervisor import DPSupervisor
from vllm.logger import init_logger

logger = init_logger(__name__)

_SUPERVISOR_PORT = 19256
_CHILD_PORT_BASE = 18000
_N_CHILDREN = 2
_PROBE_INTERVAL = 0.05  # small interval so tests run quickly
_POLL_INTERVAL = 0.05

# ---------------------------------------------------------------------------
# Monkeypatch to use mock vLLM server
# ---------------------------------------------------------------------------


class MockVLLMServer:
    """
    Minimal FastAPI server that mimics one vLLM replica.
    GET /health returns 200 when healthy, 503 otherwise.
    Health state is toggled by the test via set_healthy().
    """

    def __init__(self, port: int) -> None:
        self.port = port
        self._healthy = False
        self._server: uvicorn.Server | None = None
        self._serve_task: asyncio.Task | None = None

    async def start(self) -> None:
        if os.environ.get("MOCK_VLLM_FAIL_PORT") == str(self.port):
            print(f"Mock Server on port {self.port} sleeping for 2s then exiting.")
            await asyncio.sleep(2.0)
            raise ValueError(f"Mock Server on port {self.port} simulated failure.")

        app = FastAPI()

        @app.get("/health")
        async def health() -> Response:
            return Response(status_code=200 if self._healthy else 503)

        @app.get("/set_healthy")
        async def set_healthy() -> Response:
            self._healthy = True
            return Response(status_code=200)

        @app.get("/set_unhealthy")
        async def set_unhealthy() -> Response:
            self._healthy = False
            return Response(status_code=200)

        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self.port,
            log_level="warning",
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._serve_task = asyncio.create_task(self._server.serve())

        # Ensure the task is started on the eventloop.
        while not self._server.started:
            await asyncio.sleep(0.01)
        print(f"Mock Server on port {self.port} started")

        await self._serve_task

    def set_healthy(self, healthy: bool) -> None:
        self._healthy = healthy

    # def set_delayed_shutdown(self, shutdown_t: float):
    #     loop = asyncio.get_running_loop()

    #     def delayed_exit(signal: int):
    #         logger.info("Sleeping for %s seconds...", shutdown_t)
    #         time.sleep(shutdown_t)
    #         logger.info("Shutting down.")
    #         if self._server is not None:
    #             self._server.should_exit = True

    #     loop.add_signal_handler(signal.SIGTERM, delayed_exit)


def launch_mock_vllm(child_args: argparse.Namespace, env_updates: dict[str, str]):
    logger.info("Launching mock vLLM on port %s", child_args.port)
    mock_vllm = MockVLLMServer(port=child_args.port)
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
        data_parallel_probe_interval_s=_PROBE_INTERVAL,
        data_parallel_probe_timeout_s=1.0,
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
                return resp.status == expected_status
        except aiohttp.ClientError:
            # -1 means expected client error
            return expected_status == -1


async def _poll_until_ready(port: int, retries: int = 10) -> None:
    url = f"http://127.0.0.1:{port}/health"
    async with aiohttp.ClientSession() as session:
        for _ in range(retries):
            try:
                await session.get(url)
            except aiohttp.ClientError:
                print("Not started yet, sleeping for 1s")
                await asyncio.sleep(1.0)


async def _set_healthy(port: int) -> None:
    url = f"http://127.0.0.1:{port}/set_healthy"
    async with aiohttp.ClientSession() as session, session.get(url) as resp:
        assert resp.status == 200


async def _set_unhealthy(port: int) -> None:
    url = f"http://127.0.0.1:{port}/set_unhealthy"
    async with aiohttp.ClientSession() as session, session.get(url) as resp:
        assert resp.status == 200


# ---------------------------------------------------------------------------
# Core context manager
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def _run_supervisor(args: argparse.Namespace, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dp_sup, "_run_vllm_dp_server", launch_mock_vllm)
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
            await _poll_until_ready(port)

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
        await asyncio.sleep(0.05)

        # E: the supervisor should immediately be not ready
        assert not supervisor.is_ready
        assert await _poll_supervisor_health(503)
        print("/health is 503 --- expected!")

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

    # One of the vLLM servers crashes during startup...,.
    monkeypatch.setenv("MOCK_VLLM_FAIL_PORT", str(_CHILD_PORT_BASE + 1))
    args = _make_args()

    async with _run_supervisor(args, monkeypatch) as (supervisor, _task):
        # A: DPSupervisor is up but children are not healthy yet.
        assert await _poll_supervisor_health(503)
        assert not supervisor.is_ready

        # B: DPSupervisor and all vLLM processes should shut down.
        # If the DPSupervisor does not shut down, it raises a TimeoutError.
        await asyncio.wait_for(_task, timeout=5.0)
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
            await _poll_until_ready(port)

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
        await asyncio.sleep(1.0)
        assert await _poll_supervisor_health(503)
        assert not supervisor.is_ready
        print("/health is 200 --- expected!")

        # E: everything should clean up.
        await asyncio.wait_for(_task, timeout=5.0)
        for p in supervisor._processes:
            assert not p.is_alive()
        print("everything was cleaned up!")


# ---------------------------------------------------------------------------
# Test 5 – Graceful drain on shutdown
# ---------------------------------------------------------------------------


# @pytest.mark.asyncio
# async def test_shutdown_drains_before_force_kill(monkeypatch: pytest.MonkeyPatch):
#     """
#     When shutdown_timeout > 0, the supervisor gives children time to drain
#     before resorting to force-killing them.

#     Children have join_delay < total drain window, so they exit gracefully and
#     kill_process_tree should never be called.
#     """
#     patched_grace = 0.1  # patch CHILD_EXIT_GRACE_S to keep the test fast
#     shutdown_timeout = 0.1
#     join_delay = 0.05  # each child exits well within the drain window

#     # monkeypatch.setattr(dp_supervisor_mod, "CHILD_EXIT_GRACE_S", patched_grace)

#     force_kill_calls: list[int] = []
#     monkeypatch.setattr(
#         dp_supervisor_mod,
#         "kill_process_tree",
#         lambda pid: force_kill_calls.append(pid),
#     )
#     # Intercept os.kill so we don't accidentally signal the test process.
#     monkeypatch.setattr(os, "kill", lambda _pid, _sig: None)

#     args = _make_args(shutdown_timeout=shutdown_timeout)
#     servers = [FakeVllmServer(_CHILD_PORT_BASE + i) for i in range(_N_CHILDREN)]
#     # Give processes a pid so the code reaches the kill_process_tree path if needed.
#     processes = [
#         FakeProcess(
#             name=f"APIServer_DPRank_{i}",
#             pid=os.getpid(),
#             join_delay=join_delay,
#         )
#         for i in range(_N_CHILDREN)
#     ]

#     async with _run_supervisor(args, servers, processes) as (supervisor, task):
#         for s in servers:
#             s.set_healthy(True)
#         await _poll_supervisor_health(200, timeout=5.0)

#         # Trigger graceful shutdown.
#         supervisor._handle_signal(signal.SIGTERM)
#         await asyncio.wait_for(asyncio.shield(task), timeout=5.0)

#     # Children exited within the drain window → no force-kill needed.
#     assert force_kill_calls == [], (
#         f"kill_process_tree was called unexpectedly for pids: {force_kill_calls}"
#     )
