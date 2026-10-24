# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A graceful shutdown stops accepting new connections before anything drains:
the listener closes, requests already accepted finish, and the launcher can ask
its API server workers to do the same with SIGUSR1."""

import asyncio
import os
import signal
import socket
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI

from vllm.entrypoints.launchers.launcher import _NOT_ACCEPTING, serve_http
from vllm.v1.utils import APIServerProcessManager


def _bound_socket() -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    return sock


async def _start_server(peer_loads):
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/slow")
    async def slow():
        await asyncio.sleep(0.5)
        return {"ok": True}

    app.state.engine_client = None
    sock = _bound_socket()
    port = sock.getsockname()[1]
    task = asyncio.create_task(
        serve_http(
            app,
            sock,
            peer_loads=peer_loads,
            host="127.0.0.1",
            port=port,
            log_level="warning",
        )
    )
    url = f"http://127.0.0.1:{port}"
    async with httpx.AsyncClient() as client:
        for _ in range(100):
            try:
                if (await client.get(url + "/health")).status_code == 200:
                    break
            except httpx.ConnectError:
                await asyncio.sleep(0.05)
        else:
            task.cancel()
            pytest.fail("server did not start")
    return app, task, url


async def _stop_server(app, task) -> None:
    app.state.server.should_exit = True
    shutdown_coro = await asyncio.wait_for(task, timeout=10)
    shutdown_coro.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "use_peer_loads", [False, True], ids=["uvicorn_accept", "shared_socket_accept"]
)
async def test_stop_accepting_refuses_new_connections_and_finishes_in_flight(
    use_peer_loads,
):
    loads = [0, 0]
    app, task, url = await _start_server((loads, 0) if use_peer_loads else None)
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            in_flight = asyncio.create_task(client.get(url + "/slow"))
            await asyncio.sleep(0.1)  # the request is being served

            # What the multi-API-server launcher sends to its workers.
            os.kill(os.getpid(), signal.SIGUSR1)
            await asyncio.sleep(0.1)

            with pytest.raises(httpx.ConnectError):
                await httpx.AsyncClient(timeout=2).get(url + "/health")
            # The request accepted before the stop still completes ...
            assert (await in_flight).status_code == 200
            # ... but its keep-alive connection is closed afterwards, so the
            # same client's next request needs a new connection and is refused.
            with pytest.raises((httpx.ConnectError, httpx.RemoteProtocolError)):
                await client.get(url + "/health")
        if use_peer_loads:
            assert loads[0] == _NOT_ACCEPTING
    finally:
        await _stop_server(app, task)


@pytest.mark.asyncio
async def test_stop_accepting_is_idempotent_and_shutdown_still_clean():
    app, task, url = await _start_server(None)
    server = app.state.server
    server.stop_accepting()
    server.stop_accepting()
    with pytest.raises(httpx.ConnectError):
        await httpx.AsyncClient(timeout=2).get(url + "/health")
    await _stop_server(app, task)


def test_process_manager_stop_accepting_signals_live_workers():
    manager = APIServerProcessManager.__new__(APIServerProcessManager)
    manager.processes = [
        SimpleNamespace(pid=101, is_alive=lambda: True),
        SimpleNamespace(pid=102, is_alive=lambda: False),  # already exited
        SimpleNamespace(pid=None, is_alive=lambda: True),  # never started
        SimpleNamespace(pid=104, is_alive=lambda: True),
    ]
    with patch("vllm.v1.utils.os.kill") as kill:
        manager.stop_accepting()
    assert sorted(c.args for c in kill.call_args_list) == [
        (101, signal.SIGUSR1),
        (104, signal.SIGUSR1),
    ]
