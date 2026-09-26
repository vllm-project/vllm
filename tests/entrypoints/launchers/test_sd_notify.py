# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The systemd readiness notification of the API server launcher."""

import asyncio
import socket
import sys

import httpx
import pytest
from fastapi import FastAPI

from vllm.entrypoints.launchers.launcher import serve_http
from vllm.entrypoints.launchers.utils.sd_notify import NOTIFY_SOCKET_ENV, sd_notify
from vllm.utils.network_utils import get_open_port


@pytest.fixture
def notify_socket(tmp_path, monkeypatch):
    """A bound notify socket, exposed to the code under test like systemd
    would (``NOTIFY_SOCKET``). Yields the receiving end."""
    path = str(tmp_path / "notify")
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
        sock.bind(path)
        sock.settimeout(10)
        monkeypatch.setenv(NOTIFY_SOCKET_ENV, path)
        yield sock


def test_noop_without_notify_socket(monkeypatch):
    monkeypatch.delenv(NOTIFY_SOCKET_ENV, raising=False)
    assert sd_notify("READY=1") is False


def test_sends_state_to_notify_socket(notify_socket):
    assert sd_notify("READY=1") is True
    assert notify_socket.recv(64) == b"READY=1"


@pytest.mark.skipif(sys.platform != "linux", reason="abstract sockets are Linux only")
def test_abstract_notify_socket(monkeypatch):
    name = f"@vllm-test-notify-{get_open_port()}"
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
        sock.bind("\0" + name[1:])
        sock.settimeout(10)
        monkeypatch.setenv(NOTIFY_SOCKET_ENV, name)
        assert sd_notify("READY=1") is True
        assert sock.recv(64) == b"READY=1"


def test_unreachable_notify_socket_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.setenv(NOTIFY_SOCKET_ENV, str(tmp_path / "nobody-listens"))
    assert sd_notify("READY=1") is False


@pytest.mark.asyncio
async def test_serve_http_notifies_ready_once_serving(notify_socket):
    """READY=1 arrives only once the HTTP server answers requests."""
    app = FastAPI()
    app.state.engine_client = None  # no engine, hence no watchdog
    port = get_open_port()
    serving = asyncio.create_task(
        serve_http(app, sock=None, host="127.0.0.1", port=port, log_level="warning")
    )
    try:
        loop = asyncio.get_running_loop()
        state = await loop.run_in_executor(None, notify_socket.recv, 64)
        assert state == b"READY=1"
        # The unit is "started" now: the server must already be reachable.
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://127.0.0.1:{port}/openapi.json")
        assert response.status_code == 200
    finally:
        app.state.server.should_exit = True
        shutdown = await asyncio.wait_for(serving, timeout=10)
        await shutdown
