# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``HeartbeatService``.

These tests mock ``httpx.AsyncClient`` so the real network is never
touched, and verify that ``send_heartbeat`` builds the expected query
parameters from the configured app host/port and target nodes.
"""

# Standard
import asyncio
import json

# Third Party
import pytest

# First Party
from lmcache.lmcache_frontend.heartbeat import HeartbeatService


class _FakeResponse:
    def __init__(self, status_code: int = 200, content: bytes = b""):
        self.status_code = status_code
        self.content = content

    def raise_for_status(self):  # noqa: D401 - match httpx API
        if self.status_code >= 400:
            raise RuntimeError("boom")


class _FakeAsyncClient:
    """Captures the last ``get`` call for later assertions."""

    last_url: str | None = None
    last_params: dict | None = None
    version_response: _FakeResponse = _FakeResponse(200, b"")
    heartbeat_response: _FakeResponse = _FakeResponse(200, b"ok")

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, params=None):
        type(self).last_url = url
        type(self).last_params = params
        # Distinguish version probe vs heartbeat post
        if url.endswith("/version"):
            return type(self).version_response
        return type(self).heartbeat_response


@pytest.fixture
def patched_httpx(monkeypatch):
    """Patch ``httpx.AsyncClient`` used inside ``heartbeat`` module."""
    # First Party
    from lmcache.lmcache_frontend import heartbeat as hb_mod

    _FakeAsyncClient.last_url = None
    _FakeAsyncClient.last_params = None
    _FakeAsyncClient.version_response = _FakeResponse(200, b"")
    _FakeAsyncClient.heartbeat_response = _FakeResponse(200, b"ok")
    monkeypatch.setattr(hb_mod.httpx, "AsyncClient", _FakeAsyncClient)
    # Deterministic IP so assertions are stable
    monkeypatch.setattr(HeartbeatService, "get_local_ip", lambda self: "10.0.0.42")
    return _FakeAsyncClient


def test_send_heartbeat_builds_expected_params(patched_httpx):
    svc = HeartbeatService()
    svc.set_app_config(
        host="0.0.0.0",
        port=8085,
        target_nodes=[
            {
                "name": "proxy1",
                "host": "127.0.0.1",
                "port": "8001",
                "nodes": [
                    {"name": "n1", "host": "127.0.0.1", "port": "9001"},
                    {"name": "n2", "host": "127.0.0.1", "port": "9002"},
                ],
            },
        ],
    )

    ok = asyncio.run(svc.send_heartbeat("http://disc.example/heartbeat"))
    assert ok is True

    params = patched_httpx.last_params
    assert params is not None
    assert params["api_address"] == "http://10.0.0.42:8085"
    assert params["pid"] > 0
    # total children across proxies
    other = json.loads(params["other_info"])
    assert other["nodes_count"] == 2
    # version fallback when /version returns empty body
    assert params["version"] == "1.0.0"


def test_send_heartbeat_returns_false_on_http_error(patched_httpx):
    patched_httpx.heartbeat_response = _FakeResponse(500, b"err")

    svc = HeartbeatService()
    svc.set_app_config(host="0.0.0.0", port=8085, target_nodes=[])

    ok = asyncio.run(svc.send_heartbeat("http://disc.example/heartbeat"))
    assert ok is False


def test_send_heartbeat_uses_version_from_target_nodes(patched_httpx):
    patched_httpx.version_response = _FakeResponse(200, b'"1.2.3"')

    svc = HeartbeatService()
    svc.set_app_config(
        host="0.0.0.0",
        port=8085,
        target_nodes=[
            {
                "name": "proxy1",
                "host": "127.0.0.1",
                "port": "8001",
                "nodes": [
                    {"name": "n1", "host": "127.0.0.1", "port": "9001"},
                ],
            },
        ],
    )

    ok = asyncio.run(svc.send_heartbeat("http://disc.example/heartbeat"))
    assert ok is True
    assert patched_httpx.last_params["version"] == "1.2.3"


def test_status_reports_running_flag():
    svc = HeartbeatService()
    status = svc.status()
    assert status["running"] in (False, None)
    assert "startup_time" in status
    assert "current_time" in status


def test_send_heartbeat_uses_report_host_when_set(patched_httpx, monkeypatch):
    """Explicit report_host must bypass get_local_ip() auto-detection."""
    # Poison get_local_ip so the test fails loudly if it is ever called.
    monkeypatch.setattr(
        HeartbeatService,
        "get_local_ip",
        lambda self: pytest.fail("get_local_ip() must not be called"),
    )

    svc = HeartbeatService()
    svc.set_app_config(
        host="0.0.0.0",
        port=8085,
        target_nodes=[],
        report_host="127.0.0.1",
    )

    ok = asyncio.run(svc.send_heartbeat("http://disc.example/heartbeat"))
    assert ok is True
    assert patched_httpx.last_params["api_address"] == "http://127.0.0.1:8085"


def test_empty_report_host_falls_back_to_auto_detect(patched_httpx):
    """Empty string report_host is treated as unset and auto-detects."""
    svc = HeartbeatService()
    svc.set_app_config(
        host="0.0.0.0",
        port=8085,
        target_nodes=[],
        report_host="",
    )

    ok = asyncio.run(svc.send_heartbeat("http://disc.example/heartbeat"))
    assert ok is True
    assert patched_httpx.last_params["api_address"] == "http://10.0.0.42:8085"
