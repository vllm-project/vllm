# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the optional fastapi-guard middleware wiring."""

import asyncio
import itertools

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from vllm.entrypoints.serve.middleware.guard import (
    init_guard_middleware,
    is_guard_available,
)

# Rate limiting and ban state in fastapi-guard are process-wide, so every
# test drives its own TEST-NET client IP to stay hermetic.
_IP = itertools.count(1)


def _unique_ip() -> str:
    n = next(_IP)
    return f"198.51.{n // 250}.{(n % 250) + 1}"


def _app() -> FastAPI:
    app = FastAPI()

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    init_guard_middleware(app)
    return app


def _run(scenario, client_ip):
    async def runner():
        transport = ASGITransport(app=_app(), client=(client_ip, 50000))
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            await scenario(client)

    asyncio.run(runner())


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_GUARD_ENABLED", raising=False)
    app = FastAPI()
    init_guard_middleware(app)
    assert len(app.user_middleware) == 0


def test_enabled_without_package_raises(monkeypatch):
    if is_guard_available():
        pytest.skip("fastapi-guard is installed")
    monkeypatch.setenv("VLLM_GUARD_ENABLED", "True")
    with pytest.raises(RuntimeError, match="vllm\\[guard\\]"):
        init_guard_middleware(FastAPI())


pytest.importorskip("guard")


def test_blocked_ip_is_rejected(monkeypatch):
    blocked = _unique_ip()
    monkeypatch.setenv("VLLM_GUARD_ENABLED", "True")
    monkeypatch.setenv("VLLM_GUARD_BLOCKED_IPS", blocked)

    async def scenario(client):
        response = await client.get("/ping")
        assert response.status_code == 403

    _run(scenario, blocked)


def test_rate_limit_returns_429(monkeypatch):
    client_ip = _unique_ip()
    monkeypatch.setenv("VLLM_GUARD_ENABLED", "True")
    monkeypatch.setenv("VLLM_GUARD_RATE_LIMIT", "2")
    monkeypatch.setenv("VLLM_GUARD_RATE_LIMIT_WINDOW", "60")

    async def scenario(client):
        assert (await client.get("/ping")).status_code == 200
        assert (await client.get("/ping")).status_code == 200
        assert (await client.get("/ping")).status_code == 429

    _run(scenario, client_ip)


def test_passive_mode_never_blocks(monkeypatch):
    blocked = _unique_ip()
    monkeypatch.setenv("VLLM_GUARD_ENABLED", "True")
    monkeypatch.setenv("VLLM_GUARD_PASSIVE_MODE", "True")
    monkeypatch.setenv("VLLM_GUARD_BLOCKED_IPS", blocked)

    async def scenario(client):
        response = await client.get("/ping")
        assert response.status_code == 200

    _run(scenario, blocked)


def test_zero_trust_and_safe_defaults(monkeypatch):
    """No proxy is trusted unless explicitly listed, bodies are not
    signature-scanned by default (inference input is data, not commands
    the server executes), and rate-limit violations feed the ban engine
    (the advertised auto-ban behavior)."""
    from vllm.entrypoints.serve.middleware.guard import _build_guard_config

    monkeypatch.delenv("VLLM_GUARD_TRUSTED_PROXIES", raising=False)
    monkeypatch.delenv("VLLM_GUARD_SCAN_BODY", raising=False)
    monkeypatch.delenv("VLLM_GUARD_RATE_LIMIT_AUTO_BAN", raising=False)
    config = _build_guard_config()
    assert config.trusted_proxies == ()
    assert config.detection_scan_body is False
    assert config.enable_rate_limit_auto_ban is True

    monkeypatch.setenv("VLLM_GUARD_TRUSTED_PROXIES", "10.0.0.1,10.0.0.2")
    monkeypatch.setenv("VLLM_GUARD_SCAN_BODY", "True")
    monkeypatch.setenv("VLLM_GUARD_RATE_LIMIT_AUTO_BAN", "0")
    config = _build_guard_config()
    assert config.trusted_proxies == ("10.0.0.1", "10.0.0.2")
    assert config.detection_scan_body is True
    assert config.enable_rate_limit_auto_ban is False
