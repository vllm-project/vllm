# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end wiring of the production recorder without multiprocess setup.

`PROMETHEUS_MULTIPROC_DIR` is unset here on purpose: this is what a default
`vllm serve` process looks like, and the metrics have to be registered and
scrapable on the registry that `/metrics` is served from.

The first test drives a real ``FastAPI`` app through the real ASGI stack. It
mounts the dev RLHF router and the instrumentator's ``/metrics`` route with the
production helpers while the recorder does not exist yet, then asserts that the
first real weight operation creates it and that the *same* ``/metrics`` endpoint
exports it. Assertions are before/after deltas so they do not depend on test
order or on private prometheus_client internals.

These tests create the process-wide singleton on the default registry, so the
``_default_registry_cleanup`` fixture removes those collectors again; it also
stops the singleton from being re-created. Other test modules use private
registries, which is why they are unaffected.
"""

import asyncio
import socket
import threading
import time
import urllib.request

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from vllm.entrypoints.serve.dev.rlhf import metrics as rlhf_metrics
from vllm.entrypoints.serve.dev.rlhf.api_router import attach_router as attach_rlhf
from vllm.entrypoints.serve.instrumentator.metrics import (
    attach_router as attach_metrics_endpoint,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]
COUNTER = "vllm:rl_weight_update_operations_total"
DURATION = "vllm:rl_weight_update_operation_duration_seconds"
IN_FLIGHT = "vllm:rl_weight_update_operations_in_flight"
_NAMES = (COUNTER, DURATION, IN_FLIGHT)


class _Engine:
    """Engine stub for the ASGI stack."""

    def __init__(self, fail_start: bool = False, block: bool = False):
        self.calls: list[str] = []
        self.fail_start = fail_start
        self.block = block
        self.started = threading.Event()
        self.release = threading.Event()

    async def start_weight_update(self, *args, **kwargs):
        self.calls.append("start_weight_update")
        if self.block:
            self.started.set()
            # Waits on a threading event so the test can release it from another
            # thread without touching the server's event loop.
            while not self.release.is_set():
                await asyncio.sleep(0.01)
        if self.fail_start:
            raise RuntimeError("engine down")


@pytest.fixture(autouse=True)
def _default_registry_cleanup():
    """Leave the default registry and the singleton as they were found."""
    yield
    for name in _NAMES:
        collector = REGISTRY._names_to_collectors.get(name)
        if collector is not None:
            REGISTRY.unregister(collector)
    rlhf_metrics._metrics = None


def _build_app(engine) -> FastAPI:
    """Mount exactly what production mounts, using the production helpers."""
    app = FastAPI()
    app.state.engine_client = engine
    attach_metrics_endpoint(app)
    attach_rlhf(app)
    return app


def _sample_value(text: str, name: str, labels: str) -> float:
    """Read one sample from scraped exposition text; 0 when absent."""
    target = f"{name}{{{labels}}}"
    for line in text.splitlines():
        if line.startswith(target + " "):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def test_metrics_endpoint_serves_lazily_created_recorder(monkeypatch):
    """The /metrics route exists before the recorder, and serves it afterwards.

    Locks: route mounted -> no collector yet -> first weight operation creates the
    collector -> the same production /metrics endpoint exports it.
    """
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    engine = _Engine()
    labels = 'operation="start",status="success"'

    # Mount /metrics and the RLHF routes first, with no recorder in existence.
    client = TestClient(_build_app(engine))
    assert rlhf_metrics._metrics is None, "the app must not create the recorder"

    before = _sample_value(client.get("/metrics").text, COUNTER, labels)

    response = client.post("/start_weight_update", json={})

    assert response.status_code == 200
    assert engine.calls == ["start_weight_update"]
    assert rlhf_metrics._metrics is not None, "the operation must create the recorder"

    scrape = client.get("/metrics")
    assert scrape.status_code == 200
    assert "# TYPE vllm:rl_weight_update_operations_total counter" in scrape.text

    after = _sample_value(scrape.text, COUNTER, labels)
    assert after - before == 1, "the lazily created counter is not served by /metrics"
    assert _sample_value(scrape.text, IN_FLIGHT, 'operation="start"') == 0
    assert _sample_value(scrape.text, DURATION + "_count", 'operation="start"') >= 1


def test_metrics_endpoint_serves_the_error_path(monkeypatch):
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    client = TestClient(
        _build_app(_Engine(fail_start=True)), raise_server_exceptions=False
    )
    error_labels = 'operation="start",status="error"'
    success_labels = 'operation="start",status="success"'
    before_error = _sample_value(client.get("/metrics").text, COUNTER, error_labels)
    before_success = _sample_value(client.get("/metrics").text, COUNTER, success_labels)

    response = client.post("/start_weight_update", json={})

    assert response.status_code == 500
    scrape = client.get("/metrics")
    assert _sample_value(scrape.text, COUNTER, error_labels) - before_error == 1
    assert _sample_value(scrape.text, COUNTER, success_labels) - before_success == 0


def test_singleton_registers_on_the_served_registry():
    """``registry=None`` makes prometheus_client skip registration entirely."""
    metrics = rlhf_metrics.weight_operation_metrics()
    assert metrics is rlhf_metrics.weight_operation_metrics()

    for name in _NAMES:
        assert name in REGISTRY._names_to_collectors, name


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _LiveServer:
    """A real uvicorn server over a loopback socket."""

    def __init__(self, app: FastAPI):
        uvicorn = pytest.importorskip(
            "uvicorn", reason="uvicorn ships with fastapi[standard]"
        )
        self.port = _free_port()
        config = uvicorn.Config(
            app, host="127.0.0.1", port=self.port, log_level="error"
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self) -> "_LiveServer":
        self._thread.start()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self._server.started:
                return self
            time.sleep(0.02)
        raise AssertionError("server did not start")

    def __exit__(self, *exc) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=30)

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def scrape(self) -> str:
        with urllib.request.urlopen(self.url("/metrics"), timeout=10) as response:
            return response.read().decode()


def test_client_disconnect_does_not_cancel_the_operation(monkeypatch):
    """A dropped socket is not an operation cancellation on this frontend.

    The Python route does not watch the connection, so the engine await keeps
    running and the operation is still recorded as ``success`` - unlike the Rust
    frontend, where a dropped handler future records ``error``. Pinned here so a
    later lifecycle change is a deliberate, visible decision.
    """
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    engine = _Engine(block=True)
    success_labels = 'operation="start",status="success"'
    error_labels = 'operation="start",status="error"'

    with _LiveServer(_build_app(engine)) as server:
        before_success = _sample_value(server.scrape(), COUNTER, success_labels)
        before_error = _sample_value(server.scrape(), COUNTER, error_labels)

        # Send the request, then drop the socket while the engine call is running.
        conn = socket.create_connection(("127.0.0.1", server.port), timeout=10)
        body = b"{}"
        conn.sendall(
            b"POST /start_weight_update HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n".encode()
            + b"Connection: close\r\n\r\n"
            + body
        )
        assert engine.started.wait(timeout=30), "engine call never started"
        conn.close()

        # The operation outlives the disconnect and completes normally.
        engine.release.set()

        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            scrape = server.scrape()
            if _sample_value(scrape, COUNTER, success_labels) - before_success == 1:
                break
            time.sleep(0.05)

        assert _sample_value(scrape, COUNTER, success_labels) - before_success == 1, (
            "the operation should still be recorded as success after a disconnect"
        )
        assert _sample_value(scrape, COUNTER, error_labels) - before_error == 0
        assert _sample_value(scrape, IN_FLIGHT, 'operation="start"') == 0
