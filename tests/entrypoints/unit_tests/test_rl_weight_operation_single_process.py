# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end wiring of the production recorder without multiprocess setup.

`PROMETHEUS_MULTIPROC_DIR` is unset here on purpose: this is what a default
`vllm serve` process looks like, and the metrics have to be registered and
scrapable on the registry that `/metrics` is served from.

The central test drives a real ``FastAPI`` app through the real ASGI stack. It
mounts the dev RLHF router and the instrumentator's ``/metrics`` route with the
production helpers while the recorder does not exist yet, then asserts that the
first real weight operation creates it and that the *same* ``/metrics`` endpoint
exports it. Assertions are before/after deltas so they do not depend on test
order or on private prometheus_client internals.
"""

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from vllm.entrypoints.serve.dev.rlhf import api_router
from vllm.entrypoints.serve.dev.rlhf import metrics as rlhf_metrics
from vllm.entrypoints.serve.dev.rlhf.api_router import attach_router as attach_rlhf
from vllm.entrypoints.serve.instrumentator.metrics import (
    attach_router as attach_metrics_endpoint,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]
COUNTER = "vllm:rl_weight_update_operations_total"
DURATION = "vllm:rl_weight_update_operation_duration_seconds"
IN_FLIGHT = "vllm:rl_weight_update_operations_in_flight"

ROUTES = {
    "init": api_router.init_weight_transfer_engine,
    "update": api_router.update_weights,
    "start": api_router.start_weight_update,
}


class _Engine:
    """Engine stub for both the ASGI stack and the direct route calls."""

    def __init__(self, fail_start: bool = False):
        self.calls: list[str] = []
        self.fail_start = fail_start

    async def update_weights(self, *args, **kwargs):
        self.calls.append("update_weights")

    async def init_weight_transfer_engine(self, *args, **kwargs):
        self.calls.append("init_weight_transfer_engine")

    async def start_weight_update(self, *args, **kwargs):
        self.calls.append("start_weight_update")
        if self.fail_start:
            raise RuntimeError("engine down")


class _RejectingEngine(_Engine):
    async def update_weights(self, *args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("engine reached for an invalid payload")

    async def init_weight_transfer_engine(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("engine reached for an invalid payload")


class _Request:
    def __init__(self, engine, body):
        self.app = SimpleNamespace(state=SimpleNamespace(engine_client=engine))
        self._body = body

    async def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


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


def _snapshot(operation: str) -> tuple[float, float]:
    """Current (success, error) counter values for one operation."""
    metrics = rlhf_metrics.weight_operation_metrics()
    success = metrics.operations.labels(operation=operation, status="success")
    error = metrics.operations.labels(operation=operation, status="error")
    return success._value.get(), error._value.get()


# --------------------------------------------------------------------------- #
# Real HTTP path: FastAPI app -> route -> production singleton -> GET /metrics
# --------------------------------------------------------------------------- #
def test_metrics_endpoint_serves_lazily_created_recorder(monkeypatch):
    """The /metrics route exists before the recorder, and serves it afterwards.

    Locks: route mounted -> no collector yet -> first weight operation creates the
    collector -> the same production /metrics endpoint exports it.
    """
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    monkeypatch.setattr(rlhf_metrics, "_metrics", None)
    engine = _Engine()
    labels = 'operation="start",status="success"'
    duration = 'operation="start"'

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
    assert _sample_value(scrape.text, IN_FLIGHT, duration) == 0
    assert _sample_value(scrape.text, DURATION + "_count", duration) >= 1


def test_metrics_endpoint_serves_the_error_path(monkeypatch):
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    monkeypatch.setattr(rlhf_metrics, "_metrics", None)
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


def test_singleton_registers_on_the_served_registry(monkeypatch):
    """``registry=None`` makes prometheus_client skip registration entirely."""
    monkeypatch.setattr(rlhf_metrics, "_metrics", None)

    metrics = rlhf_metrics.weight_operation_metrics()
    assert metrics is rlhf_metrics.weight_operation_metrics()

    for name in (COUNTER, DURATION, IN_FLIGHT):
        assert name in REGISTRY._names_to_collectors, name


# --------------------------------------------------------------------------- #
# Payload shape boundary on the direct routes
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,body",
    [
        ("/init_weight_transfer_engine", 1),
        ("/init_weight_transfer_engine", []),
        ("/init_weight_transfer_engine", "x"),
        ("/init_weight_transfer_engine", None),
        ("/init_weight_transfer_engine", True),
        ("/update_weights", 1),
        ("/update_weights", []),
        ("/update_weights", "x"),
        ("/update_weights", None),
        ("/update_weights", True),
    ],
)
async def test_non_object_top_level_body_is_rejected(path, body):
    """A valid JSON scalar/array must not reach ``body.get`` (AttributeError)."""
    operation = "init" if "init" in path else "update"
    before = _snapshot(operation)

    with pytest.raises(HTTPException) as exc_info:
        await ROUTES[operation](_Request(_RejectingEngine(), body))

    assert exc_info.value.status_code == 400
    assert "JSON object" in exc_info.value.detail
    assert _snapshot(operation) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,body",
    [
        # missing / null
        ("/init_weight_transfer_engine", {}),
        ("/init_weight_transfer_engine", {"init_info": None}),
        ("/update_weights", {}),
        ("/update_weights", {"update_info": None}),
        # wrong inner shape: rejected by the dataclasses and by the Rust frontend
        ("/init_weight_transfer_engine", {"init_info": 1}),
        ("/init_weight_transfer_engine", {"init_info": []}),
        ("/init_weight_transfer_engine", {"init_info": [{"rank": 0}]}),
        ("/init_weight_transfer_engine", {"init_info": "x"}),
        ("/update_weights", {"update_info": 1}),
        ("/update_weights", {"update_info": "x"}),
        ("/update_weights", {"update_info": [{}, None]}),
        ("/update_weights", {"update_info": [1]}),
        ("/update_weights", {"update_info": [None]}),
    ],
)
async def test_invalid_payload_records_nothing(path, body):
    operation = "init" if "init" in path else "update"
    before = _snapshot(operation)

    with pytest.raises(HTTPException) as exc_info:
        await ROUTES[operation](_Request(_RejectingEngine(), body))

    assert exc_info.value.status_code == 400
    # The recorder must not have been entered, so neither status moved.
    assert _snapshot(operation) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,body",
    [
        ("/update_weights", {"update_info": {}}),
        ("/update_weights", {"update_info": [{"rank": 0}]}),
        # An empty list is accepted on both frontends: all([]) is true, as is the
        # Rust iter().all(...) over an empty array.
        ("/update_weights", {"update_info": []}),
        ("/init_weight_transfer_engine", {"init_info": {}}),
        ("/init_weight_transfer_engine", {"init_info": {"rank": 0}}),
    ],
)
async def test_valid_payload_shapes_are_recorded(path, body):
    operation = "init" if "init" in path else "update"
    engine = _Engine()
    before_success, _ = _snapshot(operation)

    response = await ROUTES[operation](_Request(engine, body))

    assert response.status_code == 200
    assert engine.calls, "engine was not called for a valid payload"
    after_success, after_error = _snapshot(operation)
    assert after_success - before_success == 1
    assert after_error == 0


@pytest.mark.asyncio
async def test_malformed_json_is_rejected_without_recording():
    request = _Request(_RejectingEngine(), json.JSONDecodeError("bad", "{}", 0))
    before = _snapshot("update")

    with pytest.raises(HTTPException) as exc_info:
        await ROUTES["update"](request)

    assert exc_info.value.status_code == 400
    assert _snapshot("update") == before
