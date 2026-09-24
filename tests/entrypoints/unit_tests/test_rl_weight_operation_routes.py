# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Every weight-operation route records its operation; rejected input never does."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from prometheus_client import CollectorRegistry

from vllm.entrypoints.serve.dev.rlhf import api_router
from vllm.entrypoints.serve.dev.rlhf import metrics as rlhf_metrics
from vllm.entrypoints.serve.dev.rlhf.metrics import WeightOperationMetrics

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]
PREFIX = "vllm:rl_weight_update_"

INIT_INFO = {"init_info": {"names": ["model.weight"]}}
UPDATE_INFO = {"update_info": {"names": ["model.weight"]}}

# (operation, route attribute, request body, extra kwargs, engine method)
OPERATION_ROUTES = [
    (
        "init",
        "init_weight_transfer_engine",
        INIT_INFO,
        {},
        "init_weight_transfer_engine",
    ),
    ("start", "start_weight_update", {}, {}, "start_weight_update"),
    ("start_draft", "start_draft_weight_update", {}, {}, "start_draft_weight_update"),
    ("update", "update_weights", UPDATE_INFO, {}, "update_weights"),
    ("finish", "finish_weight_update", {}, {}, "finish_weight_update"),
    (
        "set_version",
        "update_weight_version",
        {},
        {"new_version": "step-2"},
        "update_weight_version",
    ),
]
OPERATION_IDS = [case[0] for case in OPERATION_ROUTES]

ROUTES_BY_OPERATION = {
    "init": api_router.init_weight_transfer_engine,
    "update": api_router.update_weights,
}


class _Request:
    def __init__(self, engine, body):
        self.app = SimpleNamespace(state=SimpleNamespace(engine_client=engine))
        self._body = body

    async def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class _StubEngine:
    """Blocks inside the engine call until the test releases it."""

    def __init__(self, fail: bool = False):
        self.called: list[str] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.fail = fail

    async def _call(self, name, *args, **kwargs):
        self.called.append(name)
        self.started.set()
        await self.release.wait()
        if self.fail:
            raise RuntimeError(f"{name} failed")

    async def init_weight_transfer_engine(self, *args, **kwargs):
        await self._call("init_weight_transfer_engine", *args, **kwargs)

    async def start_weight_update(self, *args, **kwargs):
        await self._call("start_weight_update", *args, **kwargs)

    async def start_draft_weight_update(self, *args, **kwargs):
        await self._call("start_draft_weight_update", *args, **kwargs)

    async def update_weights(self, *args, **kwargs):
        await self._call("update_weights", *args, **kwargs)

    async def finish_weight_update(self, *args, **kwargs):
        await self._call("finish_weight_update", *args, **kwargs)

    async def update_weight_version(self, *args, **kwargs):
        await self._call("update_weight_version", *args, **kwargs)


class _FinishingEngine:
    """Finishes immediately so the version handshake is the only variable."""

    def __init__(self, fail_version: bool = False):
        self.finished = 0
        self.versions: list[str] = []
        self.fail_version = fail_version

    async def finish_weight_update(self):
        self.finished += 1

    async def update_weight_version(self, new_version):
        if self.fail_version:
            raise RuntimeError("version rejected")
        self.versions.append(new_version)


class _Engine:
    """Records the engine calls it receives without blocking."""

    def __init__(self):
        self.calls: list[str] = []

    async def init_weight_transfer_engine(self, *args, **kwargs):
        self.calls.append("init_weight_transfer_engine")

    async def update_weights(self, *args, **kwargs):
        self.calls.append("update_weights")


class _UnreachableEngine:
    """Fails loudly if a route reaches the engine before validating input."""

    async def update_weights(self, request):  # pragma: no cover - must not run
        raise AssertionError("engine called for a rejected request")

    async def init_weight_transfer_engine(self, request):  # pragma: no cover
        raise AssertionError("engine called for a rejected request")


def _metrics(monkeypatch) -> CollectorRegistry:
    registry = CollectorRegistry()
    metrics = WeightOperationMetrics(registry)
    monkeypatch.setattr(rlhf_metrics, "_metrics", metrics)
    return registry


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation,route_name,body,kwargs,engine_method",
    OPERATION_ROUTES,
    ids=OPERATION_IDS,
)
async def test_route_records_in_flight_then_success(
    monkeypatch, operation, route_name, body, kwargs, engine_method
):
    registry = _metrics(monkeypatch)
    engine = _StubEngine()
    request = _Request(engine, body)
    gauge = PREFIX + "operations_in_flight"
    labels = {"operation": operation}

    task = asyncio.create_task(getattr(api_router, route_name)(request, **kwargs))
    await engine.started.wait()
    assert registry.get_sample_value(gauge, labels) == 1

    engine.release.set()
    response = await task

    assert response.status_code == 200
    assert engine.called == [engine_method]
    assert registry.get_sample_value(gauge, labels) == 0
    assert (
        registry.get_sample_value(
            PREFIX + "operations_total", {**labels, "status": "success"}
        )
        == 1
    )
    assert (
        registry.get_sample_value(PREFIX + "operation_duration_seconds_count", labels)
        == 1
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation,route_name,body,kwargs,engine_method",
    OPERATION_ROUTES,
    ids=OPERATION_IDS,
)
async def test_route_records_error_and_releases_gauge(
    monkeypatch, operation, route_name, body, kwargs, engine_method
):
    registry = _metrics(monkeypatch)
    engine = _StubEngine(fail=True)
    request = _Request(engine, body)
    gauge = PREFIX + "operations_in_flight"
    labels = {"operation": operation}

    task = asyncio.create_task(getattr(api_router, route_name)(request, **kwargs))
    await engine.started.wait()
    engine.release.set()
    with pytest.raises(RuntimeError, match=f"{engine_method} failed"):
        await task

    assert registry.get_sample_value(gauge, labels) == 0
    assert (
        registry.get_sample_value(
            PREFIX + "operations_total", {**labels, "status": "error"}
        )
        == 1
    )
    assert (
        registry.get_sample_value(
            PREFIX + "operations_total", {**labels, "status": "success"}
        )
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route_name,body,operation",
    [
        ("update_weights", {}, "update"),
        ("update_weights", {"update_info": None}, "update"),
        ("init_weight_transfer_engine", {}, "init"),
        ("init_weight_transfer_engine", {"init_info": None}, "init"),
    ],
)
async def test_rejected_input_touches_no_series(
    monkeypatch, route_name, body, operation
):
    registry = _metrics(monkeypatch)
    request = _Request(_UnreachableEngine(), body)

    with pytest.raises(HTTPException) as exc_info:
        await getattr(api_router, route_name)(request)

    assert exc_info.value.status_code == 400
    labels = {"operation": operation}
    for status in ("success", "error"):
        assert (
            registry.get_sample_value(
                PREFIX + "operations_total", {**labels, "status": status}
            )
            is None
        )
    assert registry.get_sample_value(PREFIX + "operations_in_flight", labels) is None
    assert (
        registry.get_sample_value(PREFIX + "operation_duration_seconds_count", labels)
        is None
    )


@pytest.mark.asyncio
async def test_finish_with_version_records_two_operations(monkeypatch):
    """One request, two logical operations, failure attributed to set_version."""
    registry = _metrics(monkeypatch)
    engine = _FinishingEngine(fail_version=True)
    request = _Request(engine, {})

    with pytest.raises(RuntimeError, match="version rejected"):
        await api_router.finish_weight_update(request, weight_version="step-1")

    assert engine.finished == 1
    assert (
        registry.get_sample_value(
            PREFIX + "operations_total", {"operation": "finish", "status": "success"}
        )
        == 1
    )
    assert (
        registry.get_sample_value(
            PREFIX + "operations_total", {"operation": "set_version", "status": "error"}
        )
        == 1
    )
    assert (
        registry.get_sample_value(
            PREFIX + "operations_total", {"operation": "finish", "status": "error"}
        )
        is None
    )
    assert (
        registry.get_sample_value(
            PREFIX + "operations_in_flight", {"operation": "set_version"}
        )
        == 0
    )


@pytest.mark.asyncio
async def test_finish_without_version_records_one_operation(monkeypatch):
    registry = _metrics(monkeypatch)
    engine = _FinishingEngine()
    request = _Request(engine, {})

    response = await api_router.finish_weight_update(request)

    assert response.status_code == 200
    assert engine.finished == 1
    assert engine.versions == []
    assert (
        registry.get_sample_value(
            PREFIX + "operations_total", {"operation": "finish", "status": "success"}
        )
        == 1
    )
    assert (
        registry.get_sample_value(
            PREFIX + "operations_total",
            {"operation": "set_version", "status": "success"},
        )
        is None
    )


def _assert_no_series(registry: CollectorRegistry, operation: str) -> None:
    """Rejected input must not touch any of the three families."""
    for status in ("success", "error"):
        assert (
            registry.get_sample_value(
                PREFIX + "operations_total",
                {"operation": operation, "status": status},
            )
            is None
        )
    assert (
        registry.get_sample_value(
            PREFIX + "operations_in_flight", {"operation": operation}
        )
        is None
    )
    assert (
        registry.get_sample_value(
            PREFIX + "operation_duration_seconds_count", {"operation": operation}
        )
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,body,operation",
    [
        # A valid JSON scalar or array must not reach body.get() (AttributeError).
        ("/init_weight_transfer_engine", 1, "init"),
        ("/init_weight_transfer_engine", [], "init"),
        ("/init_weight_transfer_engine", "x", "init"),
        ("/init_weight_transfer_engine", None, "init"),
        ("/init_weight_transfer_engine", True, "init"),
        ("/update_weights", 1, "update"),
        ("/update_weights", [], "update"),
        ("/update_weights", "x", "update"),
        ("/update_weights", None, "update"),
        ("/update_weights", True, "update"),
    ],
)
async def test_non_object_body_is_rejected_before_recording(
    monkeypatch, path, body, operation
):
    registry = _metrics(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await ROUTES_BY_OPERATION[operation](_Request(_UnreachableEngine(), body))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Request body must be a JSON object"
    _assert_no_series(registry, operation)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,body,operation",
    [
        ("/init_weight_transfer_engine", {"init_info": 1}, "init"),
        ("/init_weight_transfer_engine", {"init_info": []}, "init"),
        ("/init_weight_transfer_engine", {"init_info": [{"rank": 0}]}, "init"),
        ("/init_weight_transfer_engine", {"init_info": "x"}, "init"),
        ("/update_weights", {"update_info": 1}, "update"),
        ("/update_weights", {"update_info": "x"}, "update"),
        ("/update_weights", {"update_info": [{}, None]}, "update"),
        ("/update_weights", {"update_info": [1]}, "update"),
        ("/update_weights", {"update_info": [None]}, "update"),
    ],
)
async def test_invalid_inner_shape_is_rejected_before_recording(
    monkeypatch, path, body, operation
):
    registry = _metrics(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await ROUTES_BY_OPERATION[operation](_Request(_UnreachableEngine(), body))

    assert exc_info.value.status_code == 400
    _assert_no_series(registry, operation)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation,body",
    [
        ("init", {"init_info": {}}),
        ("init", {"init_info": {"rank": 0}}),
        ("update", {"update_info": {}}),
        ("update", {"update_info": [{"rank": 0}]}),
        # An empty list is accepted on both frontends: all([]) is true, as is the
        # Rust iter().all(...) over an empty array.
        ("update", {"update_info": []}),
    ],
)
async def test_valid_payload_shapes_are_recorded(monkeypatch, operation, body):
    registry = _metrics(monkeypatch)
    engine = _Engine()

    response = await ROUTES_BY_OPERATION[operation](_Request(engine, body))

    assert response.status_code == 200
    assert engine.calls, "engine was not called for a valid payload"
    assert (
        registry.get_sample_value(
            PREFIX + "operations_total", {"operation": operation, "status": "success"}
        )
        == 1
    )
    assert (
        registry.get_sample_value(
            PREFIX + "operations_total", {"operation": operation, "status": "error"}
        )
        is None
    )


@pytest.mark.asyncio
async def test_malformed_json_is_rejected_before_recording(monkeypatch):
    registry = _metrics(monkeypatch)
    body = json.JSONDecodeError("bad json", "{}", 0)

    with pytest.raises(HTTPException) as exc_info:
        await api_router.update_weights(_Request(_UnreachableEngine(), body))

    assert exc_info.value.status_code == 400
    _assert_no_series(registry, "update")
