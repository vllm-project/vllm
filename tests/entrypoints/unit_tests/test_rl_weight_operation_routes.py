# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Every weight-operation route records its operation; rejected input never does."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from prometheus_client import CollectorRegistry

from vllm.entrypoints.serve.dev.rlhf import api_router
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


class _Request:
    def __init__(self, engine, body):
        self.app = SimpleNamespace(state=SimpleNamespace(engine_client=engine))
        self._body = body

    async def json(self):
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


class _UnreachableEngine:
    """Fails loudly if a route reaches the engine before validating input."""

    async def update_weights(self, request):  # pragma: no cover - must not run
        raise AssertionError("engine called for a rejected request")

    async def init_weight_transfer_engine(self, request):  # pragma: no cover
        raise AssertionError("engine called for a rejected request")


def _metrics(monkeypatch) -> CollectorRegistry:
    registry = CollectorRegistry()
    metrics = WeightOperationMetrics(registry)
    monkeypatch.setattr(api_router, "_weight_metrics", lambda: metrics)
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
