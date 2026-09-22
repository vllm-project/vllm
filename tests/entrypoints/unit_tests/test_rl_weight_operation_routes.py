# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from prometheus_client import CollectorRegistry

from vllm.entrypoints.serve.dev.rlhf import api_router
from vllm.entrypoints.serve.dev.rlhf.metrics import WeightOperationMetrics

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]
PREFIX = "vllm:rl_weight_update_"


class _Request:
    def __init__(self, engine, body):
        self.app = SimpleNamespace(state=SimpleNamespace(engine_client=engine))
        self._body = body

    async def json(self):
        return self._body


class _BlockingEngine:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def update_weights(self, request):
        self.started.set()
        await self.release.wait()


class _RecordingEngine:
    def __init__(self, fail_version=False):
        self.finished = 0
        self.versions = []
        self.fail_version = fail_version

    async def finish_weight_update(self):
        self.finished += 1

    async def update_weight_version(self, new_version):
        if self.fail_version:
            raise RuntimeError("version rejected")
        self.versions.append(new_version)


class _UnreachableEngine:
    """Fails loudly if the route reaches the engine before validating input."""

    async def update_weights(self, request):  # pragma: no cover - must not run
        raise AssertionError("engine called for a rejected request")


def _metrics(monkeypatch) -> tuple[WeightOperationMetrics, CollectorRegistry]:
    registry = CollectorRegistry()
    metrics = WeightOperationMetrics(registry)
    monkeypatch.setattr(api_router, "_weight_metrics", lambda: metrics)
    return metrics, registry


@pytest.mark.asyncio
async def test_update_weights_route_records_in_flight_and_success(monkeypatch):
    _, registry = _metrics(monkeypatch)
    engine = _BlockingEngine()
    request = _Request(engine, {"update_info": {"names": ["model.weight"]}})
    labels = {"operation": "update"}

    task = asyncio.create_task(api_router.update_weights(request))
    await engine.started.wait()
    assert registry.get_sample_value(PREFIX + "requests_in_flight", labels) == 1

    engine.release.set()
    response = await task
    assert response.status_code == 200
    assert registry.get_sample_value(PREFIX + "requests_in_flight", labels) == 0
    assert (
        registry.get_sample_value(
            PREFIX + "requests_total", {**labels, "status": "success"}
        )
        == 1
    )
    assert (
        registry.get_sample_value(PREFIX + "request_duration_seconds_count", labels)
        == 1
    )


@pytest.mark.asyncio
async def test_update_weights_route_rejects_before_recording(monkeypatch):
    _, registry = _metrics(monkeypatch)
    request = _Request(_UnreachableEngine(), {})

    with pytest.raises(HTTPException) as exc_info:
        await api_router.update_weights(request)

    assert exc_info.value.status_code == 400
    # A rejected request never enters the recorder: no counter and no gauge.
    assert (
        registry.get_sample_value(
            PREFIX + "requests_total",
            {"operation": "update", "status": "success"},
        )
        is None
    )
    assert (
        registry.get_sample_value(
            PREFIX + "requests_in_flight", {"operation": "update"}
        )
        is None
    )


@pytest.mark.asyncio
async def test_finish_records_success_even_when_version_fails(monkeypatch):
    _, registry = _metrics(monkeypatch)
    engine = _RecordingEngine(fail_version=True)
    request = _Request(engine, {})

    with pytest.raises(RuntimeError, match="version rejected"):
        await api_router.finish_weight_update(request, weight_version="step-1")

    assert engine.finished == 1
    assert (
        registry.get_sample_value(
            PREFIX + "requests_total", {"operation": "finish", "status": "success"}
        )
        == 1
    )
    assert (
        registry.get_sample_value(
            PREFIX + "requests_total", {"operation": "set_version", "status": "error"}
        )
        == 1
    )
    assert (
        registry.get_sample_value(
            PREFIX + "requests_total", {"operation": "set_version", "status": "success"}
        )
        is None
    )


@pytest.mark.asyncio
async def test_finish_without_version_does_not_record_set_version(monkeypatch):
    _, registry = _metrics(monkeypatch)
    engine = _RecordingEngine()
    request = _Request(engine, {})

    response = await api_router.finish_weight_update(request)

    assert response.status_code == 200
    assert engine.finished == 1
    assert engine.versions == []
    assert (
        registry.get_sample_value(
            PREFIX + "requests_total", {"operation": "finish", "status": "success"}
        )
        == 1
    )
    assert (
        registry.get_sample_value(
            PREFIX + "requests_total", {"operation": "set_version", "status": "success"}
        )
        is None
    )


@pytest.mark.asyncio
async def test_update_weight_version_route_is_recorded(monkeypatch):
    _, registry = _metrics(monkeypatch)
    engine = _RecordingEngine()
    request = _Request(engine, {})

    response = await api_router.update_weight_version(request, new_version="step-2")

    assert response.status_code == 200
    assert engine.versions == ["step-2"]
    assert (
        registry.get_sample_value(
            PREFIX + "requests_total", {"operation": "set_version", "status": "success"}
        )
        == 1
    )
    assert (
        registry.get_sample_value(
            PREFIX + "requests_in_flight", {"operation": "set_version"}
        )
        == 0
    )
